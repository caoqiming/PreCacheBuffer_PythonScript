# -*- coding: utf-8 -*-
import time
import json
import threading
import socket
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import util.Glimmer_send
import util.Glimmer_op

from CF_Autoencoder import CFA #具体使用的模型


class FLserver:
    #核心部分--控制
    command_id=set()
    isroot=False
    running=False
    state_lock = threading.Lock()
    state=0#工作状态 0未初始化 1正常工作 2正在训练/等待下层聚合 3向上层继续聚合/完成聚合 4发送新模型、更新模型
    flag={}
    flag["glimmer_op_update"]=False
    optimizer="SGD"
    glimmer_op=None #优化器的实例
    FL_round=0      #进行的FL轮数，用来判断是否需要更新glimmer_op
    #核心部分--缓存
    local_ip=None
    cloud_server=None #云服务器，向云服务器发送状态信息
    up_server=None
    n_down_server=None
    down_server=None
    is_child_participant=[]#0未初始化不知道是否参与，1已经初始化，-1不参与
    received_weight=[]#接收各个ip下数据的权重，在发送最后一个数据时发送
    #使用的模型
    mymodel=None
    #参数部分
    wait_time=3    #等待时间,接收数据间隔超过这个时间则认为丢包
    port=None
    local_weight=1#本地权重，如果不参与训练则为0
    isparticipant=True#是否参与本轮训练
    FSVRG_h=0.01
    FSVRG_m=100
    SGD_h=0.1
    SGD_b=1000


    #---------------主要函数、控制相关-----------------
    def __init__(self,config_file):
        #获取计算机名称
        hostname=socket.gethostname()
        #获取本机IP
        self.local_ip=socket.gethostbyname(hostname)
        self.G_s=util.Glimmer_send.Glimmer_send((self.local_ip,self.port),self)
        self.local_model=self.create_model()
        self.temp_model=self.create_model()

        with open(config_file,'r',encoding='utf-8')as fp:
            json_data = json.load(fp)
            print('config file:',json_data)
            self.up_server=json_data['up_server']
            self.down_server=json_data['down_server']
            self.port=json_data['port']
            self.send_bytes_once=json_data['send_bytes_once']
            self.data_id=json_data['data_id']
            self.n_down_server=len(self.down_server)
            if "local_ip" in json_data:
                self.local_ip=json_data["local_ip"]
            if "wait_time" in json_data:
                self.wait_time=json_data["wait_time"]
            if "isparticipant" in json_data:
                self.isparticipant=json_data["isparticipant"]
            if "send_speed" in json_data:
                self.G_s.send_speed=json_data["send_speed"]
            if "cloud_server" in json_data:
                self.cloud_server=json_data["cloud_server"]
            if "isdebug" in json_data:
                self.isdebug=json_data["isdebug"]
            if "optimizer" in json_data:
                self.optimizer=json_data["optimizer"]
            if json_data['local_model']==1:
                self.local_model_read()
            if self.up_server=="":
                self.isroot=True
        self.clear_buffer()
        self.G_s.local_ip=self.local_ip
        self.mymodel=CFA(6040,3952,29,18,64,128)

        return
    def listen(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.bind(("", self.port)) 
        while(self.running):
            server_socket.settimeout(3)  #每3秒钟检测是否关闭
            try:
                receive_data, client = server_socket.recvfrom(1400)
                #print("来自客户端%s,发送的%s\n" % (client, receive_data))  #打印接收的内容
                josn_end=receive_data.index(b"}")+1
                receive_data2=receive_data[josn_end:]
                json_data = json.loads(receive_data[:josn_end].decode('utf-8'))
                if "feedback_id" in json_data:
                    if not self.feedback(json_data):
                        json_data={}#已经处理过的任务就跳过
                if "order" in json_data:
                    #-------------<Glimmer_send>---------------
                    if (json_data["order"]=="data"):
                        self.G_s.receive_data(json_data,receive_data2)
                    elif (json_data["order"]=="send_ready"):
                        th_receive=threading.Thread(target=self.G_s.start_receive, args=(json_data,))
                        th_receive.start()
                    elif(json_data["order"]=="start_send_data"):
                        self.G_s.flag_start_to_send=True
                    elif(json_data["order"]=="resend"):
                        self.G_s.resend_id_queue.put(json_data["resend_id_list"])
                    elif(json_data["order"]=="receive_complete"):
                        self.G_s.flag_receive_complete=True
                    #-------------</Glimmer_send>---------------
                    elif (json_data["order"]=="call_back"):
                        th_callback=threading.Thread(target=eval("self.{}".format(json_data["call_back"])), args=(json_data,))
                        th_callback.start()
                    #==================================================================
                    elif (json_data['order']=="aggregation_request"):#开始聚合的请求
                        if(self.isroot):
                            self.start_FL()
                        else:
                            self.command_send((self.up_server,self.port),json.dumps(json_data))#不是根节点就向上传递请求
                    elif(json_data['order']=="aggregation_confirm"):
                        self.start_FL()
                    elif(json_data['order']=="test_local_model"):#外部命令，测试模型准确度
                        self.test_local_model()
                    elif(json_data['order']=="read_model"):#外部命令，读取模型
                        self.local_model_read()
                    elif(json_data['order']=="reset_model"):#外部命令，重置模型
                        self.local_model_reset()
                    elif(json_data['order']=="save_model"):#外部命令，保存模型
                        self.local_model_save()
                    elif(json_data['order']=="change_topology"):#外部命令,修改拓扑结构
                        self.change_topology(json_data)
                    elif(json_data['order']=="update_part1"):#外部命令,更新推理模型
                        th_update_part1=threading.Thread(target=self.update_part1, args=("localhost",))
                        th_update_part1.start()
                    elif(json_data['order']=="change_parameter"):#外部命令,更改参数
                        self.change_parameter(json_data)
                    elif(json_data['order']=="send_op_data"):#上传优化器所需数据
                        th_send_op_data=threading.Thread(target=self.send_op_data, args=(json_data,))
                        th_send_op_data.start()
            except socket.timeout:
                pass
        print("stop listen")
        return
    def start(self):
        self.state_lock.acquire()
        self.state=1
        self.state_lock.release()
        self.running=True
        self.th_listen= threading.Thread(target=self.listen, args=())
        self.th_listen.start()
        print('Federated Learning v2.0.0 started at '+self.local_ip+":"+str(self.port))
        self.th_show= threading.Thread(target=self.show_state, args=())
        self.th_show.start()
        self.th_timed_task= threading.Thread(target=self.timed_task, args=())
        self.th_timed_task.start()

        #获取本地数据（mnist）
        npdata=np.load("./model/mnist"+str(self.data_id)+".npz")
        self.train_x=npdata["nodex"]
        self.train_y=npdata["nodey"]
        self.train_x = self.train_x / 255.0
        self.train_y=tf.one_hot(self.train_y,depth=10,dtype=tf.uint8)
        return
    def stop(self):
        self.running=False
        self.state_lock.acquire()
        self.state=0
        self.state_lock.release()
        self.th_listen.join()
        self.th_show.join()
        return
    def show_state(self):
        last_state=0
        while self.running:
            time.sleep(1)
            if self.state!=last_state:
                #print("state changes to:",self.state)
                last_state=self.state
            order_json=json.dumps({"order":"send_state","ip":s.local_ip,"isroot":self.isroot,"state":self.state,"up_server":self.up_server,"down_server":self.down_server})
            self.udp_send((self.cloud_server,self.port),order_json.encode('utf-8'))   
        return#向云服务器上传状态
    def change_topology(self,json_data):
        print("start to change_topology")
        if self.state!=1:
            text="ip:"+self.local_ip+",isn't in state 1,refuse to change topology"
            sendjson=json.dumps({"order":"print","text":""})
            self.udp_send((self.cloud_server,self.port),sendjson.encode('utf-8'))
            return
        self.state_lock.acquire()
        self.up_server=json_data['up_server']
        self.down_server=json_data['down_server']
        if self.up_server=="":
            self.isroot=True
        else:
            self.isroot=False
        self.clear_buffer()
        self.state_lock.release()
        sendjson=json.dumps({"order":"feedback","ip":self.local_ip ,"detail":"topology_changed"})
        self.udp_send((self.cloud_server,self.port),sendjson.encode('utf-8'))
        print("------------change_topology success----------")
        print("is_root:{}\nup_server:{}\ndown_server:{}".format(self.isroot,self.up_server,self.down_server))
        return
    def timed_task(self):
        current_time=time.time()
        timmer1=current_time
        while(self.running):
            time.sleep(1)
            current_time=time.time()
            if current_time-timmer1>300:#每五分钟清理一次
                self.command_id.clear()
                timmer1=current_time
        return

    #---------------模型训练、预测、更新、存储、读取-----------------

    def get_weights_from_h5(self,path):
        self.temp_model.load_weights(path)
        temp_layers=self.temp_model.get_weights()
        temp_layers=np.array(temp_layers)
        return temp_layers

    #---------------联邦学习相关-----------------
    def start_FL(self):#启动联邦学习
        assert self.state==1 ,"start FL while my state is not 1"
        self.FL_round+=1
        if self.optimizer=="SGD":
            th_start_aggregate= threading.Thread(target=self.start_aggregation, args=())
            th_start_aggregate.start()
        else :
            print("优化器%s,在这个版本还没实现"%(self.optimizer))
        return

    def average_aggregate_gradient_root(self):#根节点的聚合
        #先跟非根节点一样将子节点的权值相加
        all_weight=self.add_aggregate_gradient()
        #将相加之后的结果除以权重再计算得到新参数
        self.mymodel.aggregate_root(all_weight)
        print("average_aggregate_model_root complete")
        return

    def add_aggregate_gradient(self):#将子节点的模型相加，返回总模型权重
        #本地模型
        print("start aggregation...")
        weight_all=0
        path_list=[]
        for i in range(0,self.n_down_server):
            if self.is_child_participant[i]==1:
                path_list.append("./model/aggregate_gradient_"+self.down_server[i]+".zip")
                weight_all+=self.received_weight[i]

        if self.isparticipant:
            path_list.append("./model/local_gradient.zip")
            weight_all+=self.local_weight
        
        self.mymodel.add_aggregate_gradient(path_list)
        print("add_aggregate_model complete")
        return weight_all

    def start_update(self):#根节点在聚合完成后执行该函数，非根节点在接收完新模型之后执行该函数
        self.state_lock.acquire()
        self.state=4
        self.state_lock.release()

        self.mymodel.update_load_model("./model/UpdateModel.zip")
        
        with open("./model/UpdateModel.zip","rb") as f:
            data_bytes = f.read()
        for i in range(0,self.n_down_server):#向下层发送更新数据
            print("send new model to "+self.down_server[i])
            self.G_s.send((self.down_server[i],self.port),{"call_back":"cb_update_model"},data_bytes)#向下层发送新模型
        self.state_lock.acquire()
        self.state=1
        self.state_lock.release()
        return

    def start_aggregation(self):#向下传递开始聚合的命令并进入聚合状态
        print("start_aggregation")
        self.state_lock.acquire()
        self.state=2
        self.state_lock.release()
        self.clear_buffer()
        if(self.n_down_server!=0):#向子节点传递开始联邦学习的命令
            for ds in self.down_server:
                self.command_send((ds,self.port),'{"order":"aggregation_confirm"}')

        if(self.isparticipant):
            local_gradient_path=self.mymodel.local_model_train()#开始本地训练
        if(self.n_down_server!=0):#有子节点才需要等待接收
            self.wait_for_child() #等待所有子节点上传完毕
        #计算平均、向上传输
        self.state_lock.acquire()
        self.state=3
        self.state_lock.release()
        #判断是否参与聚合
        aggregate_empty=True#如果自己与子节点都不参与则不参与聚合
        if self.isparticipant:
            aggregate_empty=False
        else:
            for i in range(0,self.n_down_server):
                if self.is_child_participant[i]==1:
                    aggregate_empty=False#自己不参与,但有子节点参与
        if aggregate_empty:
            self.G_s.send((self.up_server,self.port),{"call_back":"cb_child_aggregation","weight":str(0)},b"aggregate_empty")#向上层发送聚合结果
            return

        if not self.isroot:#非根节点
            all_weight=self.add_aggregate_gradient()#本地聚合
            with open("./model/AggregateGradient.zip","rb") as f:
                data_bytes = f.read()
            self.G_s.send((self.up_server,self.port),{"call_back":"cb_child_aggregation","weight":str(all_weight)},data_bytes)#向上层发送聚合结果
            self.state_lock.acquire()#非root进入状态1等待分发新模型
            self.state=1
            self.state_lock.release()
        else:#是根节点
            if aggregate_empty:#本节点与子节点都不参与聚合
                print("this server and it's down server don't take part in the aggregration")
                self.state_lock.acquire()
                self.state=1
                self.state_lock.release()
                return
            self.average_aggregate_gradient_root()#计算平均
            #开始分发模型
            self.start_update()
        return

    def cb_child_aggregation(self,data,json_data):#接收并保存子节点发来的数据 判断子节点是否参与聚合
        addr=json_data["addr"]
        weight=int(json_data["weight"])
        down_ip=addr[0]
        path="./model/aggregate_gradient_"+down_ip+".zip"
        ip_index=self.down_server.index(down_ip)
        if down_ip not in self.down_server:
            raise ValueError("receive data from"+down_ip+" ,but this ip is not my down server")
        if weight==0:#表示该IP不参与聚合
            self.is_child_participant[ip_index]=-1
            print(down_ip+" don't take part in the aggregation")
            return
        self.received_weight[ip_index]=weight
        with open(path,"wb") as f:
            f.write(data)
        self.is_child_participant[ip_index]=1
        return

    def cb_update_model(self,data,json_data):
        addr=json_data["addr"]
        up_ip=addr[0]
        if up_ip != self.up_server:
            raise ValueError("receive data from"+up_ip+" ,but this ip is not my up server")
        with open("./model/UpdateModel.zip","wb") as f:
            f.write(data)
        self.start_update()
        return


    def send_op_data(self,json_data):
        if json_data["FL_round"]==self.glimmer_op.FL_round:
            self.glimmer_op.save("./model/optimizer.npz")
            with open("./model/optimizer.npz","rb") as f:
                data_bytes = f.read()
            self.G_s.send((json_data["root_ip"],self.port),{"call_back":"cb_update_optimizer"},data_bytes)
        else:
            for ip in self.down_server:
                self.command_send((ip,self.port),json.dumps(json_data))#上一轮不是根节点就向下传递请求
        return

    def cb_update_optimizer(self,data,json_data):
        with open("./model/optimizer.npz","wb") as f:
            f.write(data)
        self.glimmer_op.load("./model/optimizer.npz")
        assert self.glimmer_op.FL_round==self.FL_round ,"cb_update_optimizer received wrong optimizer data"
        self.flag["glimmer_op_update"]=True
        return


    #---------------其他函数-----------------
    def clear_buffer(self):#初始化缓存
        self.is_child_participant=[]#0未初始化不知道是否参与，1参与，-1不参与
        self.received_weight=[]#接收各个ip下数据的权重，在发送最后一个数据时发送
        for i in self.down_server:
            self.is_child_participant.append(0)
            self.received_weight.append(0)
        self.n_down_server=len(self.down_server)
        return
    def udp_send(self,address,message):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_socket.sendto(message, address) #将msg内容发送给指定接收方
        client_socket.close()
        return
    def command_send(self,address,message :str):#需要回执的命令 成功则返回true
        message_dic=json.loads(message)
        random_id=np.random.randint(1e6)
        if "feedback_id" in message_dic or "addr" in message_dic:
            print("warning: \"feedback_id\" \"addr\" in message_dic is overrode")
        message_dic['feedback_id']=random_id
        message_dic['addr']=(self.local_ip,self.port)

        message=json.dumps(message_dic)
        self.udp_send(address,message.encode("utf-8"))
        start_time=time.time()
        check_time=0.1#检查是否收到回执的间隙
        resend_time=0.5#重发消息的间隙
        resend_n=round(resend_time/check_time)
        n=0
        while(time.time()-start_time<3):
            time.sleep(check_time)
            n+=1
            if random_id in self.command_id:
                return True
            elif n==resend_n:
                self.udp_send(address,message.encode("utf-8"))#没有相应，重发命令
                n=0
        print("node {} is unresponsive, task id:{}, message{}".format(address,random_id,message))
        return False
    def feedback(self,json_data): #已经处理过的命令返回False，新命令返回True
        id=json_data["feedback_id"]
        ip=json_data["addr"][0]
        port=json_data["addr"][1]
        if "order" in json_data:#是feedback发送来的，不是新命令
            if json_data["order"]=="feedback":
                self.command_id.add(id)
                return False#回执不需要处理
        #可能是新命令
        send_dic={"order":"feedback","feedback_id":id}
        send_dic['addr']=(self.local_ip,self.port)
        sned_json=json.dumps(send_dic)
        self.udp_send((ip,port),sned_json.encode('utf-8'))
        if id in self.command_id:
            return False
        self.command_id.add(id)
        return True
    def change_parameter(self,json_data):
        name=json_data["parameter_name"]
        value=json_data["parameter_value"]
        order="self."+name+"="+str(value)
        exec(order)
        time_str=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        with open("./result.txt","a") as f:
            f.write("{}:{}\n".format(time_str,order)) 
        return
    def wait_for_flag(self,key):
        i=0
        while(True):
            if self.flag[key]:
                self.flag[key]=False
                print("\rwaite for {} complete ".format(key))
                return
            else:
                time.sleep(0.5)
                i+=1
                a=["--","\\","|","/"][i%4]
                print("\rwaite for {} {} ".format(key,a),end='')
        return
    def wait_for_child(self):
        ii=0
        def is_child_aggregate_complete():
            for i in range(self.n_down_server):
                if self.is_child_participant[i]!=1:
                    return False
            return True
        while(True):
            if is_child_aggregate_complete():
                for i in range(self.n_down_server):
                    print("\rwaite for",end='')
                    print("{:>15}:{:<2}".format(self.down_server[i],"√"),end='')
                print("")
                return
            else:
                time.sleep(0.5)
                ii+=1
                a=["--","\\","|","/"][ii%4]
                print("\rwaite for",end='')
                for j in range(self.n_down_server):
                    if self.is_child_participant[j]!=1:
                        print("{:>15}:{:<2}".format(self.down_server[j],a),end='')
                    else:
                        print("{:>15}:{:<2}".format(self.down_server[j],"√"),end='')
        return


if __name__ == '__main__':
    s=FLserver("config/config.json")
    s.start()


    

    
