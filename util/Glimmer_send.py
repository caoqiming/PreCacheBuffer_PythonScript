import time
import json
import threading
from math import ceil
import queue


class Glimmer_send:
    local_ip=None
    port=23336
    send_bytes_once=1024
    send_speed=20
    wait_time=20#发送完毕之后等待时间


    resend_id_queue=queue.Queue()
    flag_receive_complete=False #发送端接收接收端的反馈，知道接收端是否接收完毕
    flag_start_to_send=False    #发送端接收到开始发送的命令
    flag_is_receving=False      #接收端是否正在接收，目前同时只接收一个

    th_check_receive=None
    father_instance=None
    def __init__(self,addr,instance):
        self.local_ip=addr[0]
        self.port=addr[1]
        self.father_instance=instance
        return

    def check_receive(self,s_addr):
        received_timmer=time.time()
        last_receive_time=0
        last_receive_n=0
        while(True):
            is_update_completed=True
            time.sleep(1)#每1s检查是否收齐，不加sleep会占用listen线程的资源，会很慢
            if last_receive_time==0:
                last_receive_time=time.time()
            N=len(self.received_buffer)
            n=0
            for j in range(0,N):
                if self.received_buffer[j]!=[]:
                    n+=1
            if n<N:
                is_update_completed=False
                if n !=last_receive_n:
                    last_receive_n=n
                    last_receive_time=time.time()
                else:
                    if time.time()-last_receive_time>2 and n!=0:#超过时间没有收到数据认为丢包
                        resend_id_list=[]
                        for j in range(0,N):
                            if self.received_buffer[j]==[]:
                                resend_id_list.append(j)
                        resend_id_n=len(resend_id_list)#丢包的个数
                        print("packet lost {}/{},try to resend".format(resend_id_n,N))
                        for ii in range(0,ceil(resend_id_n/50)):
                            resend_order_json=json.dumps({"order":"resend","resend_id_list":resend_id_list[50*ii:50*(ii+1)]})
                            self.father_instance.udp_send(tuple(s_addr),resend_order_json.encode('utf-8'))
                            time.sleep(0.023)
                        last_receive_time=time.time()
            if(is_update_completed):
                break
        ans=self.father_instance.command_send(tuple(s_addr),'{"order":"receive_complete"}')
        #print("\rreceive_complete ",ans,"        ")
        return

    def start_receive(self,json_data):
        while(True):
            if self.flag_is_receving:
                time.sleep(1)
            else:
                self.flag_is_receving=True
                break
        call_back=json_data["call_back"]
        N=int(json_data["N"])
        s_addr=json_data["addr"]
        self.received_buffer=[]
        for i in range(0,N):#初始化缓存
            self.received_buffer.append([])

        th_check_receive=threading.Thread(target=self.check_receive, args=(s_addr,))
        th_check_receive.start()
        ans=self.father_instance.command_send(tuple(s_addr),'{"order":"start_send_data"}')
        #print("please start_send_data",ans)
        th_check_receive.join()
        strb=b"".join(self.received_buffer)
        self.received_buffer=[]#清空缓存
        self.flag_is_receving=False
        eval("self.father_instance.{}".format(call_back))(strb,json_data)#调用父实例中对应的回调函数
        return

    def prepare_send(self,r_addr:str,send_dic:dict,send_bytes:bytes):#允许发送比特数据和一个额外的字典类型数据，必须包含call_back，即回调函数的名字
        self.send_bytes=send_bytes
        data_length=len(self.send_bytes)
        send_turns=ceil(data_length/self.send_bytes_once)
        if "order" in send_dic or "N" in send_dic :
            raise RuntimeWarning("order N in send_dic is overrode")
        send_dic["order"]="send_ready"
        send_dic["N"]=str(send_turns)
        assert "call_back" in send_dic ,"Glimmer_send.send receives dic without key call_back"
        sned_json=json.dumps(send_dic)
        assert len(sned_json)<1400 ,"Glimmer_send send_dic is too big, big data shall be put in send_bytes"
        ans=self.father_instance.command_send(r_addr,sned_json)
        return ans

    def receive_data(self,json_data,receive_data2):
        self.received_buffer[json_data["data_id"]]=receive_data2
        return

    def send_data(self,addr):
        check_turns=50
        data_length=len(self.send_bytes)
        send_turns=ceil(data_length/self.send_bytes_once)
        should_have_time=check_turns*self.send_bytes_once/self.send_speed*10e-6
        last_time=time.time()
        for i in range(0,send_turns):#发送/接收过程
            #print('\r','{}/{}'.format(i+1,send_turns),end='')
            if i%check_turns==0:
                sleep_time=should_have_time-(time.time()-last_time)
                if sleep_time>0:
                    time.sleep(sleep_time)
                last_time=time.time()
            sendjson='{"order":"data","ip":"'+self.local_ip+'","data_id":'+str(i)+'}'
            self.father_instance.udp_send(addr,sendjson.encode('utf-8')+self.send_bytes[i*self.send_bytes_once:(i+1)*self.send_bytes_once])
        #发送完之后等待有无需要重传的
        wait_time=self.wait_time
        while(wait_time>0):
            if self.flag_receive_complete:
                self.flag_receive_complete=False
                #print("\rsend complete, received feed back")
                return
            if self.resend_id_queue.qsize()>0:
                resend_id_list=self.resend_id_queue.get()
                for i in resend_id_list:#发送/接收过程
                    sendjson='{"order":"data","ip":"'+self.local_ip+'","data_id":'+str(i)+'}'
                    self.father_instance.udp_send(addr,sendjson.encode('utf-8')+self.send_bytes[i*self.send_bytes_once:(i+1)*self.send_bytes_once])
            else:
                wait_time-=1
                time.sleep(1)
        print("send_data complete but didn't receive feedback")
        return

    def send(self,addr,send_dic,send_bytes):
        ans=self.prepare_send(addr,send_dic,send_bytes)
        if not ans:
            print("Glimmer_send failed,{} is irresponsive".format(addr))
            return
        waite_time=60#最大等待时间
        while(True):
            if self.flag_start_to_send:
                self.flag_start_to_send=False
                break
            if waite_time>0:
                waite_time-=1
                #print("\rwaiting",end='')
            else:
                print("Glimmer_send waite time out")
                return
            time.sleep(1)
        #print("start to send",end='')
        self.send_data(addr)
        return

