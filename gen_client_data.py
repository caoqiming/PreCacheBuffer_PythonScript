#%%
#利用part_i里的数据生成模拟访问的数据 供c++的client部分使用
import json
import sys
from data_preprocess import BinarySearch

def read_rating(s):
    ans=s.split("::")
    return ans[0],ans[1],ans[2],ans[3] #UserID::MovieID::Rating::Timestamp

class my_rating:
    def __init__(self,uid,movieid,rating,timestamp):
        self.uid_=int(uid)
        self.movieid_=int(movieid)
        self.rating_=int(rating)
        if(self.rating_>3):
            self.rating_=1.0
        else:
            self.rating_=-1.0
        self.timestamp_=int(timestamp)

with open("./part_1/ratings_ordered.dat", "r") as f:  # 打开文件
    rating_data_1 = f.readlines()  # 读取文件，每个基站读取的数据是不一样的，只包含自己的用户的部分
rating_list_1=[] #储存打分数据的list
for one in rating_data_1:
    uid,movieid,rating,timestamp=read_rating(one)
    if int(rating)>3: #打分超过3分的就访问
        rating_list_1.append(my_rating(uid,movieid,rating,timestamp))

with open("./part_2/ratings_ordered.dat", "r") as f:  
    rating_data_2 = f.readlines()  
rating_list_2=[] #储存打分数据的list
for one in rating_data_2:
    uid,movieid,rating,timestamp=read_rating(one)
    if int(rating)>3:
        rating_list_2.append(my_rating(uid,movieid,rating,timestamp))

with open("./part_3/ratings_ordered.dat", "r") as f:  
    rating_data_3 = f.readlines()  
rating_list_3=[] #储存打分数据的list
for one in rating_data_3:
    uid,movieid,rating,timestamp=read_rating(one)
    if int(rating)>3:
        rating_list_3.append(my_rating(uid,movieid,rating,timestamp))



# rating_list_1 rating_list_2 rating_list_3中包含了全部的访问数据，但是太多了
# 只筛选一部分出来
start_time=float(sys.argv[1]) 
end_time=float(sys.argv[2])   
data_size=int(sys.argv[3])    #大约提取多少个数据

def get_index_between_time(start:float,end:float,rating_list:list): #不用date_preprocess里的，因为要把rating_list更换
    #输入时间范围，最大范围0~1,是一个比例 返回训练数据里对应的索引范围
    time_start=956703932
    time_end=1046454590
    start_time=int((time_end-time_start)*start+time_start)
    end_time=int((time_end-time_start)*end+time_start)
    return BinarySearch(rating_list,start_time),BinarySearch(rating_list,end_time)

rating_list_1_choosed=[]
rating_list_2_choosed=[]
rating_list_3_choosed=[]
rating_list_common=[] #会访问其他基站的部分


start_index,end_index=get_index_between_time(start_time,end_time,rating_list_1)
num_all=end_index-start_index
num_need=int(data_size/3)
x=int(num_all/num_need)
for i in list(range(start_index,end_index)):
    if i%x==0:
        rating_list_1_choosed.append(rating_list_1[i])
    elif i%x==1:
        rating_list_common.append(rating_list_1[i])
print("part_1，符合要求的数据有%d条,提取2/%d"%(num_all,x))

start_index,end_index=get_index_between_time(start_time,end_time,rating_list_2)
num_all=end_index-start_index
num_need=int(data_size/3)
x=int(num_all/num_need)
for i in list(range(start_index,end_index)):
    if i%x==0:
        rating_list_2_choosed.append(rating_list_2[i])
    elif i%x==1:
        rating_list_common.append(rating_list_2[i])
print("part_2，符合要求的数据有%d条,提取2/%d"%(num_all,x))

start_index,end_index=get_index_between_time(start_time,end_time,rating_list_3)
num_all=end_index-start_index
num_need=int(data_size/3)
x=int(num_all/num_need)
for i in list(range(start_index,end_index)):
    if i%x==0:
        rating_list_3_choosed.append(rating_list_3[i])
    elif i%x==1:
        rating_list_common.append(rating_list_3[i])
print("part_3，符合要求的数据有%d条,提取2/%d"%(num_all,x))

#将rating_list_common加入rating_list_i_choosed并排序
for one in rating_list_common:
    rating_list_1_choosed.append(one)
    rating_list_2_choosed.append(one)
    rating_list_3_choosed.append(one)

rating_list_1_choosed=sorted(rating_list_1_choosed,key=lambda x: x.timestamp_)
rating_list_2_choosed=sorted(rating_list_2_choosed,key=lambda x: x.timestamp_)
rating_list_3_choosed=sorted(rating_list_3_choosed,key=lambda x: x.timestamp_)


#读取网址数据
with open("data/image.txt","r") as f:
    urls=f.readlines()
#开始写入
with open("part_1/data.txt","w") as f:
    i=0
    for one in rating_list_1_choosed:
        i+=1
        data={}
        data['time']=i/2.0
        data['url']=urls[one.movieid_-1].strip('\n')
        data['client']=one.uid_
        data_str=json.dumps(data)
        f.write(data_str+"\n")

with open("part_2/data.txt","w") as f:
    i=0
    for one in rating_list_2_choosed:
        i+=1
        data={}
        data['time']=i/2.0
        data['url']=urls[one.movieid_-1].strip('\n')
        data['client']=one.uid_
        data_str=json.dumps(data)
        f.write(data_str+"\n")

with open("part_3/data.txt","w") as f:
    i=0
    for one in rating_list_3_choosed:
        i+=1
        data={}
        data['time']=i/2.0
        data['url']=urls[one.movieid_-1].strip('\n')
        data['client']=one.uid_
        data_str=json.dumps(data)
        f.write(data_str+"\n")

