#利用part_i里的数据生成模拟访问的数据 供c++的client部分使用
import json

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
    if int(timestamp)>3: #打分超过3分的就访问
        rating_list_1.append(my_rating(uid,movieid,rating,timestamp))

with open("./part_2/ratings_ordered.dat", "r") as f:  
    rating_data_2 = f.readlines()  
rating_list_2=[] #储存打分数据的list
for one in rating_data_2:
    uid,movieid,rating,timestamp=read_rating(one)
    if int(timestamp)>3:
        rating_list_2.append(my_rating(uid,movieid,rating,timestamp))

with open("./part_3/ratings_ordered.dat", "r") as f:  
    rating_data_3 = f.readlines()  
rating_list_3=[] #储存打分数据的list
for one in rating_data_3:
    uid,movieid,rating,timestamp=read_rating(one)
    if int(timestamp)>3:
        rating_list_3.append(my_rating(uid,movieid,rating,timestamp))

with open("data/image.txt","r") as f:
    urls=f.readlines()

with open("part_1/data.txt","w") as f:
    i=0
    for one in rating_list_1:
        i+=1
        data={}
        data['time']=i
        data['url']=urls[one.movieid_-1].strip('\n')
        data['client']=one.uid_
        data_str=json.dumps(data)
        f.write(data_str+"\n")

with open("part_2/data.txt","w") as f:
    i=0
    for one in rating_list_2:
        i+=1
        data={}
        data['time']=i
        data['url']=urls[one.movieid_-1].strip('\n')
        data['client']=one.uid_
        data_str=json.dumps(data)
        f.write(data_str+"\n")

with open("part_3/data.txt","w") as f:
    i=0
    for one in rating_list_3:
        i+=1
        data={}
        data['time']=i
        data['url']=urls[one.movieid_-1].strip('\n')
        data['client']=one.uid_
        data_str=json.dumps(data)
        f.write(data_str+"\n")

