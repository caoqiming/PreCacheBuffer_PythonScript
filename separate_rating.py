# 根据用户 将数据集划分为不同的部分
#%%
import matplotlib.pyplot as plt
import numpy as np 
from get_meta import get_user_by_id,get_movie_by_id
with open("ratings.dat", "r") as f:  # 打开文件
    rating_data = f.readlines()  # 读取文件
m=6040 #人的个数
n=3952 #电影的个数
rating_nums=1000209 #打分的个数


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

rating_list=[]
for one in rating_data:
    uid,movieid,rating,timestamp=read_rating(one)
    rating_list.append(my_rating(uid,movieid,rating,timestamp))


RatingMatrix=np.zeros((m,n),dtype=np.float32)
for one in rating_list:
    RatingMatrix[one.uid_-1][one.movieid_-1]=one.rating_




class my_rating2:
    def __init__(self,uid,movieid,rating,timestamp):
        self.uid_=uid
        self.movieid_=movieid
        self.rating_=rating
        self.timestamp_=int(timestamp)

rating_list=[]
for one in rating_data:
    uid,movieid,rating,timestamp=read_rating(one)
    rating_list.append(my_rating2(uid,movieid,rating,timestamp))

i=0
with open("ratings_part_1.dat", "w") as f:  # 打开文件
    while(i<rating_nums):
        if int(rating_list[i].uid_)%3==0:
            one=rating_list[i]
            f.write("%s::%s::%s::%d\n"%(one.uid_,one.movieid_,one.rating_,one.timestamp_))
        i+=1
i=0
with open("ratings_part_2.dat", "w") as f:  # 打开文件
    while(i<rating_nums):
        if int(rating_list[i].uid_)%3==1:
            one=rating_list[i]
            f.write("%s::%s::%s::%d\n"%(one.uid_,one.movieid_,one.rating_,one.timestamp_))
        i+=1
i=0
with open("ratings_part_3.dat", "w") as f:  # 打开文件
    while(i<rating_nums):
        if int(rating_list[i].uid_)%3==2:
            one=rating_list[i]
            f.write("%s::%s::%s::%d\n"%(one.uid_,one.movieid_,one.rating_,one.timestamp_))
        i+=1

