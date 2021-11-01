import numpy as np 
from get_meta import get_user_by_id,get_movie_by_id
with open("ratings_ordered.dat", "r") as f:  # 打开文件
    rating_data = f.readlines()  # 读取文件，每个基站读取的数据是不一样的，只包含自己的用户的部分
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

rating_list=[] #储存打分数据的list
for one in rating_data:
    uid,movieid,rating,timestamp=read_rating(one)
    rating_list.append(my_rating(uid,movieid,rating,timestamp))


RatingMatrix=np.zeros((m,n),dtype=np.float32)
for one in rating_list:
    RatingMatrix[one.uid_-1][one.movieid_-1]=one.rating_

def get_train_data(index : list) -> list: #输入一个序号 序号对应的rating.dat中的训练数据，支持批量
    train_data=list()
    for one_index in index:
        one_data=dict()
        rating=rating_list[one_index]
        one_data["rij"]=rating.rating_
        one_data["x"]=get_user_by_id(rating.uid_).vector_
        one_data["y"]=get_movie_by_id(rating.movieid_).vector_
        one_data["su"]=RatingMatrix[rating.uid_-1,::].reshape(1,n)
        one_data["si"]=RatingMatrix[::,rating.movieid_-1].reshape(1,m)
        train_data.append(one_data)
    return train_data

def BinarySearch(array:list(),t:int)->int: #二分查找
    low = 0
    height = len(array)-1
    while low < height:
        mid = int((low+height)/2)
        if array[mid].timestamp_ < t:
            low = mid + 1
        elif array[mid].timestamp_ > t:
            height = mid - 1
        else:
            return mid
    return mid

def get_index_between_time(start:float,end:float): 
    #输入时间范围，最大范围0~1,是一个比例 返回训练数据里对应的索引范围
    #再调用get_train_data即可获得训练数据
    time_start=956703932
    time_end=1046454590
    start_time=int((time_end-time_start)*start+time_start)
    end_time=int((time_end-time_start)*end+time_start)
    return BinarySearch(rating_list,start_time),BinarySearch(rating_list,end_time)



def get_user_info(user_ids : list) -> list: #输入用户id的列表 返回该用户的x和su
    ans=list()
    for one_id in user_ids:
        one_data=dict()
        one_data["x"]=get_user_by_id(one_id).vector_
        one_data["su"]=RatingMatrix[one_id-1,::].reshape(1,n)
        ans.append(one_data)
    return ans


def get_movie_info(movies_ids : list) -> list: #输入用户id的列表 返回该用户的x和su
    ans=list()
    for one_id in movies_ids:
        one_data=dict()
        if(get_movie_by_id(one_id)):
            one_data["y"]=get_movie_by_id(one_id).vector_
            one_data["si"]=RatingMatrix[::,one_id-1].reshape(1,m)
            ans.append(one_data)
        else:
            ans.append(None)
    return ans

# all=1000207
# index0,index1=get_index_between_time(0.202,1)
# print((index1-index0)/(all+0.0))