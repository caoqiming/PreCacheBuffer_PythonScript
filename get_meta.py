# -*- coding: utf-8 -*- 
import numpy as np

with open("users.dat", "r") as f:  # 打开文件
    users_data = f.readlines()  # 读取文件 UserID::Gender::Age::occupation::Zip-code


user_age_dic={"1":1,"18":2,"25":3,"35":4,"45":5,"50":6,"56":7}
class myuser:
    def __init__(self,oneline):
        self.uid_=int(oneline[0])
        self.gender_ = oneline[1]=='M' #index 0
        self.age_=oneline[2] # {1,18,25,35,45,50,56} index 1~7
        self.occupation_=int(oneline[3]) # 0~20 index 8~28
        self.vector_=np.zeros(29,dtype=np.float32)
        if self.gender_:
            self.vector_[0]=1
        self.vector_[user_age_dic[self.age_]]=1
        self.vector_[self.occupation_+8]=1
        self.vector_=self.vector_.reshape(1,29)
    
    def __str__(self) -> str:
        return "uid:%d gender:%d age:%d occupation:%d"%(self.uid_,self.gender_,self.age_,self.occupation_)

user_list=[]
for one in users_data:
    user_list.append(myuser(one.split("::")))

def get_user_by_id(id)->myuser:
    return user_list[id-1]

#用来获取电影元数据 有些序号对应的电影缺失了
with open("movies.dat", "r",encoding='utf-8',errors='ignore') as f:  # 打开文件
    movies_data = f.readlines()  # 读取文件 MovieID::Title::Genres

Genres_dic={
    'Action':0,
    'Adventure':1,
    'Animation':2,
    "Children's":3,
    'Comedy':4,
    'Crime':5,
    'Documentary':6,
    'Drama':7,
    'Fantasy':8,
    'Film-Noir':9,
    'Horror':10,
    'Musical':11,
    'Mystery':12,
    'Romance':13,
    'Sci-Fi':14,
    'Thriller':15,
    'War':16,
    'Western':17
    }

class mymovie:
    def __init__(self,oneline):
        self.vid_=int(oneline[0])
        self.title_ = oneline[1]
        self.vector_= np.zeros(18,dtype=np.float32)
        genres=oneline[2].split("|")
        for one in genres:
            self.vector_[Genres_dic[one.replace("\n", "")]]=1
        self.vector_=self.vector_.reshape(1,18)
    
    def __str__(self) -> str:
        return "vid:%d title:%s genres:%s"%(self.vid_,self.title_,self.genres_.__str__())
    
movie_id_list=[] #因为有些id缺失，所以需要映射一下
movie_list=[]
none_vid=[91, 221, 323, 622, 646, 677, 686, 689, 740, 817, 883, 995, 1048, 1072, 1074, 1182, 1195, 1229, 1239, 1338, 1402, 1403, 1418, 1435, 1451, 1452, 1469, 1478, 1481, 1491, 1492, 1505, 1506, 1512, 1521, 1530, 1536, 1540, 1560, 1576, 1607, 1618, 1634, 1637, 1638, 1691, 1700, 1712, 1736, 1737, 1745, 1751, 1761, 1763, 1766, 1775, 1778, 1786, 1790, 1800, 1802, 1803, 1808, 1813, 1818, 1823, 1828, 1838, 3815] #缺失的电影id

for one in movies_data:
    movie_list.append(mymovie(one.split("::")))
    while(len(movie_id_list) < movie_list[-1].vid_-1):
        movie_id_list.append(-1)
    movie_id_list.append(len(movie_list)-1)

def get_movie_by_id(id)->mymovie:
    if movie_id_list[id-1]==-1:
        return None
    return movie_list[movie_id_list[id-1]]

