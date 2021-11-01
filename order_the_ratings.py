#%%
with open("ratings.dat", "r") as f:  # 打开文件
    rating_data = f.readlines()  # 读取文件

def read_rating(s):
    ans=s.split("::")
    return ans[0],ans[1],ans[2],ans[3] #UserID::MovieID::Rating::Timestamp

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


rating_list=sorted(rating_list,key=lambda x: x.timestamp_)
with open("ratings_ordered.dat", "w") as f:  # 打开文件
    for one in rating_list:
        f.write("%s::%s::%s::%d\n"%(one.uid_,one.movieid_,one.rating_,one.timestamp_))