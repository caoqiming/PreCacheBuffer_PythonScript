#%%
from CF_Autoencoder import CFA

m=6040 #人的个数
n=3952 #电影的个数
rating_nums=1000209 #打分的个数

#用户数 电影数 用户额外信息 电影额外信息 隐藏层 l1层
model=CFA(6040,3952,29,18,64,128)

model.encoder_user.load_weights("./model/encoder_user.h5")
model.decoder_user.load_weights("./model/decoder_user.h5")
model.encoder_movie.load_weights("./model/encoder_movie.h5")
model.decoder_movie.load_weights("./model/decoder_movie.h5")
#%%
model.prepare_predict() #计算所有 得要好几个小时
#%%
Rating=model.predict_all() # 读取保存的结果
# %%
rank=model.get_movie_rank() # 获取电影受欢迎程度的排名
# %%
