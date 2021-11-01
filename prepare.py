from CF_Autoencoder import CFA
import sys

m=6040 #人的个数
n=3952 #电影的个数
rating_nums=1000209 #打分的个数

#用户数 电影数 用户额外信息 电影额外信息 隐藏层 l1层
model=CFA(6040,3952,29,18,64,128)

save_index=sys.argv[1]

model.encoder_user.load_weights("./model/%s_encoder_user.h5"%(save_index))
model.decoder_user.load_weights("./model/%s_decoder_user.h5"%(save_index))
model.encoder_movie.load_weights("./model/%s_encoder_movie.h5"%(save_index))
model.decoder_movie.load_weights("./model/%s_decoder_movie.h5"%(save_index))

model.prepare_predict() #计算所有 得要好几个小时

