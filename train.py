import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from data_preprocess import get_train_data
import random
from CF_Autoencoder import CFA

m=6040 #人的个数
n=3952 #电影的个数
rating_nums=1000209 #打分的个数



#用户数 电影数 用户额外信息 电影额外信息 隐藏层 l1层
model=CFA(6040,3952,29,18,64,128)
# tf.keras.utils.plot_model(model.encoder_user, "CFA_model_encoder_user.png", show_shapes=True)
# tf.keras.utils.plot_model(model.decoder_user, "CFA_model_decoder_user.png", show_shapes=True)
# tf.keras.utils.plot_model(model.encoder_movie, "CFA_model_encoder_movie.png", show_shapes=True)
# tf.keras.utils.plot_model(model.decoder_movie, "CFA_model_decoder_movie.png", show_shapes=True)

#optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)


model.encoder_user.load_weights("./model/encoder_user.h5")
model.decoder_user.load_weights("./model/decoder_user.h5")
model.encoder_movie.load_weights("./model/encoder_movie.h5")
model.decoder_movie.load_weights("./model/decoder_movie.h5")

for runtime in range(0,3):
    indexs=list(range(0,rating_nums))#rating_nums
    random.shuffle(indexs)
    batch_size=1
    batch=[]
    for i in range(0,len(indexs)):
        batch.append(indexs[i])
        if(i%1000==999):
            print(runtime,"轮:",i/(rating_nums+0.0))
        if(i%batch_size!=batch_size-1):
            continue

        train_data=get_train_data(batch)
        model.train_step_batch(train_data,optimizer)
        batch=[]

model.encoder_user.save_weights("./model/encoder_user.h5")
model.decoder_user.save_weights("./model/decoder_user.h5")
model.encoder_movie.save_weights("./model/encoder_movie.h5")
model.decoder_movie.save_weights("./model/decoder_movie.h5")
