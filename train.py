import tensorflow as tf
from data_preprocess import get_train_data,get_index_between_time
import random
from CF_Autoencoder import CFA
import sys

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

if __name__ == '__main__':
    start_time=float(sys.argv[1])
    end_time=float(sys.argv[2])
    save_index=sys.argv[3]
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)
    model.encoder_user.load_weights("./model/init_model/encoder_user.h5")
    model.decoder_user.load_weights("./model/init_model/decoder_user.h5")
    model.encoder_movie.load_weights("./model/init_model/encoder_movie.h5")
    model.decoder_movie.load_weights("./model/init_model/decoder_movie.h5")

    begin_index,end_index=get_index_between_time(start_time,end_time)
    indexs=list(range(begin_index,end_index))
    random.shuffle(indexs)
    rating_nums=len(indexs)
    print("读取%d条本地数据"%(rating_nums))
    batch_size=1
    batch=[]
    for i in range(0,rating_nums):
        batch.append(indexs[i])
        if(i%1000==999):
            print(i/(rating_nums+0.0))
        if(i%batch_size!=batch_size-1):
            continue

        train_data=get_train_data(batch)
        model.train_step_batch(train_data,optimizer)
        batch=[]
    
    model.encoder_user.save_weights("./model/%s_encoder_user.h5"%(save_index))
    model.decoder_user.save_weights("./model/%s_decoder_user.h5"%(save_index))
    model.encoder_movie.save_weights("./model/%s_encoder_movie.h5"%(save_index))
    model.decoder_movie.save_weights("./model/%s_decoder_movie.h5"%(save_index))
