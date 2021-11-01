import numpy as np
import tensorflow as tf
from data_preprocess import get_user_info,get_movie_info
import pickle


class TwoPartDense(tf.keras.layers.Layer): 

    def __init__(self, input_shape_1=28,input_shape_2=28, output_shape_1=10,layer_name=""):
        self.input_shape_1=input_shape_1
        self.input_shape_2=input_shape_2
        self.output_shape_1=output_shape_1
        self.layer_name_=layer_name
        if(layer_name!=""):
            super(TwoPartDense, self).__init__(name=layer_name)
        else:
            super(TwoPartDense, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w1 = tf.Variable(
            initial_value=w_init(shape=(input_shape_1,output_shape_1), dtype="float32"),trainable=True,name="w1"
            )
        self.w2 = tf.Variable(
            initial_value=w_init(shape=(input_shape_2,output_shape_1), dtype="float32"),trainable=True,name="w2"
            )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(output_shape_1,), dtype="float32"), trainable=True,name="b"
        )

    def call(self, inputs): #第一个维度必须是batch 不然模型输入必报错，也就是说矩阵乘法的时候input的必须放在前面，f**k
        return tf.keras.activations.tanh(
                tf.matmul(inputs[0],self.w1)+tf.matmul(inputs[1],self.w2)+self.b
            )#tf.keras.activations.tanh

    def get_config(self):
        return {"input_shape_1": self.input_shape_1,
                "input_shape_2": self.input_shape_2,
                "output_shape_1":self.output_shape_1,
                "layer_name":self.layer_name_
               }

class CFA(tf.keras.Model):
    """Convolutional variational autoencoder."""
    PredictRatingMatrix=None
    def __init__(self, m=10, n=10, user_side_size=2,movie_side_size=3,latent_dim=10,l1_size=64): #用户数 电影数 用户额外信息 电影额外信息 隐藏层 l1层
        super(CFA, self).__init__()
        self.m=m
        self.n=n
        # encoder_user
        vector_input=tf.keras.Input(shape=(n),name="vector_input")
        side_input=tf.keras.Input(shape=(user_side_size),name="side_input")
        l1=TwoPartDense(n,user_side_size,l1_size,layer_name="l1")([vector_input,side_input])
        latent_vector=TwoPartDense(l1_size,user_side_size,latent_dim,layer_name="l2")([l1,side_input]) #l2
        self.encoder_user = tf.keras.Model(inputs=[vector_input,side_input],outputs=[latent_vector],name="encode_user_model")

        # decoder_user
        latent_vector=tf.keras.Input(shape=(latent_dim),name="latent_input")
        side_input=tf.keras.Input(shape=(user_side_size),name="side_input")
        l3=TwoPartDense(latent_dim,user_side_size,l1_size,layer_name="l3")([latent_vector,side_input])
        l4=TwoPartDense(l1_size,user_side_size,n,layer_name="l4_1")([l3,side_input])
        side_output=TwoPartDense(l1_size,user_side_size,user_side_size,layer_name="l4_2")([l3,side_input])
        self.decoder_user = tf.keras.Model(inputs=[latent_vector,side_input],outputs=[l4,side_output],name="decode_user_model")

        # encoder_movie
        vector_input=tf.keras.Input(shape=(m),name="vector_input")
        side_input=tf.keras.Input(shape=(movie_side_size),name="side_input")
        l1=TwoPartDense(m,movie_side_size,l1_size,layer_name="l1")([vector_input,side_input])
        latent_vector=TwoPartDense(l1_size,movie_side_size,latent_dim,layer_name="l2")([l1,side_input]) #l2
        self.encoder_movie = tf.keras.Model(inputs=[vector_input,side_input],outputs=[latent_vector],name="encode_movie_model")

        # decoder_movie
        latent_vector=tf.keras.Input(shape=(latent_dim),name="latent_input")
        side_input=tf.keras.Input(shape=(movie_side_size),name="side_input")
        l3=TwoPartDense(latent_dim,movie_side_size,l1_size,layer_name="l3")([latent_vector,side_input])
        l4=TwoPartDense(l1_size,movie_side_size,m,layer_name="l4_1")([l3,side_input])
        side_output=TwoPartDense(l1_size,movie_side_size,movie_side_size,layer_name="l4_2")([l3,side_input])
        self.decoder_movie = tf.keras.Model(inputs=[latent_vector,side_input],outputs=[l4,side_output],name="decode_movie_model")

        #predict
        U_input=tf.keras.Input(shape=(latent_dim),name="U_input")
        V_input=tf.keras.Input(shape=(latent_dim),name="V_input")
        predict=tf.math.tanh(tf.matmul(U_input,tf.transpose(V_input)))
        self.get_predict = tf.keras.Model(inputs=[U_input,V_input],outputs=[predict],name="get_predict")

    
    @tf.function
    def train_step(self,inputs,a1=0.5,a2=0.5,noise_factor=0.01):
        #生成噪声
        su_noisy = inputs["su"] + noise_factor * tf.random.normal(shape=inputs["su"].shape) 
        x_noisy = inputs["x"] + noise_factor * tf.random.normal(shape=inputs["x"].shape)
        si_noisy = inputs["si"] + noise_factor * tf.random.normal(shape=inputs["si"].shape)
        y_noisy = inputs["y"] + noise_factor * tf.random.normal(shape=inputs["y"].shape)
        with tf.GradientTape(persistent=True) as tape:
            U=self.encoder_user({"vector_input":su_noisy , "side_input":x_noisy})
            su_,x_=self.decoder_user({"latent_input":U , "side_input":x_noisy})
            V=self.encoder_movie({"vector_input":si_noisy , "side_input":y_noisy})
            si_,y_=self.decoder_movie({"latent_input":V , "side_input":y_noisy})
            loss_1=10*tf.losses.mean_squared_error(
                tf.math.tanh(tf.matmul(U,tf.transpose(V))) , inputs["rij"]
                ) #最重要的部分 潜在向量相乘之后跟真正打分的误差
            loss_2=tf.losses.mean_squared_error(inputs["su"],su_)
            loss_3=tf.losses.mean_squared_error(inputs["x"],x_)
            loss_4=tf.losses.mean_squared_error(inputs["si"],si_)
            loss_5=tf.losses.mean_squared_error(inputs["y"],y_)
            loss_6=tf.norm(U, ord=2) + tf.norm(V, ord=2)  #防止过拟合
            loss=loss_1+a1*loss_2+(1-a1)*loss_3+a2*loss_4+(1-a2)*loss_5 + loss_6
        gradients_encoder_user = tape.gradient(loss,self.encoder_user.trainable_variables)
        gradients_decoder_user = tape.gradient(loss,self.decoder_user.trainable_variables)
        gradients_encoder_movie = tape.gradient(loss,self.encoder_movie.trainable_variables)
        gradients_decoder_movie = tape.gradient(loss,self.decoder_movie.trainable_variables)
        return gradients_encoder_user,gradients_decoder_user,gradients_encoder_movie,gradients_decoder_movie,loss

    @tf.function
    def train_step_batch(self,inputs,optimizer,a1=0.5,a2=0.5,noise_factor=0.02):
        #生成噪声
        batch_size=len(inputs)
        gradients_encoder_user_list=[]
        gradients_decoder_user_list=[]
        gradients_encoder_movie_list=[]
        gradients_decoder_movie_list=[]
        loss_all=0
        for input in inputs:
            gradients_encoder_user,gradients_decoder_user,gradients_encoder_movie,gradients_decoder_movie,loss=self.train_step(input,a1,a2,noise_factor)
            gradients_encoder_user_list.append(gradients_encoder_user)
            gradients_decoder_user_list.append(gradients_decoder_user)
            gradients_encoder_movie_list.append(gradients_encoder_movie)
            gradients_decoder_movie_list.append(gradients_decoder_movie)
            loss_all+=loss

        gradients_encoder_user=gradients_encoder_user_list[0]
        gradients_decoder_user=gradients_decoder_user_list[0]
        gradients_encoder_movie=gradients_encoder_movie_list[0]
        gradients_decoder_movie=gradients_decoder_movie_list[0]

        for i in range(1,batch_size):
            gradients_encoder_user+=gradients_encoder_user_list[i]
            gradients_decoder_user+=gradients_decoder_user_list[i]
            gradients_encoder_movie+=gradients_encoder_movie_list[i]
            gradients_decoder_movie+=gradients_decoder_movie_list[i]
        for layer in gradients_encoder_user:
            layer/=batch_size
        for layer in gradients_decoder_user:
             layer/=batch_size
        for layer in gradients_encoder_movie:
            layer/=batch_size
        for layer in gradients_decoder_movie:
             layer/=batch_size

        optimizer.apply_gradients(zip(gradients_encoder_user, self.encoder_user.trainable_variables))
        optimizer.apply_gradients(zip(gradients_decoder_user, self.decoder_user.trainable_variables))
        optimizer.apply_gradients(zip(gradients_encoder_movie, self.encoder_movie.trainable_variables))
        optimizer.apply_gradients(zip(gradients_decoder_movie, self.decoder_movie.trainable_variables))
        print("loss:",loss_all.numpy()/batch_size)


    def prepare_predict(self): #计算所有 得要好几个小时
        self.user_U=[]
        self.movie_V=[]
        user_info=get_user_info(list(range(1,self.m+1)))
        movie_info=get_movie_info(list(range(1,self.n+1)))
        for i in range(0,self.m):
            self.user_U.append(self.encoder_user({"vector_input":user_info[i]["su"] , "side_input":user_info[i]["x"]}))
        for i in range(0,self.n):
            if movie_info[i]==None:
                self.movie_V.append(None)
            else:
                self.movie_V.append(self.encoder_movie({"vector_input":movie_info[i]["si"] , "side_input":movie_info[i]["y"]}))
        self.PredictRatingMatrix=np.zeros((self.m,self.n),dtype=np.float32)
        for j in range(0,self.n):
            print("prepare_predict:%d/%d"%(j+1,self.n))
            if self.movie_V[j]!=None:
                for i in range(0,self.m):
                    self.PredictRatingMatrix[i][j]=self.get_predict({"U_input":self.user_U[i],"V_input":self.movie_V[j]})
        with open('./PredictRatingMatrix.pkl', 'wb') as file: #储存预测结果
            pickle.dump(self.PredictRatingMatrix, file)
        print("PredictRatingMatrix saved")

        ans=list(range(1,self.n+1))
        movie_score=[]
        for j in range(0,self.n):
            temp=0.0
            for i in range(0,self.m):
                temp+=self.PredictRatingMatrix[i][j]
            movie_score.append(temp)
        ans=sorted(ans,key=lambda x: movie_score[x-1],reverse=True)
        with open('./Movie_Rank.pkl', 'wb') as file: #储存预测结果
            pickle.dump(ans, file)
        print("Movie_Rank saved")

    def predict_all(self)->np.ndarray: #返回预测的Rij矩阵
        if(self.PredictRatingMatrix==None):
            with open('./PredictRatingMatrix.pkl', 'rb') as file:
                self.PredictRatingMatrix=pickle.load(file)
        return self.PredictRatingMatrix

    def test(self,inputs):
        err=0
        for input in inputs:
            U=self.encoder_user({"vector_input":input["su"] , "side_input":input["x"]})
            V=self.encoder_movie({"vector_input":input["si"] , "side_input":input["y"]})
            predict=tf.math.tanh(tf.matmul(U,tf.transpose(V)))
            print("预测打分:%f,实际打分:%f"%(predict,input["rij"]))
            err+=abs(predict-input["rij"])
        print("平均偏差:%f"%(err/(len(inputs)+0.0)))

    def get_movie_rank(self)->list:
        with open('./Movie_Rank.pkl', 'rb') as file:
            ans=pickle.load(file)
        return ans
