import numpy as np
class Glimmer_adam:
    beta1=0.9
    beta2=0.999
    learning_rate=0.001
    epsilon=1e-7
    mt_1=None
    vt_1=None
    FL_round=0
    def __init__(self,FL_round):
        self.FL_round=FL_round
        return
    
    def get_new_chage_weights(self,weights: np.ndarray)->np.ndarray:#输入新的梯度，输出模型应该改变的量
        if self.mt_1 is None:
            m_t=weights
            v_t=weights**2
        else:
            m_t=self.beta1*self.mt_1+(1-self.beta1)*weights
            v_t=self.beta2*self.vt_1+(1-self.beta2)*weights**2
        self.mt_1=m_t
        self.vt_1=v_t
        self.FL_round+=1
        return self.learning_rate*m_t/(pow(v_t,0.5)+self.epsilon)
    def save(self,addr):#保存参数

        np.savez(addr,mt_1=self.mt_1,vt_1=self.vt_1,FL_round=self.FL_round)
        return
    def load(self,addr):#读取参数
        npdata=np.load(addr,allow_pickle=True)
        self.mt_1=npdata["mt_1"]
        self.vt_1=npdata["vt_1"]
        self.FL_round=npdata["FL_round"]
        return



    