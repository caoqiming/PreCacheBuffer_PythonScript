#%%
# coding=utf-8
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from CF_Autoencoder import CFA

m=6040 #人的个数
n=3952 #电影的个数

#用户数 电影数 用户额外信息 电影额外信息 隐藏层 l1层
model=CFA(6040,3952,29,18,64,128)

model.encoder_user.load_weights("./model/encoder_user.h5")
model.decoder_user.load_weights("./model/decoder_user.h5")
model.encoder_movie.load_weights("./model/encoder_movie.h5")
model.decoder_movie.load_weights("./model/decoder_movie.h5")

with open("data/image.txt","r") as f:
    urls=f.readlines()

message1 = {'message': 'please use post instead of get'}
message2 = {'message': 'miss the parameter: type'}
message3 = {'message': 'request is not json'}
host = ('0.0.0.0', 8888)



def get_strategy_handler(post_dict):
    data = {'message' : 'movie_rank'}
    n=int(post_dict['nums'])
    rank=model.get_movie_rank() # 获取电影受欢迎程度的排名
    rank=rank[:n]
    movie_url=[]
    for one in rank:
        movie_url.append(urls[one].strip('\n'))
    data['rank']=movie_url
    return data


class Resquest(BaseHTTPRequestHandler):
    def do_GET(self):
        content_len = int(self.headers.get('Content-Length'))
        post_body = self.rfile.read(content_len)
        print(post_body)
        try:
            post_dict=json.loads(post_body.decode())
        except:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(message3).encode())
            return

        if 'type' not in post_dict:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(message2).encode())
            return
        if post_dict['type']=='get_strategy':
            data=get_strategy_handler(post_dict)


        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


 
if __name__ == '__main__':
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()

# %%
