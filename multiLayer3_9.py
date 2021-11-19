from utils import d2l
import numpy as np
import torch

train_iter ,test_iter = d2l.iter_pic()

num_input,num_output,num_hidden = 784,10,256
epochs = 10
batch_size = 128

for X,y in train_iter:

    print(type(X))
    break

def initParams():
    w1 = torch.tensor( np.random.normal(0.0,0.5,(num_input,num_hidden)),dtype=torch.float32)
    b1 = torch.tensor(np.zeros(num_hidden),dtype=torch.float32)
    w2 = torch.tensor( np.random.normal(0.0,0.5,(num_hidden,num_output)),dtype=torch.float32)
    b2 = torch.tensor(np.zeros(num_output),dtype=torch.float32)

    params = [w1,b1,w2,b2]
    for param in params:
        param.requires_grad_(requires_grad=True)
    return params

def relu(X):
    return torch.max(input=X,other=torch.tensor(0))


def net(X,params):
    X = X.view(X.shape[0],-1)
    hidden_Value = relu(torch.matmul(X,params[0]) + params[1])
    return torch.matmul(hidden_Value,params[2]) + params[3]

loss = torch.nn.CrossEntropyLoss()


if __name__=="__main__":
    params = initParams()
    d2l.train(net,train_iter,test_iter,epochs,batch_size,loss,params=params,lr = 10)
