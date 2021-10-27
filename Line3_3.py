import random

import torch
import numpy as np
from time import time
from matplotlib import pyplot as plt

#生成数据
num_size = 1000
num_inputs = 2

true_w = [3.4,5.6]
true_b = -3.2

batch_size = 16


def genData():
    features = torch.randn([1000,2],dtype=torch.float32)

    labels = features[:,0] * true_w[0] + features[:,1] * true_w[1] + true_b + torch.randn(1000,dtype=torch.float32)
    labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)
    return features,labels


#画图
def drawData(features,labels):
    plt.scatter(features[:,1],labels)
    plt.show()

#小批量读取 数据文本
#index_select ,第一个参数，0代表按行索引，1代表按列索引。第一个参数代表行或者列。
def readData(features,lables,size):
    num_example = len(features)
    indices = list(range(num_example))
    random.shuffle(indices)
    for i in range(0,num_example,size):
        j = torch.tensor(indices[i:min(i+size,num_example)])

        yield features.index_select(0,j),lables.index_select(0,j)


#定义模型
def modelLine(w,X,b):
    return torch.mm(X,w)+ b

#损坏函数
def calcLoss(Y,labels):
    return (labels - Y.view(labels.size())) **2 /2

#优化算法
def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

#训练模型
lr = 0.3
batch_size = 12
epoch = 50

if __name__ == "__main__":
    features,labels = genData()
    for X,y in readData(features,labels,batch_size):
        print(X,y)
        break

    #初始化参数
    w = torch.tensor(np.random.normal(0,0.1,size=(2,1)),dtype=torch.float32)
    b = torch.tensor(1.0,dtype=torch.float32)
    w.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)
    for i in range(epoch):
        for X,y in readData(features,labels,size=batch_size):
            l = calcLoss(modelLine(w,X,b) , y).sum()
            l.backward()
            sgd([w,b],lr,batch_size)

            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = calcLoss(modelLine(w,features,b),labels)
        print("epoch %d,loss %f"%(i + 1,train_l.mean().item()))
        print(true_w,'*'*10,w)
        print(true_b,'*'*10,b)


















