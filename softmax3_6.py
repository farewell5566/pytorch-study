
import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys
from torch import nn
import numpy as np

batch_size = 256

def iter_pic():
    mnist_train = torchvision.datasets.FashionMNIST(root="./data",train=True,download=True,transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root="./data",train=False,download=True,transform=transforms.ToTensor())

    if sys.platform.startswith("win"):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train,batch_size = batch_size,shuffle=True,num_workers = num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test,batch_size = batch_size,shuffle=True,num_workers = num_workers)
    return train_iter,test_iter


#初始化 参数
def initParams():
    w = torch.tensor(np.random.normal(0.0,0.1,(784,10)),dtype=torch.float32)
    b = torch.tensor(np.random.normal(0.0,0.05,(1,10)),dtype=torch.float32)
    w.requires_grad_(requires_grad = True)
    b.requires_grad_(requires_grad = True)
    return w,b

#28*28 784
def softMax(Y):
    y_exp = Y.exp()
    divisor = y_exp.sum(dim=1,keepdim =True)
    return y_exp/divisor

def modelXC(X,w,b):
    return softMax(torch.mm(X.view(-1,784),w) + b)


#损失函数
def lossXC(y_hat,labels):
    #num_vals = len(labels)
    return -torch.log(y_hat.gather(1,labels.view(-1,1)))

#计算准确率
def accuracy(y_hat,y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_accuracy(data_iter,net,params):
    acc_sum,n = 0.0,0
    for X,y in data_iter:
        y_hat = net(X,params[0],params[1])
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum/n


def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


def train_ch3(net,train_iter,test_iter,loss,num_epoches,batch_size,
              params=None,lr=None,optimizer=None):
    for i in range(num_epoches):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X,y in train_iter:
            y_hat = net(X,params[0],params[1])
            l = loss(y_hat,y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params,lr,batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) ==y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter,net,params)
        print("epoch %d,loss %.4f,train acc %.3f, test %.3f" %(i +1 ,train_l_sum / n,train_acc_sum/n,test_acc))
    return params

if __name__=="__main__":

    #测试
    # iter_train,_ = iter_pic()
    # w,b = initParams()
    # sumVals,n = 0,0
    # for X,y in iter_train:
    #     y_hat = modelXC(X,w,b)
    #     sumVals += (y_hat.argmax(dim=1)==y).float().sum().item()
    #     n += y.shape[0]
    # print(sumVals/n)

    #训练模型
    epoch = 10
    lr = 0.1
    optimizer = None
    iter_train, iter_test= iter_pic()
    w, b = initParams()
    params = train_ch3(modelXC,iter_train,iter_test,lossXC,epoch,batch_size,[w,b],lr)
    X,y = iter(iter_test).next()
    true_labels = y.numpy()
    hat_labels = modelXC(X,params[0],params[1]).argmax(dim = 1).numpy()
    print("true",true_labels)
    print("pretict",hat_labels)

    true_labels = torch.tensor(true_labels)
    hat_labels = torch.tensor(hat_labels)
    (true_labels == hat_labels).float()
    scores = (true_labels == hat_labels).float().sum().item()
    print(scores / hat_labels.shape[0])





