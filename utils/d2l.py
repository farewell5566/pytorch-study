import torch
import torchvision
from torchvision import transforms as transforms


import torch.utils.data as Data
from matplotlib import pyplot as plt
from torch import nn



def iter_pic(batch_size):
    dataset_train = torchvision.datasets.FashionMNIST(root="./data",train=True,transform=transforms.ToTensor())
    dataset_test = torchvision.datasets.FashionMNIST(root="./data",train=False,transform=transforms.ToTensor())
    iter_train = Data.DataLoader(dataset_train,shuffle= True,batch_size = batch_size )
    iter_test = Data.DataLoader(dataset_test,shuffle = True,batch_size = batch_size)



    return iter_train,iter_test


def pltXC(X,Y,xname,yname):
    plt.plot(X.detach().numpy(),Y.detach().numpy())
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()

def getRightvalues(y_hat,y):
    return (y_hat.argmax(dim = 1) == y).float().sum().item()

def sgd(params,lr,batch_size):
    for param in params:
        param.data = - lr * param.grad /batch_size



def evaluate_acc(net,test_iter,paras):
    acc_sum ,n = 0.0,0
    for X,y in test_iter:
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim =1) ==y).float().sum().item()
        n += y.shape[0]
    return acc_sum/n


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self,X):
        return X.view(X.shape[0],-1)

def train(net,iter_train,iter_test,epochs,batch_size,loss,optim =None,params = None,lr =None):
    for i in range(epochs):
  
        train_l_sum ,train_acc_sum ,n =0.0, 0.0,0
        for X,y in iter_train:
            y_hat = net(X)
            l = loss(y_hat,y).sum()

            #梯度清零
            if optim is not None:
                optim.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optim is None:
                sgd(params,lr,batch_size)
            else :
                optim.step()


            train_acc_sum += getRightvalues(y_hat,y)
            train_l_sum += l.item()
            n += y.shape[0]

        test_acc_ = evaluate_acc(net,iter_test,params)
        print("this is %d epoch, loss is %.4f, train accurate is %.4f , test accturate is %.4f" %
              (i,train_l_sum/n,train_acc_sum/n,test_acc_))





