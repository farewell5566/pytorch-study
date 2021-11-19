import torch
import torchvision
from  torchvision.transforms import transforms

from torch import nn
from torch.nn import init
import torch.utils.data as Data

batch_size = 256

def iter_pic():
    mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,transform=transforms.ToTensor())

    iter_train = Data.DataLoader(mnist_train,shuffle=True,batch_size=256)
    iter_test = Data.DataLoader(mnist_test,shuffle=True,batch_size = 256)
    return iter_train,iter_test

class modelXC(nn.Module):
    def __init__(self):
        super(modelXC,self).__init__()
        self.linear = nn.Linear(784,10)

    def forward(self,X):
        return self.linear(X.view(X.shape[0],-1))

def lossXC():
    return nn.CrossEntropyLoss()

def optim(net,lr,batch_size):
    return torch.optim.SGD(net.parameters,lr = 0.1)

if __name__ == "__main__":
    iter_train,iter_test = iter_pic()
    for X,y in iter_train:
        modelXC()

        print(X.shape)
        print(y)
        break






