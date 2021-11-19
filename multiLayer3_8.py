import torch
from torch import nn

from matplotlib import pyplot as plt

def xyplt(X,Y,xname,yname):
    plt.plot(X.detach().numpy(),Y.detach().numpy())
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()


if __name__ =="__main__":
    X = torch.arange(-5,5,0.5,requires_grad=True)
    #X.requires_grad_(requires_grad=True)
    print(type(X))
    Y = X.relu()
    Y.sum().backward()
    xyplt(X,X.grad,"xVals","Relu(xVals)")