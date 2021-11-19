from utils import d2l
import torch
from torch import nn

from torch.nn import init


num_input = 784
num_output = 10
batch_size = 128
num_hiddens = 256
epochs = 10
lr = 0.5


net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_input,num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens,num_output))

for param in net.parameters():
    init.normal_(param,mean = 0.0,std=0.1)


train_iter,test_iter = d2l.iter_pic(batch_size)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(),lr=0.5)

if __name__=="__main__":
    d2l.train(net,train_iter,test_iter,epochs,batch_size,loss,optimizer,net.parameters(),lr=lr)

