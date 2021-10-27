import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.optim as optim


import torch.utils.data as Data

#生成数据
num_input = 2
num_feature = 5000

true_w = [1.6,3.8]
true_b = 3.26

features = torch.randn((num_feature,num_input),dtype=torch.float32)
labels = features[:,0] * true_w[0] + features[:,1] * true_w[1] + true_b

labels += torch.tensor(np.random.normal(0.0,0.5),dtype=torch.float32)
print(features[0],labels[0])

from matplotlib import pyplot as plt
plt.scatter(features[:,1].numpy(),labels)
plt.show()


#读取数据

batch_size = 10
dataset = Data.TensorDataset(features,labels)
data_iter = Data.DataLoader(dataset,batch_size,shuffle=True)


#建模
# class lineNet(nn.Module):
#     def __init__(self,num_input,num_output):
#         super(lineNet, self).__init__()
#         self.linear = nn.Linear(num_input,num_output)
#
#     def forward(self,x):
#         y = self.linear(x)
#         return y
# net = lineNet(num_input,1)
# print(net.linear.weight)


#建模方法2
net = nn.Sequential(nn.Linear(num_input,1))

#初始化参数
init.normal_(net[0].weight,mean=0,std=0.01)
init.constant_(net[0].bias,val=0)

#查看待学习参数
for param in net.parameters():
    print(param)

#定义损失函数
loss = nn.MSELoss()

optimizer = optim.SGD(net.parameters(),lr = 0.03)
print(optimizer)


num_epochs = 10
for epoch in range(num_epochs):
    for X,y in data_iter:
        outPut = net(X)
        l = loss(outPut,y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print("epch %d,loss %f" %(epoch,l.item()))
print(net[0].weight)
print(net[0].bias)











