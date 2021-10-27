import sys

import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from time import time


import torch.utils.data as Data

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=True,download=True,transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=False,download=True,transform=transforms.ToTensor())

print(type(mnist_train))

print(len(mnist_test),len(mnist_train))


feature,label = mnist_train[0]

print(feature)

def getFashionLable(labels):
    text_label = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [ text_label[i] for i in labels]

X,y = [],[]
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])

_,figs = plt.subplots(1,len(X),figsize= (24,24))
for f,img,label in zip(figs,X,getFashionLable(y)):
    f.imshow(img.view((28,28)).numpy())
    f.set_title(label)
plt.show()



batch_size = 16
#用于读取数据
if sys.platform.startswith("win"):
    numWorkers = 0
else :
    numWorkers = 4
print(sys.platform)
print(numWorkers)
train_iter = Data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=numWorkers)
test_iter = Data.DataLoader(mnist_test,batch_size=batch_size,shuffle=True,num_workers=numWorkers)

start = time()
for X,y in train_iter:
    continue
print(time()-start )



