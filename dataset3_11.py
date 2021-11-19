import torch
import sys
import torchvision

import numpy as np

from matplotlib import pyplot as plt
#函数 torch.cat(A,B,num)  进行拼接， num为0 按行拼接，num为1 按列拼接。



num_train ,num_test,w_true,b_true = 100,100,[1.2,-3.4,5.6],5

num_epochs = 100

loss = torch.nn.MSELoss()

features = torch.randn((num_train + num_test,1))

features_ploys = torch.cat((features,torch.pow(features,2),torch.pow(features,3)),1)

labels = features_ploys[:,0] * w_true[0] + features_ploys[:,1]*w_true[1] + features_ploys[:,2] * w_true[2] +   b_true
labels_A = (features_ploys[:,0] * w_true[0] + features_ploys[:,1]*w_true[1] + features_ploys[:,2] * w_true[2] +   b_true)

labels_bias = torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)

labels += labels_bias

def semilogy(x_vals,y_vals,x_label,y_label,x2_vals=None,y2_vals=None,legend=None,figsize=(3.5,2.5)):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals,y2_vals,linestyle=":")
        plt.legend(legend)
    #plt.plot(x_vals.detach().numpy(),y_vals.detach().numpy())
    plt.show()


def fitAndPlot(train_features,test_features,train_labels,test_labels):
    net = torch.nn.Linear(train_features.shape[-1],1)
    batch_size = min(10,train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features,train_labels)
    iter_train = torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=batch_size)

    optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    train_ls,test_ls= [],[]
    for _ in range(num_epochs):
        for X,y in iter_train:
            l = loss(net(X),y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels =  train_labels.view(-1,1)
        test_labels = test_labels.view(-1,1)
        train_ls.append(loss(net(train_features),train_labels).item())
        test_ls.append(loss(net(test_features),test_labels).item())
    print("final epoch:train loss",train_ls[-1],",test loss:",test_ls[-1])
    semilogy(range(1,num_epochs + 1),train_ls,'epochs','loss',
             range(1,num_epochs + 1),test_ls,['train','test'])

    print('weight',net.weight.data, "\n bias",net.bias.data)


fitAndPlot(features_ploys[:num_train,:],features_ploys[num_train:,:],labels[:num_train],labels[num_train:])

# fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
#             labels[:n_train], labels[n_train:])
#
# fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],
#              labels[n_train:])