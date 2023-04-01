import time
import torch
import torch.nn as nn
import numpy as np
import new_checkpoint as checkp

x = torch.randn(64,20)
y = torch.randn(64,10)


class MyNet(nn.Module):
    def __init__(self, save_memory=False):
        super(MyNet, self).__init__()

        self.linear1 = nn.Linear(20, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, 2000)
        self.linear4 = nn.Linear(2000, 10)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.linear5 = nn.Linear(2, 2)

        self.save_memory = save_memory

    def forward(self, x):
        if self.save_memory:
            x.requires_grad_(True)
            x = checkp.checkpoint(self.linear1, x)
            x = self.relu(x)
            #x = self.linear2(x)
            x = checkp.checkpoint(self.linear3, x)
            x = self.dropout(x)
            x = self.linear4(x)
            #x = cp.checkpoint(self.linear3, x)
        else:
            x = self.linear1(x)
            x = self.relu(x)
            #x = self.linear2(x)
            x = self.linear3(x)
            x = self.dropout(x)#对上一层的输出张量进行概率性置0
            x = self.linear4(x)
            
        return x


net = MyNet(save_memory=True)
# train() enables some modules like dropout, and eval() does the opposit
net.train()

# set the optimizer where lr is the learning-rate
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = nn.CrossEntropyLoss()

for epoch in range(50):
    '''if epoch % 5 == 0:
        # call eval() and evaluate the model on the validation set
        # when calculate the loss value or evaluate the model on the validation set,
        # it's suggested to use "with torch.no_grad()" to pretrained the memory. Here I didn't use it.
        net.eval()
        out = net(x)
        loss = loss_func(out, y)
        print(loss.detach().numpy())
        # call train() and train the model on the training set
        net.train()'''

    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    
    start=time.time()
    loss.backward()
    once_time=time.time()-start
    print(str(once_time))
    
    
    optimizer.step()

    '''if epoch % 5000 == 0:
        net.eval()
        out = net(x)
        loss = loss_func(out, y)
        #print(loss.detach().numpy())
        #print('----')
        net.train()

    if epoch % 1000 == 0:
        # adjust the learning-rate
        # weight decay every 1000 epochs
        lr = optimizer.param_groups[0]['lr']
        lr *= 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr'''


