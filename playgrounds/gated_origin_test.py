import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import sys
sys.path.append('/home/sura/learn_dl/')
from gated_backward import GatedLayer,GatedFunction

def print_grads_and_activations(module, grad_input, grad_output):
    print('\n'+str(module))
    print(f'Gradients shape:')
    for gi in grad_input:
        if gi is not None:
            print(gi.shape)
    print(f'Activations shape:')
    for go in grad_output:
        if go is not None:
            print(go.shape)

# 定义一个简单的网络
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = GatedLayer(self.conv1,x)
        x = GatedLayer(self.relu1,x)
        x = GatedLayer(self.pool1,x)
        x = GatedLayer(self.conv2,x)
        x = GatedLayer(self.relu2,x)
        x = GatedLayer(self.pool2,x)
        x = x.view(-1, 16 * 13 * 13)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
# 定义一个简单的网络
class LinearNet_GatedLayer(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(1600, 3000)
        self.fc2 = nn.Linear(3000, 6400)
        self.fc3 = nn.Linear(6400, 2400)
        self.fc4 = nn.Linear(2400, 100)

    def forward(self, x):
        #x = self.fc1(x)
        x.requires_grad_(True)#很关键，保证了x = GatedLayer(self.fc1,x)输出的x的grad_fn不是none，而是指向GatedLayer(self.fc1,x)的backward，否则反向传播到grad_fn的地方就直接断了
        x = GatedLayer(self.fc1,x)
        #x = GatedFunction.apply(self.fc1,x)
        x = GatedLayer(self.fc2,x)
        #x = self.fc2(x)
        x = GatedLayer(self.fc3,x)
        x = GatedLayer(self.fc4,x,final_layer=True)
        return x
    
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(1600, 3000)
        self.fc2 = nn.Linear(3000, 6400)
        self.fc3 = nn.Linear(6400, 2400)
        self.fc4 = nn.Linear(2400, 100)

    def forward(self, x):
        #x = self.fc1(x)
        x.requires_grad_(True)#很关键，保证了x = GatedLayer(self.fc1,x)输出的x的grad_fn不是none，而是指向GatedLayer(self.fc1,x)的backward，否则反向传播到grad_fn的地方就直接断了
        x = GatedLayer(self.fc1,x)
        #x = GatedFunction.apply(self.fc1,x)
        x = GatedLayer(self.fc2,x)
        #x = self.fc2(x)
        x = GatedLayer(self.fc3,x)
        x = GatedLayer(self.fc4,x,final_layer=True)
        return x


net = LinearNet()
net.train()
'''for name, module in net.named_modules():
    module.register_backward_hook(print_grads_and_activations)'''
# 假设根据损失筛选后的样本索引如下

# 假设我们已经有一批输入数据和标签
inputs = torch.randn(64, 1600)
labels = torch.randn(64,100)
selected_indices = [0, 1, 4, 5, 9, 12, 15, 17, 19, 20, 23, 27, 30, 31]

for i in range(20):

    # 计算损失
    outputs = net(inputs)

    loss=nn.functional.mse_loss(outputs,labels)
    # 使用筛选后的样本创建新的损失
    '''selected_outputs = outputs[selected_indices]
    selected_labels = labels[selected_indices]

    selected_loss = criterion(selected_outputs, selected_labels) / len(selected_indices)'''

    # 反向传播
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    optimizer.zero_grad()

    start=time.time()
    loss.backward()
    time_1=time.time()-start
    #print(time_1)
    optimizer.step()

    
