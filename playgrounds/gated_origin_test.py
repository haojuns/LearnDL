import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import sys
sys.path.append('/home/sura/learn_dl/')
from gated_backward import GatedLayer,GatedFunction
from mevo_DataClass import update_selected_indices
import matplotlib.pyplot as plt

torch.manual_seed(24)

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
class ConvNet_GatedLayer(nn.Module):
    def __init__(self):
        super(ConvNet_GatedLayer, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3,2,1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3,2,1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 110)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(110, 100)

    def forward(self, x):
        x.requires_grad_(True)
        x = GatedLayer(self.conv1,x)
        x = self.relu1(x)
        x = self.pool1(x)
        #x = GatedLayer(self.relu1,x)
        #x = GatedLayer(self.pool1,x)
        x = GatedLayer(self.conv2,x)
        x = self.relu2(x)
        x = self.pool2(x)
        #x = GatedLayer(self.relu2,x)
        #x = GatedLayer(self.pool2,x)
        x = x.view(x.size(0), -1)
        x = GatedLayer(self.fc1,x)
        x = self.relu3(x)
        #x = F.relu(x)
        x = GatedLayer(self.fc2,x)
        x = self.relu4(x)
        #x = F.relu(x)
        x = GatedLayer(self.fc3,x)
        return x
    
# 定义一个简单的网络
class LinearNet_GatedLayer(nn.Module):
    def __init__(self):
        super(LinearNet_GatedLayer, self).__init__()
        self.fc1 = nn.Linear(6000, 15000)
        self.fc2 = nn.Linear(15000, 30000)
        self.fc3 = nn.Linear(30000, 24000)
        self.fc4 = nn.Linear(24000, 100)

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
        self.fc1 = nn.Linear(16000, 30000)
        self.fc2 = nn.Linear(30000, 6400)
        self.fc3 = nn.Linear(6400, 24000)
        self.fc4 = nn.Linear(24000, 100)

    def forward(self, x):
        #x = self.fc1(x)
        x = self.fc1(x)
        #x = GatedFunction.apply(self.fc1,x)
        x = self.fc2(x)
        #x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
log_dir = os.path.join('log', formatted_time)
os.makedirs(log_dir, exist_ok=True)

def LinearNet_test():
    net = LinearNet_GatedLayer()
    #net = LinearNet()
    net.train()
    '''for name, module in net.named_modules():
        module.register_backward_hook(print_grads_and_activations)'''
    # 假设根据损失筛选后的样本索引如下

    batch_size=256
    # 假设我们已经有一批输入数据和标签
    inputs = torch.randn(batch_size, 6000)
    labels = torch.randn(batch_size,100)

    start=time.time()
    backward_times,all_times=[],[]
    
    '''with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=4),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) as prof:'''
    for num in range(batch_size):
        start=time.time()
        backward_time=[]
        for i in range(13):
            print('\nEpoch '+str(i))
            # 计算损失
            outputs = net(inputs)

            #loss=nn.functional.mse_loss(outputs,labels)
            with torch.enable_grad():
                all_loss=nn.functional.mse_loss(outputs,labels,reduction='none')
                sample_losses = torch.mean(all_loss,dim=1)
                selected_indices=torch.nonzero((sample_losses > 1)& (sample_losses < 1.2))
                selected_losses = torch.gather(sample_losses, dim=0, index=selected_indices.squeeze())
                selected_loss = torch.mean(selected_losses)
                
            selected_indices=selected_indices.squeeze().tolist()   
            selected_indices=list(range(num))
            #selected_indices=[0]#, 1, 3, 6, 8, 11, 12, 13, 14, 17, 20, 23, 24, 27, 28, 29, 30, 32, 34, 38, 40, 42, 43, 44, 45, 46, 49, 50, 52, 54, 55, 59]

            update_selected_indices(selected_indices)

            
            # 使用筛选后的样本创建新的损失
            '''selected_outputs = outputs[selected_indices]
            selected_labels = labels[selected_indices]

            selected_loss = criterion(selected_outputs, selected_labels) / len(selected_indices)'''

            # 反向传播
            optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
            optimizer.zero_grad()

            backward_start=time.time()
            selected_loss.backward()
            time_1=time.time()-backward_start
            backward_time.append(time_1)
            
            optimizer.step()
            #prof.step()

        time_2=time.time()-start
        mean_value = sum(backward_time) / len(backward_time)
        backward_times.append(mean_value)
        all_times.append(time_2)
        #print(time_1,torch.sum(sample_losses))
    plt.plot(list(range(batch_size)), backward_times, label='backward time(s)')
    plt.plot(list(range(batch_size)), all_times, label='all time(s)')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (s) ')
    plt.title('Computational Time vs Batch Size')
    plt.legend()
    plt.show()
    plt.savefig('computational_time.png')
        
        
    
def ConvNet_test():
    #net = LinearNet_GatedLayer()
    net = ConvNet_GatedLayer()
    net.train()
    '''for name, module in net.named_modules():
        module.register_backward_hook(print_grads_and_activations)'''
    # 假设根据损失筛选后的样本索引如下

    # 假设我们已经有一批输入数据和标签
    inputs = torch.randn(64, 3,128,128)
    labels = torch.randn(64,100)
    #selected_indices = [0, 1, 4, 5, 9, 12, 15, 17, 19, 20, 23, 27, 30, 31]

    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    start=time.time()
    for i in range(50):
        
        # 计算损失
        outputs = net(inputs)

        #loss=nn.functional.mse_loss(outputs,labels)
        with torch.enable_grad():
            all_loss=nn.functional.mse_loss(outputs,labels,reduction='none')
            sample_losses = torch.mean(all_loss,dim=1)
            selected_indices=torch.nonzero((sample_losses > 0.97)& (sample_losses < 1.3))
            selected_losses = torch.gather(sample_losses, dim=0, index=selected_indices.squeeze())
            selected_loss = torch.mean(selected_losses)
            
        selected_indices=selected_indices.squeeze().tolist()   
        selected_indices=list(range(64))
        
        print(len(selected_indices),torch.sum(sample_losses))     
        update_selected_indices(selected_indices)

        #global SELECTED_INDICES
        #SELECTED_INDICES = selected_indices
        
        # 使用筛选后的样本创建新的损失
        '''selected_outputs = outputs[selected_indices]
        selected_labels = labels[selected_indices]

        selected_loss = criterion(selected_outputs, selected_labels) / len(selected_indices)'''

        # 反向传播
        
        optimizer.zero_grad()

        #start=time.time()
        selected_loss.backward()
        #time_1=time.time()-start
        
        optimizer.step()

    time_1=time.time()-start
    print(time_1,torch.sum(sample_losses))
    
#LinearNet_test()
ConvNet_test()
