import torch
import time

# 定义网络
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.fc = torch.nn.Linear(32*32*32, 10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

model = MyNet()

size=32
# 构造输入和目标
x = torch.randn(size, 3, 32, 32)
y = torch.randint(0, 10, (size,))

# 分别计算64个样本的前向传播时间
start_time = time.time()
for i in range(size):
    output = model(x[i:i+1])
    loss = torch.nn.functional.cross_entropy(output, y[i:i+1])
    loss.backward()
end_time = time.time()
print("64个样本分别计算的前向传播时间为：", end_time-start_time)

# 一次计算64个样本的前向传播时间
start_time = time.time()
output = model(x)
loss = torch.nn.functional.cross_entropy(output, y)
loss.backward()
end_time = time.time()
print("一次计算64个样本的前向传播时间为：", end_time-start_time)

'''# 打印梯度信息
print("梯度信息：")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"参数 {name} 的梯度大小为：{param.grad.norm()}")    '''
