import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import alexnet
from torch.utils.checkpoint import checkpoint_sequential
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN
from torch.hub import _get_torch_home
import time
#print(_get_torch_home())
import torch.profiler

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


weights_path = "learn_dl/weights/alexnet-owt-4df8aa71.pth"
pretrained_weights = torch.load(weights_path)

# 加载数据集
# 这里你需要加载实际的数据集，以下仅为示例
trainset = MNIST(root='learn_dl/MNIST', train=True, download=True, transform=transform)  # 使用其他数据集替换 MNIST
train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0) 

valset = MNIST(root='learn_dl/MNIST', train=False, download=True, transform=transform)  # 使用其他数据集替换 MNIST
val_loader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=0)

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 加载预训练的 AlexNet 模型
model = alexnet(weights=None)
model.load_state_dict(pretrained_weights)
model=model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 将 AlexNet 的特征提取器分为若干段
segments = 4  # 选择合适数量的分段，视情况而定
features_sequential = nn.Sequential(*list(model.features.children()))
features_segments = nn.Sequential(
    *[features_sequential[i * len(features_sequential) // segments: (i + 1) * len(features_sequential) // segments]
    for i in range(segments)])

def main():
    # 创建日志目录
    formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    log_dir = os.path.join('log', formatted_time)
    os.makedirs(log_dir, exist_ok=True)

    # 设置Profiler
    

        # 训练模型
    num_epochs = 2
    for epoch in range(num_epochs):
        train_loss = 0.0
        tmp_time = time.time()
        
        with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=25, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) as prof:
        #每 4 个迭代（一个周期）会输出一个 JSON 文件（即每个 active 阶段输出一个文件）。在这些文件中，只有在 active 阶段收集的数据是有效的
        #prof.step() 确实是在每个迭代中执行的，但是 Profiler 只在 active 阶段收集数据。
            for i, data in enumerate(train_loader, 0):
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                # 使用梯度检查点运行特征提取器
                features = checkpoint_sequential(features_segments, segments, inputs)
                # 通过分类器生成预测结果
                outputs = model.classifier(features.view(features.size(0), -1))  

                #outputs = model(inputs)
                
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                #print(f"Batch {i + 1}, Loss: {loss.item()}")

                # 添加一个 Profiler 步骤
                prof.step()

        print(f"\nEpoch {epoch + 1}, Loss: {train_loss / len(train_loader)}")
        print(f"Time: {time.time() - tmp_time}\n")


if __name__ == '__main__':
    main()
