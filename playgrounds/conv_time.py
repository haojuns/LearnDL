import time
import torch

# 定义卷积乘法的输入维度
input_channels = 1024
output_channels = 1024
input_height = 64
input_width = 64
kernel_size = 3

# 创建两个大张量
zeros = torch.zeros(input_channels, input_height, input_width)
random_tensor1 = torch.randn(input_channels, input_height, input_width)
random_tensor2 = torch.randn(output_channels, input_channels, kernel_size, kernel_size)

# 使用零张量计算卷积乘法并计时
start_time = time.time()
zeros_conv = torch.nn.functional.conv2d(zeros.unsqueeze(0), random_tensor2)
zeros_conv_time = time.time() - start_time

# 使用随机张量计算卷积乘法并计时
start_time = time.time()
random_conv = torch.nn.functional.conv2d(random_tensor1.unsqueeze(0), random_tensor2)
random_conv_time = time.time() - start_time

# 打印结果
print(f"Time taken for zeros conv: {zeros_conv_time:.6f}s")
print(f"Time taken for random conv: {random_conv_time:.6f}s")
