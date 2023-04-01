import numpy as np
import time
import torch
import torch.nn.functional as F
from scipy.signal import convolve2d

def convolve(input_data, kernel):
    output = np.zeros((input_data.shape[0], kernel.shape[0], input_data.shape[2] - 2, input_data.shape[3] - 2))
    for batch_idx in range(input_data.shape[0]):
        for kernel_idx in range(kernel.shape[0]):
            for channel_idx in range(input_data.shape[1]):
                output[batch_idx, kernel_idx] += convolve2d(input_data[batch_idx, channel_idx], kernel[kernel_idx, channel_idx], mode='valid')
    return output

# 创建输入数据和卷积核
input_data1 = np.random.rand(32, 3, 256, 256)
kernel1 = np.random.rand(64, 3, 3, 3)

input_data2 = np.random.rand(16, 3, 256, 256)
kernel2 = np.random.rand(64, 3, 3, 3)

# 计算第一个卷积运算的时间（使用numpy）
start_time = time.time()
#output1 = convolve(input_data1, kernel1)#20.686318 seconds
end_time = time.time()

print(f"Convolution with numpy (input shape 32x3x256x256 and kernel shape 64x3x3x3) took {end_time - start_time:.6f} seconds.")

# 计算第二个卷积运算的时间（使用numpy）
start_time = time.time()
#output2 = convolve(input_data2, kernel2)#10.341205 seconds
end_time = time.time()

print(f"Convolution with numpy (input shape 16x3x256x256 and kernel shape 64x3x3x3) took {end_time - start_time:.6f} seconds.")


####################################################################################




# 将numpy数组转换为PyTorch张量
input_data1_torch = torch.tensor(input_data1, dtype=torch.float32)
kernel1_torch = torch.tensor(kernel1, dtype=torch.float32)

input_data2_torch = torch.tensor(input_data2, dtype=torch.float32)
kernel2_torch = torch.tensor(kernel2, dtype=torch.float32)

# 计算第二个卷积运算的时间（使用PyTorch）
start_time = time.time()
output2_torch = F.conv2d(input_data2_torch, kernel2_torch, padding=1)
end_time = time.time()

print(f"Convolution with PyTorch (input shape 16x3x256x256 and kernel shape 64x3x3x3) took {end_time - start_time:.6f} seconds.")


# 计算第一个卷积运算的时间（使用PyTorch）
start_time = time.time()
output1_torch = F.conv2d(input_data1_torch, kernel1_torch, padding=1)
end_time = time.time()

print(f"Convolution with PyTorch (input shape 32x3x256x256 and kernel shape 64x3x3x3) took {end_time - start_time:.6f} seconds.")


####################################################################################

# 创建输入矩阵
a = torch.rand(16, 1234)
b = torch.rand(1234, 10)
# 创建输入矩阵
c = torch.rand(32, 1234)
d = torch.rand(1234, 10)

#warmup
torch.matmul(torch.rand(16, 1234), torch.rand(1234, 10))
torch.matmul(torch.rand(32, 1234), torch.rand(1234, 10))

# 计算矩阵乘法并记录时间
start_time = time.time()
z = torch.matmul(c, d)
end_time = time.time()

print(f"Matrix multiplication with PyTorch (input shape 32x1234 and 1234x10) took {end_time - start_time:.6f} seconds.")



# 计算矩阵乘法并记录时间
start_time = time.time()
z = torch.matmul(a, b)
end_time = time.time()

print(f"Matrix multiplication with PyTorch (input shape 16x1234 and 1234x10) took {end_time - start_time:.6f} seconds.")


##################################################################################
import numpy as np
import time

def warmup():
    a = np.random.rand(16, 1234)
    b = np.random.rand(1234, 10)
    np.matmul(a, b)
    c = np.random.rand(32, 1234)
    d = np.random.rand(1234, 10)
    np.matmul(c, d)

warmup()  # 热身操作

# 两个 32x3x1024x1024 的矩阵相乘
a = np.random.rand(16,1234)
b = np.random.rand(1234,10)

m, n, k = a.shape[0], b.shape[1], a.shape[1]  # 矩阵乘法的维度
flops_count = 2 * m * n * k  # 计算浮点运算次数

start_time = time.time()
out_1 = np.matmul(a, b)
end_time = time.time()
print(f"Matrix multiplication of two 16x3x1024x1024 matrices took {end_time - start_time:.6f} seconds.")
print(f"Total FLOPs: {flops_count}")


# 两个 14x3x1024x1024 的矩阵相乘
c = np.random.rand(32,1234)
d = np.random.rand(1234,10)

m, n, k = c.shape[0], d.shape[1], c.shape[1]  # 矩阵乘法的维度
flops_count = 2 * m * n * k  # 计算浮点运算次数

start_time = time.time()
out_2 = np.matmul(c, d)
end_time = time.time()
print(f"Matrix multiplication of two 32x3x1024x1024 matrices took {end_time - start_time:.6f} seconds.")
print(f"Total FLOPs: {flops_count}")



