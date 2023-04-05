import time
import numpy as np
import torch
import torch.nn.functional as F

np.random.seed(123)
list=[1, 2, 7, 8, 9, 12, 13, 16, 17, 19, 20, 23, 26, 29, 31, 35, 42, 44, 49]
a_np = np.random.rand(200, 10000)
#a_np=a_np[list]
b_np = np.random.rand(10000, 1000)

a_torch = torch.from_numpy(a_np)#.unsqueeze(0).unsqueeze(0)
b_torch = torch.from_numpy(b_np)#.unsqueeze(0).unsqueeze(0)

'''start = time.time()
c_py = a_np * b_np
end = time.time()
print("Python * time:", end - start)'''

start = time.time()
c_np = np.matmul(a_np, b_np)
end = time.time()
print("NumPy time:", end - start)

start = time.time()
c_torch = torch.matmul(a_torch, b_torch)
end = time.time()
print("PyTorch * time:", end - start)

'''start = time.time()
c_conv = F.conv2d(a_torch, b_torch)
end = time.time()
print("PyTorch conv2d time:", end - start)'''




