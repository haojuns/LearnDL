import torch
#import eigenpy 
import numpy as np
import time
#import eigenpy.numpy as e_np
import os
import ctypes
from ctypes.util import find_library

# 指定 OpenBLAS 库的路径
openblas_path = "/opt/OpenBLAS/lib/libopenblas.so"
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# 生成随机矩阵
m = 2000
n = 1000
p = 1500
a = np.random.rand(m, n).astype(np.float32)
b = np.random.rand(n, p).astype(np.float32)

# 使用 PyTorch 计算矩阵乘法
start = time.time()
c_torch = torch.matmul(torch.from_numpy(a), torch.from_numpy(b))
end = time.time()
print("PyTorch time: {:.4f} s".format(end - start))

'''# 使用 Eigen 计算矩阵乘法
start = time.time()
a_eigen = e_np.MatrixXd(a)
b_eigen = e_np.MatrixXd(b)
c_eigen = a_eigen * b_eigen
end = time.time()
print("Eigen time: {:.4f} s".format(end - start))'''

# 使用 NumPy 计算矩阵乘法
start = time.time()
c_numpy = np.matmul(a, b)
end = time.time()
print("NumPy time: {:.4f} s".format(end - start))