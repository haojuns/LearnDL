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
print(f"Matrix multiplication of two 32x3x1024x1024 matrices took {end_time - start_time:.6f} seconds.")
print(f"Total FLOPs: {flops_count}")


# 两个 14x3x1024x1024 的矩阵相乘
c = np.random.rand(32,1234)
d = np.random.rand(1234,10)

m, n, k = c.shape[0], d.shape[1], c.shape[1]  # 矩阵乘法的维度
flops_count = 2 * m * n * k  # 计算浮点运算次数

start_time = time.time()
out_2 = np.matmul(c, d)
end_time = time.time()
print(f"Matrix multiplication of two 14x3x1024x1024 matrices took {end_time - start_time:.6f} seconds.")
print(f"Total FLOPs: {flops_count}")

