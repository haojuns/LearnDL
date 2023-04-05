import time

# 创建一个包含 100000 个整数的列表
lst = list(range(100000))

# 进行连续索引切片的测试
start_time = time.time()
for i in range(10000):
    x = lst[9000:9008]
end_time = time.time()
print("连续索引切片的时间：{:.6f} 秒".format(end_time - start_time))

# 进行随机索引切片的测试
indices = [3, 7, 123, 456, 789, 2345, 6789, 98765, 99999]
start_time = time.time()
for i in range(10000):
    x = [lst[i] for i in indices]
end_time = time.time()
print("随机索引切片的时间：{:.6f} 秒".format(end_time - start_time))
