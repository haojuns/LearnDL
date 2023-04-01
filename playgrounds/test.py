import torch

x = torch.tensor([1, 2, 3], dtype=torch.float32,requires_grad=True)
y = torch.tensor([4, 5, 6], dtype=torch.float32,requires_grad=True)

# 正常计算，需要计算梯度
z = x + y
loss = z.sum()
loss.backward()

# 使用torch.no_grad()禁止计算梯度
#with torch.no_grad():
    # 下面的计算将不会被记录在计算图中，也不会计算梯度
a = x + y
b = x * y
c = a.mean() + b.mean()

# 查看x和y的梯度
print(x.grad)  # 输出：tensor([1., 1., 1.])
print(y.grad)  # 输出：tensor([1., 1., 1.])
