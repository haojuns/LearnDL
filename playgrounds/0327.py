import torch

# 创建模型和数据
model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1))
x = torch.randn(64, 10)
y = torch.randn(64, 1)

# 计算所有样本的loss和梯度
loss = torch.nn.functional.mse_loss(model(x), y)
#grads = torch.autograd.grad(loss, model.parameters())

# 创建掩码，只计算掩码中为True的样本相关的梯度
mask = torch.zeros(64, dtype=torch.bool)
mask[0:12] = True

# 计算掩码中为True的样本相关的梯度
masked_grads = torch.autograd.grad(loss, model.parameters(), grad_outputs=None, only_inputs=True, retain_graph=True, create_graph=False, allow_unused=True)
masked_grads = [g if m else None for g, m in zip(masked_grads, mask)]

# 更新模型参数
lr = 0.01
for param, grad in zip(model.parameters(), masked_grads):
    if grad is not None:
        param.data -= lr * grad.data
