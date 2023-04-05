import torch
import torch.nn.functional as F

def compute_grads(input, weight, bias, grad_output):
    grad_input = F.conv_transpose2d(grad_output, weight, padding=1)
    grad_weight = F.conv2d(input.unsqueeze(0), grad_output.flip(2, 3).unsqueeze(1).permute(1, 0, 2, 3))
    grad_bias = grad_output.sum(dim=(0, 2, 3))

    return grad_input, grad_weight, grad_bias

# 假设输入张量、权重、偏置和grad_output如下：
input = torch.randn(1, 3, 3, 3)
weight = torch.randn(2, 3, 3, 3)
bias = torch.randn(2)
grad_output = torch.randn(1, 2, 3, 3)

grad_input, grad_weight, grad_bias = compute_grads(input, weight, bias, grad_output)
print("grad_input:", grad_input)
print("grad_weight:", grad_weight)
print("grad_bias:", grad_bias)
