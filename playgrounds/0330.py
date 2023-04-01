import torch
import torch.nn as nn
from typing import Any, Iterable, List, Tuple

def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)


# 创建 Linear 类的实例对象
linear1 = Linear(12, 24)
linear2 = Linear(24, 10)


# 创建输入数据张量
input = torch.randn(32,12)

# 进行前向传递计算输出张量
tmp=linear1(input)
output = linear2(tmp)

# 计算损失函数值，例如使用 MSE 损失
target = torch.randn(32,10)
criterion = nn.MSELoss()
loss = criterion(output, target)
#loss_detach=loss.detach()
#loss.requires_grad = False#不能对 非叶子结点 直接修改这个属性

# 进行反向传播计算梯度
linear1.zero_grad()
linear2.zero_grad()

detached_output = detach_variable(tuple(output))
loss.backward()

# 输出输入数据张量和梯度张量
print("Input tensor:\n", input)
print("Gradient tensor:\n", input.grad)