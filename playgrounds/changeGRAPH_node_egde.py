import torch
import torch.nn as nn

class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = torch.matmul(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(input.t(), grad_output)
        return grad_input, grad_weight

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.func = MyFunc.apply
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.func(x, self.fc1.weight)
        x = self.fc2(x)
        return x

model = MyModel(784, 256, 10)
x = torch.randn(64, 784)
output = model(x)
