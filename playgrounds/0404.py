import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch import Tensor
import time

class MyConv2dFunc(torch.autograd.Function):
    
    @staticmethod
    def expand_grad_output(grad_output, input_width, input_height, output_height, output_width, filter_width, filter_height, padding, stride, batch):
        depth = grad_output.shape[1]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时sensitivity map的大小
        expanded_width = (input_width - filter_width + 2 * padding + 1)
        expanded_height = (input_height - filter_height + 2 * padding + 1)
        # 构建新的sensitivity_map
        expand_array = torch.zeros((batch, depth, expanded_height, expanded_width),dtype=grad_output.dtype)

        # 创建一个用于标识stride位置的张量
        i_indices = torch.arange(0, output_height * stride, step=stride)
        j_indices = torch.arange(0, output_width * stride, step=stride)
        i_indices, j_indices = torch.meshgrid(i_indices, j_indices,indexing='ij')

        # 通过矢量化操作将原始sensitivity map的误差值拷贝到新的张量中
        expand_array[:, :, i_indices, j_indices] = grad_output

        return expand_array

    @staticmethod
    def forward(ctx, input, weight,bias,padding,stride):
        ctx.save_for_backward(input, weight)
        ctx.padding=padding
        ctx.stride=stride
        output = F.conv2d(input, weight,bias,padding=padding,stride=stride)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        input, weight = ctx.saved_tensors
        padding=ctx.padding
        stride=ctx.stride
        
        grad_input,grad_weight,grad_bias=MyConv2dFunc.my_conv2d_backward(input, weight,grad_output, needs_input_grad=[True, True],padding=padding,stride=stride)
            
        return (grad_input, grad_weight,grad_bias,None,None)
        
    @staticmethod
    def my_conv2d_backward(input, weight, grad_output,needs_input_grad,padding,stride):
        grad_input = grad_weight = None
        if grad_output is None:
            return grad_input, grad_weight
        
        expanded_grad_output = MyConv2dFunc.expand_grad_output(grad_output, input.shape[2], input.shape[3], grad_output.shape[2], grad_output.shape[3], weight.shape[2], weight.shape[3], padding=padding, stride=stride, batch=input.shape[0])


        input_pad=torch.nn.functional.pad(input, (padding, padding, padding, padding, 0, 0, 0, 0))
        grad_input_pad=torch.nn.functional.pad(input, (padding, padding, padding, padding, 0, 0, 0, 0))

        if needs_input_grad[0]:
            # 反卷积
            gop = nn.ZeroPad2d(weight.shape[2] - 1)(expanded_grad_output)
            kk = torch.rot90(weight, 2, (2, 3))  # 旋转180度
            kk = torch.transpose(kk, 0, 1)
            grad_input_pad = F.conv2d(gop, kk)
        if needs_input_grad[1]:
            input_ = torch.transpose(input_pad, 0, 1)
            grad_output_ = torch.transpose(expanded_grad_output, 0, 1)
            grad_weight = F.conv2d(input_, grad_output_).transpose(0, 1)
            
        if padding > 0:
            grad_input = grad_input_pad[:,:,padding:-padding, padding:-padding]
        else:
            grad_input = grad_input_pad
            
        grad_bias = grad_output.sum(dim=(0, 2, 3))
            
        return grad_input, grad_weight, grad_bias


'''def expand_grad_output(grad_output, input_width, input_height, output_height,output_width,filter_width, filter_height, padding, stride,batch):
    depth = grad_output.shape[1]
    # 确定扩展后sensitivity map的大小
    # 计算stride为1时sensitivity map的大小
    expanded_width = (input_width - filter_width + 2 * padding + 1)
    expanded_height = (input_height - filter_height + 2 * padding + 1)
    # 构建新的sensitivity_map
    expand_array = torch.zeros((batch, depth, expanded_height, expanded_width))
    # 从原始sensitivity map拷贝误差值
    for n in range(batch):
        for i in range(output_height):
            for j in range(output_width):
                i_pos = i * stride
                j_pos = j * stride
                expand_array[n, : ,i_pos,j_pos] = grad_output[n, : ,i,j]
    return expand_array'''

def expand_grad_output(grad_output, input_width, input_height, output_height, output_width, filter_width, filter_height, padding, stride, batch):
    depth = grad_output.shape[1]
    # 确定扩展后sensitivity map的大小
    # 计算stride为1时sensitivity map的大小
    expanded_width = (input_width - filter_width + 2 * padding + 1)
    expanded_height = (input_height - filter_height + 2 * padding + 1)
    # 构建新的sensitivity_map
    expand_array = torch.zeros((batch, depth, expanded_height, expanded_width))

    # 创建一个用于标识stride位置的张量
    i_indices = torch.arange(0, output_height * stride, step=stride)
    j_indices = torch.arange(0, output_width * stride, step=stride)
    i_indices, j_indices = torch.meshgrid(i_indices, j_indices,indexing='ij')

    # 通过矢量化操作将原始sensitivity map的误差值拷贝到新的张量中
    expand_array[:, :, i_indices, j_indices] = grad_output

    return expand_array




def my_conv2d_backward(input, weight, grad_output, needs_input_grad,padding,stride):
    grad_input = grad_weight = None
    if grad_output is None:
        return grad_input, grad_weight
    
    expanded_grad_output = expand_grad_output(grad_output, input.shape[2], input.shape[3], output.shape[2], output.shape[3], weight.shape[2], weight.shape[3], padding=padding, stride=stride, batch=input.shape[0])


    input_pad=torch.nn.functional.pad(input, (padding, padding, padding, padding, 0, 0, 0, 0))
    grad_input_pad=torch.nn.functional.pad(input, (padding, padding, padding, padding, 0, 0, 0, 0))

    if needs_input_grad[0]:
        # 反卷积
        gop = nn.ZeroPad2d(weight.shape[2] - 1)(expanded_grad_output)
        kk = torch.rot90(weight, 2, (2, 3))  # 旋转180度
        kk = torch.transpose(kk, 0, 1)
        grad_input_pad = F.conv2d(gop, kk)
    if needs_input_grad[1]:
        input_ = torch.transpose(input_pad, 0, 1)
        grad_output_ = torch.transpose(expanded_grad_output, 0, 1)
        grad_weight = F.conv2d(input_, grad_output_).transpose(0, 1)
        
    if padding > 0:
        grad_input = grad_input_pad[padding:-padding, padding:-padding, :,:]
    else:
        grad_input = grad_input_pad
    
    #grad_input[0,0,0,0]=1
        
    return grad_input, grad_weight


import torch.nn as nn
from torch.autograd import gradcheck

# 定义输入大小
batch_size = 100
in_channels =16
out_channels = 32
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
dilation = 1

# 创建输入张量和随机权重
input = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True)

# 计算输出并随机创建梯度张量
output = nn.functional.conv2d(input, weight, stride=stride, padding=padding, dilation=dilation)
grad_output = torch.randn_like(output)

loss = output.sum()
start=time.time()
loss.backward()
print('torch_backward: '+str(time.time()-start))

input = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True)
bias = torch.randn(out_channels, requires_grad=True)
# 测试自定义反向传播函数和 PyTorch 反向传播函数是否生成相同的梯度
start=time.time()
res1 = my_conv2d_backward(input, weight,grad_output, needs_input_grad=[True, True],padding=padding,stride=stride)
print('my_backward: '+str(time.time()-start))
input = (torch.rand((2, 4, 10, 10), requires_grad=True, dtype=torch.double),
             torch.rand((6, 4, 5, 5), requires_grad=True, dtype=torch.double),torch.rand((6), requires_grad=True, dtype=torch.double),1,2)
test = torch.autograd.gradcheck(MyConv2dFunc.apply, input)
print(test)

#print(f"Are the gradients from the custom backward function and PyTorch's backward function equal? {res2}")
