import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import warnings
import weakref
from typing import Any, Iterable, List, Tuple
import sys
sys.path.append('/home/sura/learn_dl/')
sys.path.append('/home/sura/learn_dl/playgrounds')
from mevo_DataClass import access_selected_indices
#from gated_origin_test import SELECTED_INDICES


'''__all__ = [
    "detach_variable", "GatedLayer","GatedFunction"
]'''

def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:#这个函数把输入tensor的grad_fn置none，然后保留原始requires_grad属性。推测应该是为了避免后面的backward函数一下反传到底了
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
        


def LinearBackward(grad_output, input, function,batch_size,*args):
    
    # Convert PyTorch tensors to NumPy arrays
    grad_output_np = grad_output.numpy()
    
    input_np = input.numpy()
    weight_np = function.weight.data.numpy()
    #print(grad_output_np.shape,input_np.shape,weight_np.shape)

    time_1 = time.time()

    # Calculate input gradients by matrix multiplication
    grad_input_np = np.matmul(grad_output_np, weight_np)

    # Calculate weight gradients by matrix multiplication
    grad_weight_np = np.matmul(grad_output_np.T, input_np)
    
    time_2 = time.time()

    # Calculate bias gradients by summing up grad_output along axis 0
    grad_bias_np = np.sum(grad_output_np, axis=0)

    

    #print(str(function)+': '+str(time_2-time_1))
    # Convert NumPy arrays to PyTorch tensors
    grad_input = torch.from_numpy(grad_input_np)
    grad_weight = torch.from_numpy(grad_weight_np)
    grad_bias = torch.from_numpy(grad_bias_np)


    #print(end - start)
    fill_shape = (batch_size-len(grad_input), grad_input.shape[1])
    fill_tensor = torch.zeros(fill_shape)
    grad_input = torch.cat([grad_input, fill_tensor], dim=0)
    
    time_3 = time.time()
    
    #print(time_3-time_2,time_2-time_1)

    return grad_input, grad_weight, grad_bias


def LinearBackward_torch(grad_output, input, function,batch_size):
    
    # Convert PyTorch tensors to NumPy arrays
    print(grad_output.size(0))
    weight = function.weight.data

    time_1 = time.time()
    # Calculate input gradients by matrix multiplication
    #grad_input_np = np.matmul(grad_output_np, weight_np)
    grad_input = torch.matmul(grad_output, weight)

    # Calculate weight gradients by matrix multiplication
    #grad_weight_np = np.matmul(grad_output_np.T, input_np)
    grad_weight = torch.matmul(grad_output.T, input)

    # Calculate bias gradients by summing up grad_output along axis 0
    #grad_bias_np = np.sum(grad_output_np, axis=0)
    grad_bias = torch.sum(grad_output, axis=0)
    time_2 = time.time()
    print(str(function)+': '+str(time_2-time_1))

    #print(end - start)
    fill_shape = (batch_size-len(grad_input), grad_input.shape[1])
    fill_tensor = torch.zeros(fill_shape)
    grad_input = torch.cat([grad_input, fill_tensor], dim=0)
    
    time_3 = time.time()

    return grad_input, grad_weight, grad_bias




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

#def Conv2dBackward(input, weight, grad_output, needs_input_grad,padding,stride):
def Conv2dBackward(grad_output, input, function, batch_size):
    grad_input = grad_weight = None
    if grad_output is None:
        return grad_input, grad_weight
    
    padding=function.padding[0]
    stride=function.stride[0]
    weight=function.weight.data
    bias=function.bias.data
    
    expanded_grad_output = expand_grad_output(grad_output, input.shape[2], input.shape[3], grad_output.shape[2], grad_output.shape[3], weight.shape[2], weight.shape[3], padding=padding, stride=stride, batch=input.shape[0])


    input_pad=torch.nn.functional.pad(input, (padding, padding, padding, padding, 0, 0, 0, 0))
    grad_input_pad=torch.nn.functional.pad(input, (padding, padding, padding, padding, 0, 0, 0, 0))

    #grad_input_pad
    gop = nn.ZeroPad2d(weight.shape[2] - 1)(expanded_grad_output)
    kk = torch.rot90(weight, 2, (2, 3))  # 旋转180度
    kk = torch.transpose(kk, 0, 1)
    grad_input_pad = F.conv2d(gop, kk)
    
    #grad_weight
    input_ = torch.transpose(input_pad, 0, 1)
    grad_output_ = torch.transpose(expanded_grad_output, 0, 1)
    grad_weight = F.conv2d(input_, grad_output_).transpose(0, 1)
        
    if padding > 0:
        grad_input = grad_input_pad[:,:,padding:-padding, padding:-padding]
    else:
        grad_input = grad_input_pad
    
    #grad_bias
    grad_bias = grad_output.sum(dim=(0, 2, 3))
    
    fill_shape = (batch_size-len(grad_input), grad_input.shape[1],grad_input.shape[2],grad_input.shape[3])
    fill_tensor = torch.zeros(fill_shape)
    grad_input = torch.cat([grad_input, fill_tensor], dim=0)
    
    return grad_input, grad_weight, grad_bias





class GatedFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,run_function,final_layer: bool,*args):
        ctx.run_function=run_function
        

        ctx.inputs = []#保存所有non-tensor-type inputs
        ctx.tensor_indices = []#保存所有tensor-type inputs在all_inputs中的索引
        tensor_inputs = []#保存所有tensor-type inputs
        
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)
                
        #with torch.enable_grad():
        outputs = run_function(*args)
        tensor_inputs.append(outputs)
        ctx.save_for_backward(*tensor_inputs)#存tensor
        
        ctx.final_layer=final_layer
        ctx.batch_size=outputs.shape[0]
        
        return outputs
    
    @staticmethod
    def backward(ctx,*args):
        
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        
        
        tensors = ctx.saved_tensors[:-1]#取tensor
        outputs = ctx.saved_tensors[-1]
        
        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]
        
        #detached_inputs = detach_variable(tuple(inputs))
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        
        #################################### run backward() with only tensor that requires grad#这个模块负责利用重新计算得到的output和后层传过来的梯度，进行反向传播####
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary")
            

        selected_indices=access_selected_indices()
        selected_indices=selected_indices if ctx.final_layer else list(range(len(selected_indices)))
        backward_func=str(ctx.run_function).split('(')[0]+'Backward'
        
        with torch.no_grad():
            if backward_func in ['LinearBackward','Conv2dBackward']: 
                grads_numpy=getattr(sys.modules[__name__],backward_func)(args[0][selected_indices],inputs[0][selected_indices],ctx.run_function,ctx.batch_size)
            else:
                print('error!')
                #grads_numpy=LinearBackward(args[0][selected_indices],inputs[0][selected_indices],ctx.run_function.weight.data)
                #print(str(ctx.run_function)+'\n')

        ctx.run_function.weight.grad=grads_numpy[1]
        ctx.run_function.bias.grad=grads_numpy[2]
        
        
        grads = (grads_numpy[0], )

        return (None,None) + grads#这里需要返回的是，对应forward除ctx之外的所有input的梯度

def GatedLayer(function,*args,**kwargs):
    
    final_layer = kwargs.pop('final_layer', False)

    return GatedFunction.apply(function,final_layer, *args)











