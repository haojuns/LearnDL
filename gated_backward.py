import time
import numpy as np
import torch
import warnings
import weakref
from typing import Any, Iterable, List, Tuple

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

def LinearBackward(grad_output, input, weight):
    
    # Convert PyTorch tensors to NumPy arrays
    grad_output_np = grad_output.numpy()
    input_np = input.numpy()
    weight_np = weight.numpy()

    start = time.time()

    # Calculate input gradients by matrix multiplication
    grad_input_np = np.matmul(grad_output_np, weight_np)

    # Calculate weight gradients by matrix multiplication
    grad_weight_np = np.matmul(grad_output_np.T, input_np)

    # Calculate bias gradients by summing up grad_output along axis 0
    grad_bias_np = np.sum(grad_output_np, axis=0)

    end = time.time()

    # Convert NumPy arrays to PyTorch tensors
    grad_input = torch.from_numpy(grad_input_np)
    grad_weight = torch.from_numpy(grad_weight_np)
    grad_bias = torch.from_numpy(grad_bias_np)

    print(end - start)

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
            
        selected_indices = [0,1,2,3,5,6,7]
        #selected_indices=list(range(ctx.batch_size))
            
        selected_indices=selected_indices if ctx.final_layer else list(range(len(selected_indices)))
            
        with torch.no_grad():
            grads_numpy=LinearBackward(args[0][selected_indices],inputs[0][selected_indices],ctx.run_function.weight.data)
            print(str(ctx.run_function)+'\n')

        ctx.run_function.weight.grad=grads_numpy[1]
        ctx.run_function.bias.grad=grads_numpy[2]
        
        #c, h, w = grads_numpy[0].shape[1:]
        l = grads_numpy[0].shape[1]
        #fill_shape = (ctx.batch_size-len(selected_indices), c, h, w)
        fill_shape = (ctx.batch_size-len(selected_indices), l)
        fill_tensor = torch.zeros(fill_shape)
        grads = torch.cat([grads_numpy[0], fill_tensor], dim=0)
        #grads = torch.nn.functional.pad(grads_numpy[0], (0, 0, 0, 18), mode='constant', value=0)
        
        grads = (grads, )

        return (None,None) + grads#这里需要返回的是，对应forward除ctx之外的所有input的梯度

def GatedLayer(function,*args,**kwargs):
    
    final_layer = kwargs.pop('final_layer', False)

    return GatedFunction.apply(function,final_layer, *args)











