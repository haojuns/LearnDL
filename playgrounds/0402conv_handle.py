import time
from typing import Dict, Tuple
import torch.nn.functional as F
import numpy as np
import pytest
import torch


def conv2d_forward(input: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                   stride: int, padding: int) -> Dict[str, np.ndarray]:
    """2D Convolution Forward Implemented with NumPy

    Args:
        input (np.ndarray): The input NumPy array of shape (H, W, C).
        weight (np.ndarray): The weight NumPy array of shape
            (C', F, F, C).
        bias (np.ndarray | None): The bias NumPy array of shape (C').
            Default: None.
        stride (int): Stride for convolution.
        padding (int): The count of zeros to pad on both sides.

    Outputs:
        Dict[str, np.ndarray]: Cached data for backward prop.
    """
    h_i, w_i, c_i,n_i = input.shape
    c_o, f, f_2, c_k = weight.shape

    assert (f == f_2)
    assert (c_i == c_k)
    assert (bias.shape[0] == c_o)

    input_pad = np.pad(input, [(padding, padding), (padding, padding), (0, 0),(0,0)])

    def cal_new_sidelngth(sl, s, f, p):
        return (sl + 2 * p - f) // s + 1

    h_o = cal_new_sidelngth(h_i, stride, f, padding)
    w_o = cal_new_sidelngth(w_i, stride, f, padding)
    n_o=n_i

    output = np.empty((h_o, w_o, c_o,n_o), dtype=input.dtype)
    for i_o in range(n_o):
        for i_h in range(h_o):
            for i_w in range(w_o):
                for i_c in range(c_o):
                    h_lower = i_h * stride
                    h_upper = i_h * stride + f
                    w_lower = i_w * stride
                    w_upper = i_w * stride + f
                    input_slice = input_pad[h_lower:h_upper, w_lower:w_upper, :,i_o]
                    kernel_slice = weight[i_c]
                    output[i_h, i_w, i_c,i_o] = np.sum(input_slice * kernel_slice)
                    output[i_h, i_w, i_c,i_o] += bias[i_c]

    cache = dict()
    cache['Z'] = output
    cache['W'] = weight
    cache['b'] = bias
    cache['A_prev'] = input
    return cache


def conv2d_backward_numpy(dZ: np.ndarray, cache: Dict[str, np.ndarray], stride: int,
                    padding: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D Convolution Backward Implemented with NumPy

    Args:
        dZ: (np.ndarray): The derivative of the output of conv.
        cache (Dict[str, np.ndarray]): Record output 'Z', weight 'W', bias 'b'
            and input 'A_prev' of forward function.
        stride (int): Stride for convolution.
        padding (int): The count of zeros to pad on both sides.

    Outputs:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The derivative of W, b,
            A_prev.
    """
    W = cache['W']
    b = cache['b']
    A_prev = cache['A_prev']
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    dA_prev = np.zeros(A_prev.shape)

    _, _, c_i,n_i = A_prev.shape
    c_o, f, f_2, c_k = W.shape
    h_o, w_o, c_o_2,n_o = dZ.shape

    assert (f == f_2)
    assert (c_i == c_k)
    assert (c_o == c_o_2)

    A_prev_pad = np.pad(A_prev, [(padding, padding), (padding, padding),(0,0),(0, 0)])
    dA_prev_pad = np.pad(dA_prev, [(padding, padding), (padding, padding),(0,0),(0, 0)])

    for i_n in range(n_o):
        for i_h in range(h_o):
            for i_w in range(w_o):
                for i_c in range(c_o):
                    h_lower = i_h * stride
                    h_upper = i_h * stride + f
                    w_lower = i_w * stride
                    w_upper = i_w * stride + f

                    input_slice = A_prev_pad[h_lower:h_upper, w_lower:w_upper, :,i_n]
                    # forward
                    # kernel_slice = W[i_c]
                    # Z[i_h, i_w, i_c] = np.sum(input_slice * kernel_slice)
                    # Z[i_h, i_w, i_c] += b[i_c]

                    # backward
                    dW[i_c] += input_slice * dZ[i_h, i_w, i_c,i_n]
                    dA_prev_pad[h_lower:h_upper,
                                w_lower:w_upper, :,i_n] += W[i_c] * dZ[i_h, i_w, i_c,i_n]
                    db[i_c] += dZ[i_h, i_w, i_c,i_n]

    if padding > 0:
        dA_prev = dA_prev_pad[padding:-padding, padding:-padding, :,:]
    else:
        dA_prev = dA_prev_pad
        
    return dW, db, dA_prev

def conv2d_backward_torch(dZ: np.ndarray, cache: Dict[str, np.ndarray], stride: int,
                    padding: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D Convolution Backward Implemented with PyTorch

    Args:
        dZ: (np.ndarray): The derivative of the output of conv.
        cache (Dict[str, np.ndarray]): Record output 'Z', weight 'W', bias 'b'
            and input 'A_prev' of forward function.
        stride (int): Stride for convolution.
        padding (int): The count of zeros to pad on both sides.

    Outputs:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The derivative of W, b,
            A_prev.
    """
    W = cache['W']
    b = cache['b']
    input = cache['A_prev']
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    #dA_prev = np.zeros(input.shape)

    _, _, c_i,_ = input.shape
    c_o, f, f_2, c_k = W.shape
    h_o, w_o, c_o_2, _ = dZ.shape

    assert (f == f_2)
    assert (c_i == c_k)
    assert (c_o == c_o_2)

    A_prev_pad = np.pad(input, [(padding, padding), (padding, padding), (0, 0),(0,0)])
    #dA_prev_pad = np.pad(dA_prev, [(padding, padding), (padding, padding),(0, 0),(0,0)])

    # Convert numpy arrays to PyTorch tensors
    torch_A_prev_pad = torch.from_numpy(A_prev_pad)
    torch_W = torch.from_numpy(W)
    torch_dZ = torch.from_numpy(dZ)

    if b is None:
        torch_b = None
    else:
        torch_b = torch.from_numpy(b)
        
    torch_dZ=torch_dZ.permute(3,2,0,1)
    torch_W=torch_W.permute(0,3,1,2)
    torch_A_prev_pad=torch_A_prev_pad.permute(3,2,0,1)

    # Calculate gradients using PyTorch functions
    torch_dA_prev_pad = F.conv_transpose2d(torch_dZ, torch_W, stride=stride, padding=padding)
    if padding > 0:
        torch_dA_prev = torch_dA_prev_pad[0, :, padding:-padding, padding:-padding].numpy()
    else:
        torch_dA_prev = torch_dA_prev_pad[0].numpy()
        
    torch_dZ=torch_dZ.permute(3,2,0,1)

    torch_dW = F.conv2d(torch_A_prev_pad, torch_dZ, stride=stride, padding=padding).numpy()
    db = np.sum(dZ, axis=(0, 1))

    # Check if bias is not None
    if b is not None:
        db = np.sum(dZ, axis=(0, 1))

    return torch_dW, db, torch_dA_prev



@pytest.mark.parametrize('c_i, c_o', [(16, 32)])
@pytest.mark.parametrize('kernel_size', [3])
@pytest.mark.parametrize('stride', [1, 2])
@pytest.mark.parametrize('padding', [0, 1])
def test_conv(n_i:int,c_i: int, c_o: int, kernel_size: int, stride: int, padding: int):

    # Preprocess
    input = np.random.randn(128, 128, c_i,n_i)
    weight = np.random.randn(c_o, kernel_size, kernel_size, c_i)
    bias = np.random.randn(c_o)

    torch_input = torch.from_numpy(np.transpose(
        input, (3,2, 0, 1))).requires_grad_()
    torch_weight = torch.from_numpy(np.transpose(
        weight, (0, 3, 1, 2))).requires_grad_()
    torch_bias = torch.from_numpy(bias).requires_grad_()

    # forward
    start=time.time()
    torch_output_tensor = torch.conv2d(torch_input, torch_weight, torch_bias,
                                       stride, padding)
    print('\npytorch_forward: '+str(time.time()-start))
    torch_output = np.transpose(
        torch_output_tensor.detach().numpy(), ( 2, 3,1,0))

    start=time.time()
    cache = conv2d_forward(input, weight, bias, stride, padding)
    print('numpy_forward: '+str(time.time()-start))
    numpy_output = cache['Z']

    assert np.allclose(torch_output, numpy_output)

    # backward
    torch_sum = torch.sum(torch_output_tensor)
    start=time.time()
    torch_sum.backward()
    print('pytorch_backward: '+str(time.time()-start))
    torch_dW = np.transpose(torch_weight.grad.numpy(), (0, 2, 3, 1))
    torch_db = torch_bias.grad.numpy()
    torch_dA_prev = np.transpose(torch_input.grad.numpy(),
                                 ( 2, 3,1,0))

    dZ = np.ones(numpy_output.shape)
    start=time.time()
    #dW, db, dA_prev = conv2d_backward_torch(dZ, cache, stride, padding)
    dW, db, dA_prev = conv2d_backward_numpy(dZ, cache, stride, padding)
    print('numpy_backward: '+str(time.time()-start))

    assert np.allclose(dW, torch_dW)
    assert np.allclose(db, torch_db)
    assert np.allclose(dA_prev, torch_dA_prev)
    
test_conv(2,6,7,3,2,1)