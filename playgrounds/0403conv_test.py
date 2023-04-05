import torch

import torch.nn.functional as Func

def compute_gradients(input, weight, bias, output, grad_output, stride, padding):
    N, C, H, W = input.shape
    F, C, kH, kW = weight.shape
    output_height, output_width = output.shape[-2:]

    # Compute forward pass
    conv_output = Func.conv2d(input, weight, bias, stride, padding)

    # Compute grad_input
    grad_input = Func.conv_transpose2d(grad_output, weight, stride=stride, padding=padding)
    
    # Compute grad_weight
    grad_weight = torch.zeros_like(weight)
    input_unfold = Func.unfold(input, (kH, kW), padding=padding, stride=stride)
    for i in range(F):
        grad_weight[i] = torch.sum(
            grad_output[:, i].unsqueeze(1) * input_unfold, dim=(0, -1)
        ).view_as(grad_weight[i])

    # Compute grad_bias
    grad_bias = torch.sum(grad_output, dim=(0, 2, 3))

    return grad_input, grad_weight, grad_bias

def generate_data_and_call_compute_gradients():
    # Generate random input tensor
    N, C, H, W = 64, 3, 128, 128
    input = torch.randn(N, C, H, W)

    # Generate random weight tensor
    F, kH, kW = 6, 3, 3
    weight = torch.randn(F, C, kH, kW)

    # Generate random bias tensor
    bias = torch.randn(F)

    # Generate random grad_output tensor
    output_height, output_width = H, W  # Assuming stride=1, padding=0
    grad_output = torch.randn(N, F, output_height, output_width)

    # Set stride and padding
    stride, padding = 1, 0

    # Compute the output
    output = Func.conv2d(input, weight, bias, stride, padding)

    # Call the compute_gradients function
    grad_input, grad_weight, grad_bias = compute_gradients(
        input, weight, bias, output, grad_output, stride, padding
    )

    print("Grad Input:\n", grad_input)
    print("Grad Weight:\n", grad_weight)
    print("Grad Bias:\n", grad_bias)

# Call the function
generate_data_and_call_compute_gradients()
