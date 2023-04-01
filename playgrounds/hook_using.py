import functools
import time
import torch

# 自定义钩子函数，用于在反向传播时打印梯度和中间激活值的shape
def print_grads_and_activations(module, grad_input, grad_output):
    print(f'\nGradients shape:')
    for gi in grad_input:
        if gi is not None:
            print(gi.shape)
    print(f'Activations shape:')
    for go in grad_output:
        if go is not None:
            print(go.shape)
            

def gated_gradients(module,grad_input, grad_output,selected_indices):
    zero_indices = [elem for i, elem in enumerate(list(range(len(grad_output[0])))) if i not in selected_indices]
    #print(grad_output)
    grad_output[0][zero_indices, :] = 0
    #print(grad_output)
    #i=1
     
def modify_input_output(module, grad_input, grad_output):
    # 修改输入和输出
    input = module.in_features
    output = module.output
    input = input[:1, :]
    output = output[:2, :]
    module.input = input
    module.output = output

# 创建一个简单的神经网络
model = torch.nn.Sequential(
    torch.nn.Linear(12, 24),
    torch.nn.ReLU(),
    torch.nn.Linear(24, 10)
)


#selected_indices = torch.tensor(selected_indices, dtype=torch.long)
selected_indices = [0, 1, 4, 5, 9, 12, 15, 17, 19, 20, 23, 27, 30, 31]

hook_func = functools.partial(gated_gradients, selected_indices=selected_indices)
#list(model.named_modules())[-1][-1].register_backward_hook(hook_func)#核心一步，用于选择是否将最后一层梯度置0
#实验结果：全0矩阵相乘并不会比非0矩阵相乘更快，所以这个思路没有用

list(model.named_modules())[-1][-1].register_backward_hook(modify_input_output)

# 注册钩子函数
for name, module in model.named_modules():
    module.register_backward_hook(print_grads_and_activations)
    



# 创建输入和目标张量
x = torch.randn(32, 12)
y = torch.randn(32, 10)

start=time.time()
for i in range(1):
    # 前向传播
    output = model(x)

    # 计算损失和梯度
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()
latency=time.time()-start
print(latency)
