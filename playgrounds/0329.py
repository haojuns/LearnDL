import torch
from torch.autograd import Function
from torchviz import make_dot

class GradCoeff(Function):       
       
    @staticmethod
    def forward(ctx, x, coeff):                 # 模型前向
        ctx.coeff = coeff                       # 将coeff存为ctx的成员变量
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):             # 模型梯度反传
        return ctx.coeff * grad_output, None    # backward的输出个数，应与forward的输入个数相同，此处coeff不需要梯度，因此返回None

'''# 尝试使用
x = torch.tensor([2.], requires_grad=True)
ret = GradCoeff.apply(x, -0.1)                  # 前向需要同时提供x及coeff，设置coeff为-0.1
bard = ret ** 2                          
print(bard)                                      # tensor([4.], grad_fn=<PowBackward0>)
bard.backward()  
print(x.grad)                                   # tensor([-0.4000])，梯度已乘以相应系数'''

class Exp(Function):                    # 此层计算e^x

    @staticmethod
    def forward(ctx, i):                # 模型前向
        result = i.exp()
        ctx.save_for_backward(result)   # 保存所需内容，以备backward时使用，所需的结果会被保存在saved_tensors元组中；此处仅能保存tensor类型变量，若其余类型变量（Int等），可直接赋予ctx作为成员变量，也可以达到保存效果
        return result

    @staticmethod
    def backward(ctx, grad_output):     # 模型梯度反传
        result, = ctx.saved_tensors     # 取出forward中保存的result
        return grad_output * result     # 计算梯度并返回

# 尝试使用
x = torch.tensor([1.], requires_grad=True)  # 需要设置tensor的requires_grad属性为True，才会进行梯度反传
ret = Exp.apply(x) 
bard=ret **2# 使用apply方法调用自定义autograd function
print(bard)                                  # tensor([2.7183], grad_fn=<ExpBackward>)
#bard.backward()                              # 反传梯度
dot = make_dot(bard)
dot.render(filename='/home/sura/learn_dl/playgrounds', format='png')
print(x.grad)                               # tensor([2.7183])
