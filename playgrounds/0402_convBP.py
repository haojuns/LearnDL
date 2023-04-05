import torch.nn.functional as F
import torch

# 生成输入张量 x，卷积核张量 w，偏置张量 b，输出张量 y 和输入梯度张量 dy
batch_size = 32
in_channels = 3
out_channels = 6
H_in = 10
W_in = 10
H_out = 8
W_out = 8
kernel_size = 3

x = torch.randn(batch_size, in_channels, H_in, W_in)  # 生成随机的输入张量
w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)  # 生成随机的卷积核张量
b = torch.randn(out_channels)  # 生成随机的偏置张量
y = F.conv2d(input=x, weight=w, bias=b, stride=1, padding=1)  # 计算卷积的前向传播
dy = torch.randn(y.size(0), y.size(1), y.size(2), y.size(3))  # 生成随机的输入梯度张量

# 计算卷积的步长和填充
stride = 1
padding = 1

# 计算卷积的转置
y_unfold = F.unfold(y, kernel_size=3, stride=stride, padding=padding)  # 将输出张量 y 展开为矩阵
w_rot = w.flip([2, 3]).transpose(0, 1)  # 将卷积核 w 旋转180度，然后对其进行转置
w_mat = w_rot.reshape(w_rot.shape[0], -1)  # 将旋转后的卷积核展平成二维矩阵
dy_mat = dy.view(dy.shape[0], -1)  # 将输入梯度张量展平成二维矩阵

# 计算卷积的矩阵乘积和偏置梯度
dx_mat = dy_mat.mm(w_mat)  # 计算输入张量 x 的展开矩阵和旋转后的卷积核的矩阵乘积
db = dy.view(dy.shape[0], -1).sum(dim=1)  # 计算偏置梯度，即输入梯度张量的和

# 将矩阵乘积的结果重塑成卷积的展开形式
dx_unfold = dx_mat.view(dy.shape[0], w.shape[1], w.shape[2], w.shape[3], y.shape[2], y.shape[3])  # 将矩阵乘积的结果重塑为卷积的展开形式

# 使用 F.conv2d() 函数计算卷积的反向传播结果和卷积核权重梯度
dx = F.conv2d(input=dx_unfold.view(-1, w.shape[1], w.shape[2], w.shape[3]),
              weight=w_rot,
              stride=stride,
              padding=padding)  # 计算输入梯度张量的反向传播结果
dw = F.conv2d(input=y_unfold.view(-1, 1, y.shape[2], y.shape[3]),
              weight=dy.view(dy.shape[0], -1, 1, 1),
              stride=stride,
              padding=padding)  # 计算卷积核权重的梯度，即输出梯度张量和展开后的输出张量的卷积

# 将卷积的反向传播结果重塑成输入张量的形状
dx = dx.view(dy.shape[0], x.shape[1], x.shape[2], x.shape[3])  # 将卷积的反向传播结果重塑为输入张量的形状

# 输出卷积层的输入梯度、卷积核权重梯度和偏置梯度
print(dx.shape)  # 输出卷积层的输入梯度张量的形状
print(dw.shape)  # 输出卷积核权重的梯度张量的形状
print(db.shape)  # 输出偏置的梯度张量的形状
