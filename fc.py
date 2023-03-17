#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from functools import reduce
import random
import numpy as np
from activators import SigmoidActivator, IdentityActivator


# 全连接层实现类
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, 
                 activator):
        '''
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1,
            (output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        前向计算
        input_array: 输入向量，维度必须等于input_size
        '''
        # 式2
        # 反向传播时更新权重矩阵需要用到权重矩阵的输入，所以这里先保存一下self.input
        # 反向传播时计算激活函数的导数不需要用到激活函数的输入，只需要用到输出（也就是下一层的输入）
        # ，所以不需要保留中间激活，直接矩阵乘法和激活一起算
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        # 式8
        # 因为先做矩阵乘法再经过激活层，所以本层输出delta还需要再乘一个激活层的导数
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

    def dump(self):
        print ('W: %s\nb:%s' % (self.W, self.b))

import time
import matplotlib.pyplot as plt
# 神经网络类
class Network(object):
    #所以模型定义的并不是层或者节点，而是面向权重定义的层间权重矩阵
    def __init__(self, layers):
        '''
        构造函数
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )

    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output
    
    def predict_time(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        start=time.time()
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        end=time.time()
        return output,end-start

    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''                
        for i in range(epoch):
            infer_times, back_times, update_times = [], [], []
            for d, label in enumerate(labels):
                infer_time, back_time, update_time = self.train_one_sample(label, data_set[d], rate)
                infer_times.append(infer_time)
                back_times.append(back_time)
                update_times.append(update_time)
            if i == 0:
                avg_infer=np.sum(infer_times)
                avg_back=np.sum(back_times)
                avg_update=np.sum(update_times)
                '''plt.scatter(infer_times, list(range(len(infer_times))), label="infer time")
                plt.scatter(back_times, list(range(len(back_times))), label="back time")
                plt.scatter(update_times, list(range(len(update_times))), label="update time")
                plt.legend()
                plt.xlabel("time (s)")
                plt.ylabel("sample index")
                plt.title("Time analysis of training one epoch")
                plt.show()'''  

    def train_one_sample(self, label, sample, rate):
        _,infer_time=self.predict_time(sample)
        _,back_time=self.calc_gradient_time(label)
        update_time=self.update_weight_time(rate)
        return infer_time,back_time,update_time
        

    def calc_gradient(self, label):        
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta
    
    def calc_gradient_time(self, label):
        start=time.time()
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        end=time.time()
        return delta,end-start

    def update_weight(self, rate):
        
        for layer in self.layers:
            layer.update(rate)
            
    def update_weight_time(self, rate):
        start=time.time()
        for layer in self.layers:
            layer.update(rate)
        end=time.time()
        
        return end-start

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def loss(self, output, label):
        return 0.5 * ((label - output) * (label - output)).sum()

    def gradient_check(self, sample_feature, sample_label):
        '''
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        '''

        # 获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        # 检查梯度
        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i,j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    fc.W[i,j] -= 2*epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    fc.W[i,j] += epsilon
                    print ('weights(%d,%d): expected - actural %.4e - %.4e') % (
                        i, j, expect_grad, fc.W_grad[i,j])


from bp import train_data_set


def transpose(args):#[ [images], [labels] ]
    return list(map(lambda arg: list(map(lambda line: np.array(line).reshape(len(line), 1), arg)), args))
'''np.array([1,2,3,4]).reshape(2,2,1):
    array([[[1],
        [2]],
       [[3],
        [4]]])'''
'''np.array([1,2,3,4]).reshape(4,1):
array([[1],
       [2],
       [3],
       [4]])
'''


class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        data = map(lambda m: 0.9 if number & m else 0.1, self.mask)
        return np.array(data).reshape(8, 1)

    def denorm(self, vec):
        binary = map(lambda i: 1 if i > 0.5 else 0, vec[:,0])
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x,y: x + y, binary)

def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256):
        n = normalizer.norm(i)
        data_set.append(n)
        labels.append(n)
    return labels, data_set

def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0;
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print ('correct_ratio: %.2f%%') % (correct / 256 * 100)


def test():
    labels, data_set = transpose(train_data_set())
    net = Network([8, 3, 8])
    rate = 0.5
    mini_batch = 20
    epoch = 10
    for i in range(epoch):
        net.train(list(labels), list(data_set), rate, mini_batch)
        print ('after epoch %d loss: %f') % (
            (i + 1),
            net.loss(labels[-1], net.predict(data_set[-1]))
        )
        rate /= 2
    correct_ratio(net)


def gradient_check():
    '''
    梯度检查
    '''
    labels, data_set = transpose(train_data_set())
    net = Network([8, 3, 8])
    net.gradient_check(data_set[0], labels[0])
    return net
