#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import struct
from fc import *
from datetime import datetime
import numpy as np


# 数据加载器基类
class Loader(object):
    def __init__(self, path, count):
        '''
        初始化加载器
        path: 数据文件路径
        count: 文件中的样本个数
        '''
        self.path = path
        self.count = count

    def get_file_content(self):
        '''
        读取文件内容
        '''
        f = open(self.path, 'rb')
        content = f.read()
        f.close() 
        return content

    def to_int(self, byte):#byte实际上是十进制类型的1，而不是\x01
        '''
        将unsigned byte字符转换为整数
        '''
        #bytes是在干嘛？
        return struct.unpack('B', bytes([byte]))[0]
#bytes(10):b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
#b = bytes(“Hello”, “utf-8”):b'Hello'
#str=struct.pack("ihb", 1, 2, 3) ---> str='\x01\x00\x00\x00\x02\x00\x03'(小端)

# 图像数据加载器
class ImageLoader(Loader):
    def get_picture(self, content, index):
        '''
        内部函数，从文件中获取图像
        '''
        start = index * 28 * 28 + 16 #以字节为基本单位，前16字节是元数据。每张图片28*28字节，一个字节是一个像素（0-255），行优先存储
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(self.to_int(content[start + i * 28 + j]))
        return picture

    def get_one_sample(self, picture):
        '''
        内部函数，将图像转化为样本的输入向量
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        '''
        加载数据文件，获得全部样本的输入向量
        '''
        content = self.get_file_content()#二进制字符串
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(self.get_picture(content, index)))
            #当你打印content时，Python会把每个字节转换成十六进制形式（\x开头），以便于显示
            #但当你访问content的某个元素时，Python会返回该元素对应的十进制整数1。
        return data_set


# 标签数据加载器
class LabelLoader(Loader):
    def load(self):
        '''
        加载数据文件，获得全部样本的标签向量
        '''
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels

    def norm(self, label):
        '''
        内部函数，将一个值转换为10维标签向量
        '''
        label_value = self.to_int(label)
        label_vec= list(map(lambda a: 0.9 if label_value==a else 0.1,range(10)))
        return label_vec


def get_training_data_set():
    '''
    获得训练数据集
    '''
    image_loader = ImageLoader('dataset\\train-images.idx3-ubyte', 6000)
    label_loader = LabelLoader('dataset\\train-labels.idx1-ubyte', 6000)
    return image_loader.load(), label_loader.load()


def get_test_data_set():
    '''
    获得测试数据集
    '''
    image_loader = ImageLoader('dataset\\t10k-images.idx3-ubyte', 1000)
    label_loader = LabelLoader('dataset\\t10k-labels.idx1-ubyte', 1000)
    return image_loader.load(), label_loader.load()


def show(sample):
    str = ''
    for i in range(28):
        for j in range(28):
            if sample[i*28+j] != 0:
                str += '*'
            else:
                str += ' '
        str += '\n'
    print (str)


def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index


def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)

    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)


def now():
    return datetime.now().strftime('%c')


def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = transpose(get_training_data_set())
    test_data_set, test_labels = transpose(get_test_data_set())
    
    
    network = Network([784, 300, 10])
    best_error_ratio=1
    while epoch!=1000:

        epoch += 1
        #test=np.array(list(map(lambda x:x*x,np.arange(0,12))))
        #map的可迭代对象可以是array，但map的输出不能直接转换为np.array,必须先转换为list再转np.array
        network.train(train_labels, train_data_set, 0.001, 10)
        print ('%s epoch %d finished, loss %f' % (now(), epoch, network.loss(train_labels[-1], network.predict(train_data_set[-1]))))
        if epoch % 2 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            
            print ('%s after epoch %d, error ratio is %f,best ratio is %f' % (now(), epoch, error_ratio,best_error_ratio))
            if error_ratio < best_error_ratio:
                best_error_ratio=error_ratio
            else:
                pass

if __name__ == '__main__':
    train_and_evaluate()


