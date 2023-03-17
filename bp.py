#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import random
from numpy import *
from functools import reduce


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    def calc_output(self):
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta
        #downstream_delta:本层经过sigmoid的输出张量Ai的梯度，由所有下游直接节点的输出张量Ai+1的梯度计算而来

    def calc_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)
        #输出层的输出张量Ai的梯度
        #这里的delta实际上已经是梯度的负数了，所以update的时候是+=而不是-=

    def __str__(self):#print(node)
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str 
    #upstream是一个上游连接列表，每一条连接都是一个connection类，有——str——方法


class ConstNode(object):#偏置节点b没有上游节点
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1#初始直接输出1

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


class Layer(object):
    #每一层都会有一个constnode
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    #为第一层layer设置输入（input）
    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    #计算每一个node的output，并存在node自己的self.output里
    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print(node)


class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):#根据每个下游节点的导数和上游节点的输出计算权重的梯度
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    def get_gradient(self):
        return self.gradient

    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index, 
            self.upstream_node.node_index,
            self.downstream_node.layer_index, 
            self.downstream_node.node_index, 
            self.weight)


class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print(conn)


class Network(object):
    def __init__(self, layers):#layers:[8,3,8]
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        #节点创建
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        #连接创建
        for layer in range(layer_count - 1):
            #前一层只和后一层（n个节点）的前n-1节点连接，因为最后一个节点是bias节点，没有输入，输出是1
            #连接创建
            connections = [Connection(upstream_node, downstream_node) 
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                #给节点加上相关连接，把两者建立关联
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)


    def train(self, labels, data_set, rate, epoch):
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)
                # print 'sample %d training finished' % d

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):#这个是算每一个节点的输出中间激活的梯度
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
            
        #稀疏更新可以在这里就中止反向传播
        for layer in self.layers[-2::-1]:#-2就是倒数第2层，-1是反向step
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        #遍历所有连接
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        self.predict(sample)
        self.calc_delta(label)#计算每一个节点的（输出）导数
        self.calc_gradient()#计算每一个连接的梯度

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes[:-1]))
        #node[:-1] 不要最后一层最后一个constnode的输出

    def dump(self):
        for layer in self.layers:
            layer.dump()


class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]#从00000001到10000000

    def norm(self, number):
        return list(map(lambda m: 0.9 if number & m else 0.1, self.mask))

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x,y: x + y, binary)


def mean_square_error(vec1, vec2):
    return 0.5 * reduce(lambda a, b: a + b, 
                        map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                            zip(vec1, vec2)
                        )
                 )


def gradient_check(network, sample_feature, sample_label):
    '''
    梯度检查
    network: 神经网络对象
    sample_feature: 样本的特征
    sample_label: 样本的标签
    '''
    # 计算网络误差
    network_error = lambda vec1, vec2: 0.5 * reduce(lambda a, b: a + b, 
                      map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),zip(vec1, vec2)))

    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)

    # 对每个权重做梯度检查    
    for conn in network.connections.connections: 
        # 获取指定连接的梯度
        # #这样求出来的梯度是这个点的斜率（y是loss，x是w）
        actual_gradient = conn.get_gradient()
    
        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)
    
        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)
    
        # 根据式6计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)
    
        # 打印
        print ('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient))


def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for _ in range(0, 256, 8):#32个样本
        n = normalizer.norm(int(random.uniform(0, 256)))
        data_set.append(n)
        labels.append(n)
    return labels, data_set


def train(network):
    labels, data_set = train_data_set()
    network.train(labels, data_set, 0.05, 50)


def test(network, data):
    normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    predict_data = network.predict(norm_data)
    print ('\ttestdata(%u)\tpredict(%u)' % (
        data, normalizer.denorm(predict_data)))


def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print ('correct_ratio: %.2f%%' % (correct / 256 * 100))


def gradient_check_test():
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)


if __name__ == '__main__':
    net = Network([8, 3, 8])
    train(net)
    net.dump()
    correct_ratio(net)
    
    gradient_check_test()
    
    '''numbers=[int(random.normal(128,42)) for _ in range(10000)]
    freq=list(map(lambda a:numbers.count(a) ,range(0,256)))
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(list(range(0,256)),freq)    
    plt.show()
'''