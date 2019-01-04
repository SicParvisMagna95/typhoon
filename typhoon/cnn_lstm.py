"""system provided"""
import os

"""third party"""
import numpy as np
import matplotlib.pyplot as plt
from typhoon.train import *
'''超参数在 model.py 里定义'''

if __name__ == '__main__':
    '''实例化网络模型'''
    net = CNN_LSTM()
    print(net)

    '''是否使用GPU'''
    if USE_GPU:
        net = net.cuda()

    '''定义损失函数和优化方法'''
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    '''训练'''
    train(net=net,loss_func=loss_func,optimizer=optimizer)
    print()
