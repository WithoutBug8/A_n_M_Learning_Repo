import numpy as np

# 随机梯度下降 SGD
class SGD:
    # 初始化
    def __init__(self, lr=0.01):
        # lr是learning rate学习率
        self.lr = lr
    # 参数更新,传入参数字典和梯度字典
    def update(self, params, grads):
        # 遍历所有传入的参数，按照公式更新
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# Momentum动量法
class Momentum:
    # 初始化
    def __init__(self, lr=0.01,momentum=0.9):
            self.lr = lr
            self.momentum = momentum
            # 这里应该有一个参数v,历史负梯度的加权和
            self.v = None
    # 参数更新
    def update(self, params, grads):
        # 对v进行初始化
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        # 按照公式进行参数更新
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]