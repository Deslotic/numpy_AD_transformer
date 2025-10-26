import numpy as np


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):  # 新增 momentum 参数
        self.params = list(params)  # 确保参数列表是可迭代的
        self.lr = lr
        self.momentum = momentum

        # 初始化速度变量，字典存储
        self.velocities = {}
        if self.momentum > 0:
            for p in self.params:
                if p.requires_grad:
                    # 只为需要梯度的参数创建速度
                    self.velocities[p] = np.zeros_like(p.data)

    def step(self):
        for p in self.params:
            # 确保参数需要梯度，并且当前梯度存在
            if p.requires_grad:
                if self.momentum == 0:  # 标准SGD更新
                    p.data -= self.lr * p.grad
                else:
                    current_velocity = self.velocities[p]
                    #  v = momentum * v + lr * grad
                    new_velocity = self.momentum * current_velocity + self.lr * p.grad
                    p.data -= new_velocity
                    self.velocities[p] = new_velocity  # 保存当前速度

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.zero_grad()