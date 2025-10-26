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


class RMSprop:
    def __init__(self, params, lr=0.01, beta=0.99, eps=1e-9):
        """
        初始化RMSprop优化器
        :param params:
        :param lr:
        :param beta: 梯度平方的指数移动平均衰减率，即过去梯度对当前的影响程度。β越大过去的影响越大，当前梯度的影响越小
        :param eps: 防止除0的极小值
        """
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.eps = eps

        # 与SGD的实现一样，字典存储
        self.v = {}
        for p in self.params:
            if p.requires_grad:
                self.v[p] = np.zeros_like(p.data)

    def step(self):
        for p in self.params:
            if p.requires_grad:
                g = p.grad
                current_v = self.v[p]
                # v_t = beta * v_{t-1} + (1 - beta) * g_t^2
                new_v = self.beta * current_v + (1 - self.beta) * g ** 2

                # theta_t = theta_{t-1} - lr * g_t / (sqrt(v_t) + eps)
                p.data = p.data - self.lr * g / (np.sqrt(new_v) + self.eps)
                self.v[p] = new_v  # 更新字典

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.zero_grad()


class Adam:
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-9):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # 时间步，用于消除训练初期偏差问题

        self.m, self.v = {}, {}
        for p in self.params:
            if p.requires_grad:
                self.m[p] = np.zeros_like(p.data)
                self.v[p] = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        for p in self.params:
            if p.requires_grad:
                g = p.grad
                current_m, current_v = self.m[p], self.v[p]
                new_m = self.beta1 * current_m + (1 - self.beta1) * g
                new_v = self.beta2 * current_v + (1 - self.beta2) * g ** 2

                # 使用时间步进行偏差修正
                m_modified = new_m / (1 - self.beta1 ** self.t)
                v_modified = new_v / (1 - self.beta2 ** self.t)

                # 梯度下降
                p.data -= self.lr * m_modified / (np.sqrt(v_modified) + self.eps)

                # 更新字典
                self.m[p] = new_m
                self.v[p] = new_v

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.zero_grad()
