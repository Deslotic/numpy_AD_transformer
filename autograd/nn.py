import numpy as np
from autograd.tensor import Tensor


class Module:
    def parameters(self):
        params = []
        for name, attr in self.__dict__.items():
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, list):  # 用于 ModuleList
                for item in attr:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        # 去重
        return list(dict.fromkeys(params))

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs):
        pass

    def train(self, is_train=True):
        if hasattr(self, 'is_train'):
            self.is_train = is_train
        for name, attr in self.__dict__.items():
            if isinstance(attr, Module):
                attr.train(is_train)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.W = Tensor.parameter(in_features, out_features)
        self.b = Tensor.parameter(out_features) if bias else None

    def forward(self, x):
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Tensor.parameter(num_embeddings, embedding_dim)

    def forward(self, indices):
        # indices 是一个 numpy int 数组
        return self.weight[indices]

class Embedding2(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.weight = Tensor.parameter(num_embeddings, embedding_dim)

    def forward(self, indices):
        return self._one_hot(indices) @ self.weight

    def _one_hot(self, indices):
        ret = np.zeros((len(indices), self.num_embeddings))
        for i,j in enumerate(indices):
            ret[i,j] = 1
        return ret

class LayerNorm(Module):
    def __init__(self, normalized_shape_len,):
        self.gamma = Tensor.ones(normalized_shape_len)
        self.beta = Tensor.zeros(normalized_shape_len)
        self.gamma.requires_grad = True
        self.beta.requires_grad = True

    def forward(self, x):
        # x: (B, L, D) -> 归一化最后一个维度 (D)
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)

        x_normalized = (x - mean) / (var.sqrt() + 1e-9)

        # 广播 gamma 和 beta
        return (self.gamma * x_normalized) + self.beta


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Dropout(Module):
    def __init__(self, p=0.1):
        self.p = p
        self.is_training = True  # 模拟 train/eval 模式

    def forward(self, x):
        if not self.is_training or self.p == 0:
            return x

        # 创建一个 numpy mask
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        # 缩放，尺度补偿
        out = (x * mask) * (1.0 / (1.0 - self.p))

        def _backward():
            if x.requires_grad:
                x.grad += out.grad * mask * (1.0 / (1.0 - self.p))

        out._prev = {x}
        out._op = 'dropout'
        out._backward = _backward
        out.requires_grad = x.requires_grad

        return out