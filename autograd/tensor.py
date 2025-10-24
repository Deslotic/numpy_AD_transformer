import numpy as np


# 设置一个固定的随机种子，以便复现
# np.random.seed(42)


class Tensor:
    """支持自动微分的手写Tensor类，与torch.Tensor相似"""

    # --------------------------------- 基础方法 --------------------------------
    def __init__(self, data, requires_grad=False, _parents=(), _op=''):
        """

        :param data: ndarray或者可以转换为ndarray的数据类型，是Tensor实际的数据存放位置
        :param requires_grad: 是否需要梯度
        :param _parents: 该张量的来源，用于存储计算图以进行反向传播
        :param _op: 标识符，标志着该张量经由什么操作得来，不参与实际运算
        """

        # 确保data被转化为float32的ndarray
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data, dtype=np.float32)
            except ValueError:
                raise TypeError(f"Tensor 仅支持数值型数据, 收到: {type(data)}")

        if data.dtype != np.float32:
            data = data.astype(np.float32)

        self.data = data
        self.shape = self.data.shape
        self.requires_grad = requires_grad

        # 自动微分核心
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_parents)  # 去重
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape}, requires_grad={self.requires_grad}, op='{self._op}')"

    def backward(self, gradient=None):
        """执行反向传播"""
        if not self.requires_grad:
            return  # 不需要梯度则直接返回

        if gradient is None:
            if self.data.size != 1:
                raise ValueError("对于非标量张量，需要指定gradient参数")
            gradient = np.array(1.0, dtype=np.float32)

        if not isinstance(gradient, np.ndarray):
            gradient = np.array(gradient, dtype=np.float32)

        if gradient.shape != self.shape:
            raise ValueError(f"张量与梯度形状不匹配")

        # 基于DFS的拓扑排序
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 初始化所有梯度
        # 保证梯度累加的过程中不出错
        for v in topo:
            if v.requires_grad:
                if v.grad is None:
                    v.grad = np.zeros_like(v.data)

        # 反向传播
        self.grad = gradient

        for node in reversed(topo):
            node._backward()  # 这里的_backward()方法基于不同的运算有不同的求梯度方法

    def zero_grad(self):
        """清除梯度"""
        self.grad = np.zeros_like(self.data) if self.requires_grad else None

    @staticmethod
    def parameter(*shape, init_method='xavier'):
        """
        创建参数对象。事实上，在pytorch中，参数对象是被包装为Parameter实例的;
        在这里，我们只返回一个需要梯度的、经过初始化的Tensor实例，其本质是相同的。
        支持xavier初始化（transformer原始论文选择）、kaiming初始化（更适合relu激活函数）
        """

        # 支持1维的bias初始化以及2维的weight初始化
        fan_in = shape[0]
        fan_out = shape[-1] if len(shape) > 1 else 0

        if init_method == 'xavier':
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            data = np.random.uniform(-limit, limit, size=shape).astype(np.float32)
            return Tensor(data, requires_grad=True)
        elif init_method == 'kaiming':
            std = np.sqrt(2.0 / fan_in)
            data = np.random.normal(0.0, std, size=shape).astype(np.float32)
            return Tensor(data, requires_grad=True)
        else:
            raise(NotImplementedError('仅支持xavier初始化以及kaiming初始化！'))

    @staticmethod
    def zeros(*shape, requires_grad=False):
        return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, requires_grad=False):
        return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)

    def detach(self):
        """分离计算图"""
        return Tensor(self.data, requires_grad=False)

    # --------------------------------- 反向传播相关的方法 --------------------------------
    @staticmethod
    def _unbroadcast(target_shape, grad):
        """
        反向传播时处理广播的辅助函数。
        :param target_shape: 目标形状
        :param grad: 梯度
        :return: 缩减后的梯度
        """
        # 撤销不存在的维度的广播 eg (3) -> (3,3)
        while len(grad.shape) > len(target_shape):
            grad = grad.sum(axis=0)

        # 撤销维度为1的广播 eg (1,3) -> (3,3)
        for i, (dim_target, dim_grad) in enumerate(zip(target_shape, grad.shape)):
            if dim_target == 1 and dim_grad > 1:
                grad = grad.sum(axis=i, keepdims=True)

        # 标量情况，如果target_shape是() (标量)，而grad是(1,)
        if grad.shape != target_shape:
            grad = grad.sum()
        return grad

    # --- 基础数学 ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            data=self.data + other.data,
            _parents=(self, other),
            _op='+'
        )
        out.requires_grad = self.requires_grad or other.requires_grad

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(self.shape, out.grad)
            if other.requires_grad:
                other.grad += other._unbroadcast(other.shape, out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            data=self.data * other.data,
            _parents=(self, other),
            _op='*'
        )
        out.requires_grad = self.requires_grad or other.requires_grad

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(self.shape, other.data * out.grad)
            if other.requires_grad:
                other.grad += other._unbroadcast(other.shape, self.data * out.grad)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            data=self.data / other.data,
            _parents=(self, other),
            _op='/'
        )
        out.requires_grad = self.requires_grad or other.requires_grad

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(self.shape, out.grad / other.data)
            if other.requires_grad:
                other.grad += other._unbroadcast(other.shape, - (self.data * out.grad / other.data ** 2))

        out._backward = _backward
        return out

    # def __truediv__(self, other):
    #     return self * (other ** -1)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            data=self.data @ other.data,
            _parents=(self, other),
            _op='@'
        )
        out.requires_grad = self.requires_grad or other.requires_grad

        def _backward():
            if self.requires_grad:
                other_T_axes = list(range(len(other.shape)))
                # 转置后两个维度
                if len(other_T_axes) >= 2:
                    other_T_axes[-1], other_T_axes[-2] = other_T_axes[-2], other_T_axes[-1]

                self.grad += self._unbroadcast(
                    self.shape,
                    out.grad @ np.transpose(other.data, other_T_axes)
                )

            if other.requires_grad:
                self_T_axes = list(range(len(self.shape)))
                if len(self_T_axes) >= 2:
                    self_T_axes[-1], self_T_axes[-2] = self_T_axes[-2], self_T_axes[-1]

                other.grad += other._unbroadcast(
                    other.shape,
                    np.transpose(self.data, self_T_axes) @ out.grad
                )

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "只支持标量幂"
        out = Tensor(
            data=self.data ** other,
            _parents=(self,),
            _op=f'**{other}'
        )
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad: self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (other * -1.0)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Tensor(other) - self

    # --- 激活与归约 ---
    def relu(self):
        out = Tensor(
            data=np.maximum(0, self.data),
            _parents=(self,),
            _op='ReLU'
        )
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad: self.grad += (self.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(
            data=np.exp(self.data),
            _parents=(self,),
            _op='exp'
        )
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad: self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(
            data=np.log(self.data),
            _parents=(self,),
            _op='log'
        )
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad: self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sqrt(self):
        out = Tensor(
            data=np.sqrt(self.data),
            _parents=(self,),
            _op='sqrt'
        )
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad: self.grad += 1 / (2 * out.data + 1e-9) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            data=np.sum(self.data, axis=axis, keepdims=keepdims),
            _parents=(self,),
            _op='sum'
        )
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad:
                grad_shape = self.shape
                if axis is not None and not keepdims:
                    grad = np.expand_dims(out.grad, axis)
                else:
                    grad = out.grad
                self.grad += np.broadcast_to(grad, grad_shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        N = np.prod(self.data.shape) if axis is None else self.data.shape[axis]
        out = self.sum(axis=axis, keepdims=keepdims)  / N
        out._op = 'mean'
        return out

    def max(self, axis=None, keepdims=False):
        out = Tensor(
            data=np.max(self.data, axis=axis, keepdims=keepdims),
            _parents=(self,),
            _op='max'
        )
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad:
                max_val = out.data
                if axis is not None and not keepdims:
                    max_val = np.expand_dims(max_val, axis)

                mask = (self.data == max_val).astype(np.float32)
                mask_sum = mask.sum(axis=axis, keepdims=True)  # 求mask在对应轴上的和，即每个轴上有多少个最大值
                mask = mask / np.where(mask_sum == 0, 1.0, mask_sum)  # 平分梯度，使用where避免除以0

                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis)

                self.grad += mask * grad

        out._backward = _backward
        return out

    # --- shaping ---

    def reshape(self, *shape):
        out = Tensor(
            data=self.data.reshape(*shape),
            _parents=(self,),
            _op='reshape'
        )
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad: self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        out = Tensor(
            data=np.transpose(self.data, axes),
            _parents=(self,),
            _op='transpose'
        )
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad:
                inv_axes = np.argsort(axes)
                self.grad += np.transpose(out.grad, inv_axes)

        out._backward = _backward
        return out

    def __getitem__(self, item):
        out = Tensor(
            data=self.data[item],
            _parents=(self,),
            _op='getitem'
        )
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad:
                np.add.at(self.grad, item, out.grad)

        out._backward = _backward
        return out

    @staticmethod
    def concatenate(tensors, axis=0):
        data = np.concatenate([t.data for t in tensors], axis=axis)
        out = Tensor(
            data=data,
            _parents=set(tensors),
            _op='concat'
        )
        out.requires_grad = any(t.requires_grad for t in tensors)

        def _backward():
            indices = np.cumsum([t.shape[axis] for t in tensors])
            slices = np.split(out.grad, indices[:-1], axis=axis)
            for t, s_grad in zip(tensors, slices):
                if t.requires_grad:
                    t.grad += s_grad

        out._backward = _backward
        return out

    @staticmethod
    def cat(tensors, axis=0):
        return Tensor.concatenate(tensors, axis)

    # --- Transformer ---

    def masked_fill(self, mask, value):
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask 必须是 numpy 数组")

        value = float(value)
        out = Tensor(
            data=np.where(mask, value, self.data),
            _parents=(self,),
            _op='masked_fill'
        )
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (1.0 - mask.astype(np.float32))

        out._backward = _backward
        return out

    def logsumexp(self, axis=-1, keepdims=False):
        max_val = self.max(axis=axis, keepdims=True).detach()
        x_shifted = self - max_val
        exp_shifted = x_shifted.exp()
        sum_exp_shifted = exp_shifted.sum(axis=axis, keepdims=keepdims)
        log_sum_exp = sum_exp_shifted.log() + max_val.reshape(sum_exp_shifted.shape)
        log_sum_exp._op = 'logsumexp'
        return log_sum_exp

    def logsoftmax(self, axis=-1):
        lse = self.logsumexp(axis=axis, keepdims=True)
        out = self - lse
        out._op = 'logsoftmax'
        return out

    def softmax(self, axis=-1):
        return self.logsoftmax(axis=axis).exp()

    def take_along_axis(self, indices, axis=-1):
        if not isinstance(indices, np.ndarray):
            raise TypeError("indices 必须是 numpy 数组")

        out = Tensor(
            data=np.take_along_axis(self.data, indices, axis=axis),
            _parents=(self,),
            _op='take_along_axis'
        )
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad:
                grad_in = np.zeros_like(self.data)
                np.put_along_axis(grad_in, indices, out.grad, axis=axis)
                self.grad += grad_in

        out._backward = _backward
        return out

    def squeeze(self):
        target_shape = []
        for s in self.shape:
            if s != 1:
                target_shape.append(s)
        return self.reshape(*target_shape)

    def unsqueeze(self, axis):
        target_shape = list(self.shape)[:axis] + [1] + list(self.shape)[axis:]
        return self.reshape(*target_shape)


if __name__ == '__main__':
    # test_tensor = Tensor([[1,2,3],[4,5,6]])
    # test_tensor = test_tensor.unsqueeze(0)
    # print(test_tensor.data)
    # print((Tensor(0)/Tensor(0)))
    print(
        Tensor([0,0,0])==0
    )
