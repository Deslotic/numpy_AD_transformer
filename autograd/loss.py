import numpy as np
from .nn import Module
from .tensor import Tensor


class CrossEntropyLoss(Module):
    def __init__(self, ignore_index=-1):
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # 计算 LogSoftmax,输出对数概率
        log_probs = logits.logsoftmax(axis=-1)  # (B, L, V)

        # 将target升维以匹配take_along_axis方法
        target_axis = np.expand_dims(targets, axis=-1)

        # 取nll并降维
        nll = -log_probs.take_along_axis(target_axis, axis=-1).squeeze()

        # 屏蔽pad_id并应用mask
        mask = (targets != self.ignore_index).astype(np.float32)
        loss = (nll * mask).sum()

        # 缩放损失
        num_active = mask.sum()
        # 如果num_active=0，即batch为空批次，会直接得到一个值为nan的Tensor，在反向传播后会将所有梯度都计算为nan，以告知训练出错
        mean_loss = loss / num_active
        mean_loss._op = 'CrossEntropyLoss'

        return mean_loss


if __name__ == '__main__':
    print((Tensor(0) / 0).data)