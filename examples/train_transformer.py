import numpy as np
from autograd.loss import CrossEntropyLoss
from autograd.optim import SGD
from transformer.original_transformer import Transformer

# 设置超参数
SRC_VOCAB = 10
TGT_VOCAB = 12
D_MODEL = 8
D_FF = 16
NUM_HEADS = 2
NUM_LAYERS = 2
MAX_LEN = 10
BATCH_SIZE = 1
SRC_LEN = 5
TGT_LEN = 6
NUM_EPOCHS = 1000
LEARNING_RATE = 0.01
PRINT_EVERY = 10  # 每 10 个 epoch 打印一次日志
PAD_ID = -1

# 创建模型
model = Transformer(SRC_VOCAB, TGT_VOCAB, D_MODEL, NUM_LAYERS,
                    NUM_HEADS, D_FF, MAX_LEN, 0.1, PAD_ID)

# 创建损失函数和优化器
criterion = CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

# 虚拟样本
src_indices = np.random.randint(1, SRC_VOCAB, size=(BATCH_SIZE, SRC_LEN))
tgt_indices = np.random.randint(1, TGT_VOCAB, size=(BATCH_SIZE, TGT_LEN))  # 输入decoder的数据

targets = np.roll(tgt_indices, -1, axis=-1)  # 做损失的数据，与tgt_indices有一个相位差
targets[:, -1] = PAD_ID  # 最后一个设为 <pad>

# 开始训练
model.train()
total_loss = 0.0
for epoch in range(NUM_EPOCHS):
    logits = model(src_indices, tgt_indices)
    loss = criterion(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.data  # 累加损失

    if (epoch + 1) % PRINT_EVERY == 0:
        avg_loss = total_loss / PRINT_EVERY
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], 平均 Loss: {avg_loss:.4f}")
        total_loss = 0.0