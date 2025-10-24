# NumpyAD-Transformer: 从零手写自动微分框架与Transformer模型

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目简介

本项目旨在展现对现代深度学习框架底层原理及Transformer架构的深刻理解。核心内容是**完全基于NumPy从零构建的极简自动微分（AD）引擎**，该引擎支持动态计算图构建与反向传播。基于此AD引擎，项目进一步完整实现了**Transformer模型（Encoder-Decoder结构）**，严格遵循了 "Attention Is All You Need" 论文中的原始设计。

本项目的首要目标是学习和验证，基于数学原理剖析绝大部分深度学习开发者眼中的黑箱操作（如.backward()、优化器等）。
本项目不追求极致的运行效率：基于numpy、python，注定本框架无法投入实际的生产过程中。但是在原理上，本项目力求通过最优的算法（维度操作、gathering操作等）保证一定程度上的性能。此外，本项目还注重基于计算机硬件结构的数值问题规避，力求实现接近pytorch的效果。

## ✨ 核心特性与亮点

本项目亮点如下：

* **从零构建AD引擎:**
    * 实现了支持自动微分的核心 `Tensor` 类。
    * 在前向传播过程中隐式构建**动态计算图**。
    * 基于计算图的**拓扑排序**实现**反向传播**算法 。
* **推导并实现核心算子梯度:**
    * 手动推导并编码实现了关键运算的反向传播逻辑，包括：
        * 矩阵乘法 (`__matmul__`)。
        * 广播机制处理 (通过 `_unbroadcast` 辅助函数) 。
        * 逐元素运算 (`__add__`, `__mul__`, `__pow__` 等) 。
        * 激活函数 (ReLU, Softmax/LogSoftmax，并考虑了数值稳定性) 。
        * 现代激活函数 (SiLU, GELU, Mish) 。
        * 索引与切片 (`__getitem__`，使用 `np.add.at` 确保梯度累加的安全性) 。
* **复现标准神经网络模块:**
    * 基于自定义的 `Tensor` 类，构建了标准的神经网络层 (`nn.Module`, `nn.Linear`, `nn.LayerNorm`, `nn.Embedding`, `nn.ReLU`, `nn.Dropout`)。
* **完整实现Transformer架构:**
    * 构建了Transformer的所有关键组件：
        * 缩放点积注意力 (Scaled Dot-Product Attention, 含masking和 `sqrt(d_k)` 缩放)。
        * 多头注意力机制 (Multi-Head Attention)。
        * 前馈神经网络网络 (Feed-Forward Networks) 。
        * 位置编码 (Positional Encoding, 使用正弦/余弦函数，基于原始论文复现) 。
        * 编码器和解码器堆栈 (Encoder & Decoder Stacks) 。
    * 在模拟任务上成功训练了模型，验证了整个框架的正确性 。
* **现代transformer架构的尝试(进行中):**
    * 稠密混合专家模型(DenseMOE, 从广播机制和元素级乘法实现einsum的本质)

* **🚀 版本迭代与未来工作 (进行中):**
    * **v1.0:** 自动微分与基础Transformer的实现。
    * **v1.1 (当前):** 现代激活函数以及现代Transformer的部分实现、修正优化了已有代码。

## 📚 项目结构

```
NumpyAD-Transformer/
├── autograd/           # 核心自动微分引擎
│   ├── tensor.py       # -> 核心：包含反向传播逻辑的 Tensor 类
│   ├── nn.py           # -> 神经网络模块 (Linear, LayerNorm 等)
│   ├── loss.py         # -> 损失函数 (CrossEntropy)
│   └── optim.py        # -> 优化器 (SGD)
│
├── transformer/        # 基于 'autograd' 实现的 Transformer 模型
│   ├── blocks.py       # -> Attention, FFN, Positional Encoding 等组件
│   └── origin_transformer.py # -> 组装好的 Encoder-Decoder Transformer
│
├── docs/               # 详细文档存放处
│   ├── v1.md           # -> 深入的技术细节与代码解析文档
│   └── v1.1.md         # -> 新版本增改的代码解析文档
│
├── examples/           # 示例代码
│   └── train_transformer.py # -> 在模拟数据上运行的训练脚本
│
├── .gitignore          # Git 忽略文件配置
├── LICENSE             # 项目许可证
└── README.md           # 项目入口与摘要
```

## 🛠️ 技术深度解析

关于自动微分的原理、梯度推导细节、各模块的具体实现、Transformer组件背后的数学思想等**详细内容**，请参阅完整的技术文档：

**➡️ [详细技术文档](./docs/v1.md)**

## ▶️ 使用示例

运行示例训练脚本，在模拟数据上训练Transformer模型：

```bash
python examples/train_transformer.py
```

这将执行训练循环，并打印每个epoch的损失，用以展示AD引擎和Transformer实现的有效性 。

## 📜 许可证

本项目采用 MIT 许可证。详情请见 [LICENSE](./LICENSE) 文件。
