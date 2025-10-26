import numpy as np
from autograd import nn
from autograd.tensor import Tensor
from transformer.blocks import ScaledDotProductAttention


class Gate(nn.Module):
    def __init__(self, feature_in, num_experts):
        self.linear = nn.Linear(feature_in, num_experts)

    def forward(self, x):
        return self.linear(x).softmax(-1)


class DenseMOE(nn.Module):
    def __init__(self, feature_in, feature_out, num_experts):
        self.w = Tensor.parameter(num_experts, feature_in, feature_out)
        self.b = Tensor.parameter(num_experts, feature_out)
        self.gate = Gate(feature_in, num_experts)

    def forward(self, x):
        # x: b, s, i (feature_in)

        # 传入gate得到概率
        probs = self.gate(x)  # b, s, n (num_experts)

        # x并行传入experts
        # x: b, s, i, w: n, i, o (feature_out) -> b, s, n, o
        # 基于einsum,i维度在x和w都出现但在结果不出现。因此是沿i维度求和。
        # broadcast: x-> b,s,1,i,1  w-> 1,1,n,i,o 然后进行mul得到 b,s,n,i,o 的张量，再沿d维度求和降维
        x_uns = x.unsqueeze(2).unsqueeze(-1)  # b,s,1,i,1
        w_uns = self.w.unsqueeze(0).unsqueeze(0)  # 1,1,n,i,o
        experts_out_uns = x_uns * w_uns  # b,s,n,i,o
        experts_out = experts_out_uns.sum(-2)  # b,s,n,o
        experts_out = experts_out + self.b.unsqueeze(0).unsqueeze(0)  # 加偏置
        experts_out = experts_out.relu()

        # 加权求和
        # probs: b,s,n experts_out: b,s,n,o -> b,s,o
        probs_uns = probs.unsqueeze(-1)
        out_uns = experts_out * probs_uns
        return out_uns.sum(-2)  # b,s,o


class SparseMOE(nn.Module):
    def __init__(self, feature_in, feature_out, num_experts, topk, dropout_p=0.1):
        self.w = Tensor.parameter(num_experts, feature_in, feature_out)
        self.b = Tensor.parameter(num_experts, feature_out)
        self.gate = Gate(feature_in, num_experts)
        self.topk = topk
        self.num_experts = num_experts
        self.is_train = True
        self.norm = nn.RMSNorm(feature_in)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x: b, s, i (feature_in)
        shortcut = x
        x = self.norm(x)

        # 传入gate得到概率
        probs = self.gate(x)  # b, s, n (num_experts)

        topk_probs, topk_indices = probs.topk(self.topk)
        topk_probs /= topk_probs.sum(-1, keepdims=True) + 1e-9  # 概率归一化
        w_uns = self.w[topk_indices]  # 选中对应的专家，由于numpy的高级索引机制，形状会变成 b,s,k,i,o
        b_uns = self.b[topk_indices]  # b,s,k,o

        x_uns = x.unsqueeze(2).unsqueeze(-1)  # b,s,1,i,1
        experts_out_uns = x_uns * w_uns  # b,s,k,i,o
        experts_out = experts_out_uns.sum(-2)  # b,s,k,o
        experts_out = experts_out + b_uns  # 加偏置
        experts_out = experts_out.gelu()

        probs_uns = topk_probs.unsqueeze(-1)
        out_uns = experts_out * probs_uns

        if not self.is_train:
            return shortcut + self.dropout(out_uns.sum(-2))

        aux_loss = self._aux_loss(probs, topk_indices)  # 辅助损失
        return shortcut + self.dropout(out_uns.sum(-2)), aux_loss

    def _aux_loss(self, probs, indices, alpha=0.01):
        """计算辅助损失以实现负载均衡"""
        # L = alpha * N * sum(f_i * p_i)
        b, s, n = probs.shape
        assert n == self.num_experts
        P = probs.mean(0).mean(0)  # (n,) 对应每一个expert被选中的平均概率。此处调用两次mean是因为Tensor的mean方法暂时还不支持元组索引

        # 转换为one_hot编码，shape: b,s,k,n
        one_hot_indices = np.eye(n, dtype=np.float32)[indices]
        # 沿k维度求和，可以得到每一个token选中的专家， shape: b,s,n
        expert_chosen_mask = one_hot_indices.sum(axis=2)
        # 计算被选中的比例 (对 b 和 s 维度求平均)
        f = expert_chosen_mask.mean((0, 1))  # (n,)。 f一系列的计算都是基于ndarray的，因为对indices的操作不涉及梯度回传
        return alpha * n * (P * f).sum()


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads, dropout_p=0.1, max_len=5000):
        assert d_model % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.d_k = d_model // num_q_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.d_k * num_kv_heads)
        self.v_proj = nn.Linear(d_model, self.d_k * num_kv_heads)

        self.attn = ScaledDotProductAttention(self.d_k, dropout_p)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.RMSNorm(d_model)
        self.rope = RotatePositionalEncoding(self.d_k, max_len)  # rope对象

    def forward(self, query, key, value, mask=None):
        shortcut = query
        query, key, value = self.norm(query), self.norm(key), self.norm(value)  # pre-norm
        batch_size = query.shape[0]

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.reshape(batch_size, -1, self.num_q_heads, self.d_k).transpose(0, 2, 1, 3)  # b,n,s,d_k
        K = K.reshape(batch_size, -1, self.num_kv_heads, self.d_k).transpose(0, 2, 1, 3)  # b,g,s,d_k
        V = V.reshape(batch_size, -1, self.num_kv_heads, self.d_k).transpose(0, 2, 1, 3)  # b,g,s,d_k

        Q, K = self.rope(Q), self.rope(K)

        attn_out = self.attn(Q, K.repeat(self.num_groups, 1), V.repeat(self.num_groups, 1),
                             np.expand_dims(mask, 1) if mask is not None else None)  # mask在head维度升维（用于广播）以适配多头注意力
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        output = self.out_proj(attn_out)
        return shortcut + self.dropout(output)


class RotatePositionalEncoding:
    def __init__(self, dim, max_len=5000, base=10000):
        """预计算编码查找表"""
        assert dim % 2 == 0
        self.dim = dim

        # 计算频率 theta_i = base^(-2i / dim)
        inv_freq = base ** (-(np.arange(0, dim, 2, dtype=np.float32) / dim))

        # 计算位置
        pos = np.expand_dims(np.arange(max_len, dtype=np.float32), -1)  # 广播最后一个维度用于产生高阶矩阵

        # 计算角度 pos * theta_i
        freqs = pos * inv_freq

        # 计算 cos 和 sin
        self.cos_table = np.cos(freqs)
        self.sin_table = np.sin(freqs)

    def __call__(self, x):
        assert x.shape[-1] == self.dim
        cos_emb = np.expand_dims(self.cos_table[:x.shape[2]], (0, 1))
        sin_emb = np.expand_dims(self.sin_table[:x.shape[2]], (0, 1))
        # 分割x
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]

        # 旋转
        rotated_x1 = x1 * cos_emb - x2 * sin_emb
        rotated_x2 = x1 * sin_emb + x2 * cos_emb
        return Tensor.cat([rotated_x1, rotated_x2], -1)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, num_experts, topk, dropout_p=0.1, max_len=5000):
        self.gqa = GroupedQueryAttention(d_model, num_heads,
                                         num_kv_heads, dropout_p, max_len)
        self.moe = SparseMOE(d_model, d_model, num_experts, topk, dropout_p)

    def forward(self, x, src_mask=None):
        return self.moe(self.gqa(x, x, x, src_mask))  # x, aux_loss


# 基于transformer原始论文实现的Embedding层，进行了缩放，缩放词嵌入向量的尺度以适配位置编码向量
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.d_model = Tensor(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * self.d_model.sqrt()


class Encoder(nn.Module):
    def __init__(self, num_layers, vocab_size, d_model, num_heads, num_kv_heads, num_experts, topk, dropout_p, max_seq_len=100, pad_id=-1):
        self.pad_id = pad_id
        self.embedding = Embedding(vocab_size, d_model)
        self.layers = [
            EncoderLayer(d_model, num_heads, num_kv_heads,
                         num_experts, topk, dropout_p, max_seq_len) for _ in range(num_layers)
        ]

    def forward(self, x, mask=None):
        aux_loss = 0
        if mask is None:
            mask = self._get_pad_mask(x)
        x = self.embedding(x)
        for layer in self.layers:
            out = layer(x, mask)
            if isinstance(out, Tensor):
                x = out
            else:  # 带辅助损失
                x = out[0]
                aux_loss += out[1]
        return x, mask, aux_loss

    def _get_pad_mask(self, x):
        # x: B,S
        # mask: B,1,S
        data = x.data if isinstance(x, Tensor) else np.asarray(x)
        return np.expand_dims((data == self.pad_id), 1)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, num_experts, topk, dropout_p=0.1, max_len=5000):
        self.self_attn = GroupedQueryAttention(d_model, num_heads,num_kv_heads, dropout_p, max_len)
        self.cross_attn = GroupedQueryAttention(d_model, num_heads,num_kv_heads, dropout_p, max_len)
        self.moe = SparseMOE(d_model, d_model, num_experts, topk, dropout_p)

    def forward(self, tgt, encoder_out, look_ahead_mask=None, src_mask=None):
        tgt = self.self_attn(tgt, tgt, tgt, look_ahead_mask)
        tgt = self.cross_attn(tgt, encoder_out, encoder_out, src_mask)
        return self.moe(tgt)


class Decoder(nn.Module):
    def __init__(self, num_layers, vocab_size, d_model, num_heads, num_kv_heads, num_experts, topk, dropout_p, max_seq_len=100, pad_id=-1):
        self.pad_id = pad_id
        self.embedding = Embedding(vocab_size, d_model)
        self.layers = [
            DecoderLayer(d_model, num_heads, num_kv_heads,
                         num_experts, topk, dropout_p, max_seq_len) for _ in range(num_layers)
        ]

    def forward(self, tgt, encoder_out, look_ahead_mask=None, src_mask=None, aux_loss=None):
        if aux_loss is None:
            aux_loss = 0.0
        if look_ahead_mask is None:
            look_ahead_mask = self._get_mask(tgt)
        tgt = self.embedding(tgt)
        for layer in self.layers:
            out = layer(tgt, encoder_out, look_ahead_mask, src_mask)
            if isinstance(out, Tensor):
                tgt = out
            else:
                tgt = out[0]
                aux_loss += out[1]
        return tgt, aux_loss

    def _get_look_ahead_mask(self, x):
        seq_len = np.asarray(x).shape[1]
        return ~np.expand_dims(np.tril(np.ones((seq_len, seq_len)), 0), 0).astype(bool)

    def _get_pad_mask(self, x):
        # x: B,S
        # mask: B,1,S
        data = x.data if isinstance(x, Tensor) else np.asarray(x)
        return np.expand_dims((data == self.pad_id), 1)

    def _get_mask(self, x):
        return self._get_pad_mask(x) | self._get_look_ahead_mask(x)


class GeneratorWithWeightTying(nn.Module):
    def __init__(self, embedding_params):
        self.params = embedding_params

    def forward(self, x):
        return x @ self.params.transpose(1, 0)


if __name__ == '__main__':
    test = Tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]])  # 1,1,8
    # rope = RotatePositionalEncoding(4)
    # print(rope(test))
    gqa = GroupedQueryAttention(8, 4, 2)
    print(gqa(test, test, test))
