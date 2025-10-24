import numpy as np
from autograd import nn
from autograd.tensor import Tensor


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout_p=0.1):
        self.d_k = Tensor(d_k)
        self.softmax = lambda x: x.softmax(axis=-1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, Q, K, V, mask=None):
        scores = (Q @ K.transpose(0, 1, 3, 2)) / self.d_k.sqrt()
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn_weights = self.dropout(self.softmax(scores))
        return attn_weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_p=0.1):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(self.d_k, dropout_p)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        attn_out = self.attn(Q, K, V, np.expand_dims(mask, 1))  # mask在head维度升维（用于广播）以适配多头注意力
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        output = self.out_proj(attn_out)
        res = self.dropout(output) + query
        return self.ln(res)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.dropout(self.fc2(self.relu(self.fc1(x))))
        return self.ln(out + x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_p=0.1):
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_p)
        self.ffn = FeedForward(d_model, d_ff, dropout_p)

    def forward(self, x, src_mask=None):
        return self.ffn(self.mha(x, x, x, src_mask))


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        pe = np.zeros((max_len, d_model))
        pos = np.expand_dims(np.arange(0, max_len), -1)
        div_term = np.pow(np.array(10000), np.arange(0, d_model, 2) / d_model)
        inner = pos / div_term
        pe[:, ::2] = np.sin(inner)
        pe[:, 1::2] = np.cos(inner)
        self.pe = Tensor(pe).unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


# 基于transformer原始论文实现的Embedding层，进行了缩放，缩放词嵌入向量的尺度以适配位置编码向量
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = Tensor(d_model)
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self, x):
        return self.embedding(x) * self.d_model.sqrt()


class Encoder(nn.Module):
    def __init__(self, num_layers, vocab_size, d_model, num_heads, d_ff, dropout_p, max_seq_len=100, pad_id=-1):
        self.pad_id = pad_id
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_enc = PositionEncoding(d_model, max_seq_len)
        self.layers = [EncoderLayer(d_model, num_heads, d_ff, dropout_p) for _ in range(num_layers)]

    def forward(self, x, mask=None):
        if mask is None:
            mask = self._get_pad_mask(x)
        x = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            x = layer(x, mask)
        return x, mask

    def _get_pad_mask(self, x):
        # x: B,S
        # mask: B,1,S
        data = x.data if isinstance(x, Tensor) else np.asarray(x)
        return np.expand_dims((data==self.pad_id), 1)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_p=0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_p)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout_p)
        self.ffn = FeedForward(d_model, d_ff, dropout_p)

    def forward(self, tgt, encoder_out, look_ahead_mask=None, src_mask=None):
        tgt = self.self_attn(tgt, tgt, tgt, look_ahead_mask)
        tgt = self.cross_attn(tgt, encoder_out, encoder_out, src_mask)
        return self.ffn(tgt)


class Decoder(nn.Module):
    def __init__(self, num_layers, vocab_size, d_model, num_heads, d_ff, dropout_p, max_seq_len=100, pad_id=-1):
        self.pad_id = pad_id
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_enc = PositionEncoding(d_model, max_seq_len)
        self.layers = [DecoderLayer(d_model, num_heads, d_ff, dropout_p) for _ in range(num_layers)]

    def forward(self, tgt, encoder_out, look_ahead_mask=None, src_mask=None):
        if look_ahead_mask is None:
            look_ahead_mask = self._get_mask(tgt)
        tgt = self.pos_enc(self.embedding(tgt))
        for layer in self.layers:
            tgt = layer(tgt, encoder_out, look_ahead_mask, src_mask)
        return tgt

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


class Generator(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)


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

        # 加权求和
        # probs: b,s,n experts_out: b,s,n,o -> b,s,o
        probs_uns = probs.unsqueeze(-1)
        out_uns = experts_out * probs_uns
        return out_uns.sum(-2)  # b,s,o




if __name__ == '__main__':
    # x = np.random.randn(5, 5)
    # mask = np.ones_like(x.data)
    # mask = 1 - np.tril(mask, 0)
    # pass
    test_tensor = Tensor([[[1,2,3,4]]])  # 1,1,4
    moe = DenseMOE(4,4,2)
    print(moe(test_tensor))