from autograd.nn import Module
from transformer.modern_blocks import Encoder, Decoder, GeneratorWithWeightTying


class Transformer(Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers,
                 num_heads, num_kv_heads, num_experts, topk, dropout_p=0.1, max_seq_len=100, pad_id=-1):
        self.encoder = Encoder(num_layers, src_vocab_size, d_model, num_heads,
                               num_kv_heads, num_experts, topk, dropout_p, max_seq_len, pad_id)
        self.decoder = Decoder(num_layers, tgt_vocab_size, d_model, num_heads,
                               num_kv_heads, num_experts, topk, dropout_p, max_seq_len, pad_id)
        self.generator = GeneratorWithWeightTying(self.decoder.embedding.weight)

    def forward(self, src_indices, tgt_indices, src_mask=None, tgt_mask=None):
        aux_loss = 0.0
        encoder_out, src_mask, tmp_aux_loss = self.encoder.forward(src_indices, src_mask)
        aux_loss += tmp_aux_loss
        tgt, tmp_aux_loss = self.decoder.forward(tgt_indices, encoder_out, tgt_mask, src_mask)
        aux_loss += tmp_aux_loss
        return self.generator(tgt), aux_loss


if __name__ == '__main__':
    transformer = Transformer(
        10,10,8,2,4,2,8,2
    )
    src = [[1,2,3,4]]
    tgt = [[1,3,4]]
    print(transformer(src, tgt))
