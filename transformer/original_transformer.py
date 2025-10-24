from autograd.nn import Module
from .blocks import Encoder, Decoder, Generator


class Transformer(Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers,
                 num_heads, d_ff, max_seq_len=100, dropout_p=0.1, pad_id=-1):
        self.encoder = Encoder(num_layers, src_vocab_size, d_model, num_heads,
                               d_ff, dropout_p, max_seq_len, pad_id)
        self.decoder = Decoder(num_layers, tgt_vocab_size, d_model, num_heads,
                               d_ff, dropout_p, max_seq_len, pad_id)
        self.generator = Generator(tgt_vocab_size, d_model)

    def forward(self, src_indices, tgt_indices, src_mask=None, tgt_mask=None):
        encoder_out, src_mask = self.encoder.forward(src_indices, src_mask)
        tgt = self.decoder.forward(tgt_indices, encoder_out, tgt_mask, src_mask)
        return self.generator(tgt)
