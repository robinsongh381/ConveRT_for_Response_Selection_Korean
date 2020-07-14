from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math
from . import Embeddings
# from embedding import Embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, config) -> None:
        
        """Instantiating Transformer class
        Args:
            config (Config): model config, the instance of data_utils.utils.Config
            vocab (Vocabulary): the instance of data_utils.vocab_tokenizer.Vocabulary
        """
        
        super(TransformerEncoder, self).__init__()
        self.input_embedding = Embeddings(config=config)
        self.pad_idx = config.pad_idx #vocab
        self.src_mask = None
        
        d_model = config.d_model #512
        n_head = config.n_head #8
        encoder_layers = config.encoder_layers
        feed_forward1_hidden = config.feed_forward1_hidden #1024
        feed_forward2_hidden = config.feed_forward2_hidden #1024
        dropout = config.dropout #0.1
        
        encoder_layers = TransformerEncoderLayer(d_model, n_head, feed_forward1_hidden, dropout)
        

        
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.proj_vocab_layer = nn.Linear(in_features=d_model, out_features=config.vocab_size)

        self.apply(self._initailze)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor) -> torch.Tensor:
        x_enc_embed = self.input_embedding(enc_input.long())
        
#         if self.src_mask is None or self.src_mask.size(0) != len(src):
#             device = src.device
#             mask = self._generate_square_subsequent_mask(len(src)).to(device)
#             self.src_mask = mask
        # Masking
        src_mask = enc_input == self.pad_idx

        # einsum ref: https://pytorch.org/docs/stable/torch.html#torch.einsum
        x_enc_embed = torch.einsum('ijk->jik', x_enc_embed)

        # transformer ref: https://pytorch.org/docs/stable/nn.html#torch.nn.Transformer
        feature = self.transfomrer(src = x_enc_embed,
                                   tgt = x_dec_embed,
                                   src_key_padding_mask = src_key_padding_mask,
                                   tgt_key_padding_mask = tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   tgt_mask = tgt_mask.to(device)) # src: (S,N,E) tgt: (T,N,E)

        logits = self.proj_vocab_layer(feature)
        logits = torch.einsum('ijk->jik', logits)

        return logits

    def _initailze(self, layer):
        if isinstance(layer, (nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight)