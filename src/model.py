from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import DistilBertModel
import os
import math
from module import MultiHeadedAttention, Embeddings


class SharedBERTEncoder(nn.Module):
    
    def __init__(self, config):
        super(SharedBERTEncoder, self).__init__()
        self.config = config
        self.bert = DistilBertModel.from_pretrained('monologg/distilkobert')
        self.linear = nn.Linear(self.config.bert_hidden, self.config.d_model )
        
    def forward(self, input_ids):
        valid_length = (input_ids!=0).sum(1)
        attention_mask = self.get_attention_mask(input_ids, valid_length)
        outputs = self.bert(input_ids=input_ids.long(), attention_mask=attention_mask)
        all_encoder_layers = outputs[0]
        
        # Square-root N reduction
        encoder_out = all_encoder_layers.sum(1)/ math.sqrt(all_encoder_layers.size(1)) # (batch, seqlen, hidden) -> (batch, hidden)
        encoder_out = self.linear(encoder_out)
        return encoder_out
    
    def get_attention_mask(self, input_ids, valid_length):
        attention_mask = torch.zeros_like(input_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    
class SharedTransformerEncoder(nn.Module):
    """ Shared(context, reply) encoder for generate sentence representation.

        1. Embedding : sub-word(token) + positional embedding
        2. Transformer Encoder : multi-layer (6-layers on paper) transformer encoder
        3. 2-Head Self-attention
    """
    def __init__(self, config):
        super(SharedTransformerEncoder, self).__init__()
    
        self.embedding = Embeddings(config)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.nhead_inner, dim_feedforward=config.feed_forward1_hidden)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layer)
        self.attention = MultiHeadedAttention(h=config.nhead_outter, d_model=config.d_model)
    
    def forward(self, input_tensor):
        
        # 1. Embedding
        input_embedding = self.embedding(input_tensor) # (batch, maxlen, hidden)
        
        # 2. Transformer
        padding_mask = input_tensor == 0
        transformer_output = self.transformer_encoder(torch.einsum('ijk->jik', input_embedding),
                                                                src_key_padding_mask=padding_mask) # (maxlen, batch, hidden)
        transformer_output = torch.einsum('ijk->jik', transformer_output) # (maxlen, batch, hidden) -> batch, maxlen, hidden
        
        # 3. Two-Head Self Attetnion
        mask = (input_tensor > 0).unsqueeze(1).repeat(1, input_tensor.size(1), 1).unsqueeze(1) # (batch, 1, maxlen, maxlen)
        attention_out = self.attention(query=transformer_output, 
                                  key=transformer_output,
                                  value=transformer_output,
                                  mask=mask)
        
        # Square-root N reduction
        encoder_out = attention_out.sum(1)/ math.sqrt(attention_out.size(1)) # (batch, seqlen, hidden) -> (batch, hidden)        
        return encoder_out
        

        
class ContextOuterFeedForward(nn.Module):
    """3 Fully-Connected layers which are applied after SharedEncoder"""

    def __init__(self, config):
        super(ContextOuterFeedForward, self).__init__()
        
        input_hidden = config.d_model # 512
        intermediate_hidden = config.feed_forward2_hidden # 1024
        dropout_rate = config.dropout # 0.1

        self.linear_1 = nn.Linear(input_hidden, intermediate_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(intermediate_hidden, intermediate_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_3 = nn.Linear(intermediate_hidden, intermediate_hidden)
        self.norm = LayerNorm(intermediate_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print('outer feedforward size')
        #print(x.size())
        x = self.linear_1(x)
        x = self.linear_2(self.dropout(x))
        x = self.linear_3(self.dropout(x))
        return F.gelu(self.norm(x))
    

class ResponseOuterFeedForward(nn.Module):
    """3 Fully-Connected layers which are applied after SharedEncoder"""

    def __init__(self, config):
        super(ResponseOuterFeedForward, self).__init__()
        
        input_hidden = config.d_model # 512
        intermediate_hidden = config.feed_forward2_hidden # 1024
        dropout_rate = config.dropout # 0.1

        self.linear_1 = nn.Linear(input_hidden, intermediate_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(intermediate_hidden, intermediate_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_3 = nn.Linear(intermediate_hidden, intermediate_hidden)
        self.norm = LayerNorm(intermediate_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.linear_1(x)
        x = self.linear_2(self.dropout(x))
        x = self.linear_3(self.dropout(x))
        return fnn.gelu(self.norm(x))
    
    
class ConveRTDualEncoder(nn.Module):
    """ DualEncoder calculate similairty between context and reply by dot-product.
    
    1. shared_encoder
    2. context feedforward
    3. response feedforward
    """

    def __init__(self, config, args ,logger):

        super(ConveRTDualEncoder, self).__init__()
        if args.kobert==True:
            logger.info('Distill KoBERT Encoder!')
            self.shared_encoder=SharedBERTEncoder(config)
        else:
            logger.info('Transformer Encoder!')
            self.shared_encoder=SharedTransformerEncoder(config)
            
        self.context_feedfoward = ContextOuterFeedForward(config)
        self.encoder_feedfoward = ResponseOuterFeedForward(config)

    def forward(self, context_input=None, response_input=None, context_only=False, response_only=False):
        
        if context_input is not None:
            context_embed = self.shared_encoder(context_input)
            context_embed = self.context_feedfoward(context_embed) 
            if context_only:
                return context_embed
        
        if response_input is not None:
            response_embed = self.shared_encoder(response_input)
            response_embed = self.context_feedfoward(response_embed)     
            if response_only:
                return response_embed

        return context_embed, response_embed