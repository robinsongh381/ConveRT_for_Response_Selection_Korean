from __future__ import absolute_import
import torch
from typing import NamedTuple
# GPU device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Model
bert_hidden=768
maxlen = 32
hidden_size = 512
d_model = 512
attention_project = 64
nhead_inner = 2
nhead_outter = 2
encoder_layer = 3
feed_forward1_hidden = 1024
feed_forward2_hidden = 2048
dropout = 0.1
vocab_size = 8002

# Optimizer
learning_rate = 2e-4
gradient_accumulation_steps = 1
max_grad_norm = 5

# Steps
epoch = 1000
batch_size = 128

# Indicies
pad_idx = 0 # tok.convert_tokens_to_ids('[PAD]')

