from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp


# Tokenizer
_, vocab = get_pytorch_kobert_model()
_tok_path = get_tokenizer()
split_fn = nlp.data.BERTSPTokenizer(_tok_path, vocab, lower=False)


class Tokenizer:
    """ Tokenizer class"""

    def __init__(self):
        # self._vocab = vocab
        self.tokenizer = split_fn
        self.vocab_size = len(vocab)
        self.cls_idx = self.tokenizer.vocab.to_indices('[CLS]')
        self.sep_idx = self.tokenizer.vocab.to_indices('[SEP]')
        self.bos_idx = self.tokenizer.vocab.to_indices('[BOS]')
        self.eos_idx = self.tokenizer.vocab.to_indices('[EOS]')
        self.mask_idx = self.tokenizer.vocab.to_indices('[MASK]')
        self.pad_idx = self.tokenizer.vocab.to_indices('[PAD]')
 
        
    def __call__(self, text_string):
        return self.tokenizer(text_string)
    
    
    def sentencepiece_tokenizer(self, raw_text):
        return self.tokenizer(raw_text)
    
        
    def token_to_cls_sep_idx(self, text):
        
        tokenized_text = self.tokenizer(text)
        idx_tok = []
        for t in tokenized_text:
            idx = self.tokenizer.convert_tokens_to_ids(t)
            idx_tok.append(idx)
        idx_tok = [self.cls_idx] + idx_tok + [self.sep_idx]

        return idx_tok
    
    
    def idx_to_token(self, idx_list):
        out = []
        for i in idx_list:
            token = self.tokenizer.vocab.to_tokens(i)
            out.append(token)
        
        return out
    
    
    def pad(self, sequence, maxlen):
        if len(sequence)>=maxlen:
            return sequence[:maxlen]
        else:
            extra_len = maxlen-len(sequence)
            sequence = sequence + [self.pad_idx]*extra_len
            
            return sequence
    
    