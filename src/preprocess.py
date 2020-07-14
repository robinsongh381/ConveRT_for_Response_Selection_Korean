#!/usr/bin/env python
# coding: utf-8
import glob, re
from tqdm import tqdm
import torch
from random import random
from tokenizer import Tokenizer
tokenizer = Tokenizer()
regex = re.compile(".*?\((.*?)\)")

# Load files
tr_sent = list(open('../data/train.txt', 'r'))[1:]  # Exclude head
test_sent = list(open('../data/test.txt', 'r'))[1:] # Exclude head

# Preprocess
tr_cls_sep_context= []
tr_label_idx = []
tr_cls_sep_response = []

test_cls_sep_context= []
test_label_idx = []
test_cls_sep_response = []

mode = ['tr', 'test']

for m in mode:
    
    if m=='tr':
        sent = tr_sent
        save_context, save_label, save_response = tr_cls_sep_context, tr_label_idx, tr_cls_sep_response
    else:
        sent = test_sent
        save_context, save_label, save_response = test_cls_sep_context, test_label_idx, test_cls_sep_response
        
    for i in sent:
        label, context, response = i.replace('\n','').split('\t')
        context_cls_sep_idx = tokenizer.token_to_cls_sep_idx(context)
        response_cls_sep_idx = tokenizer.token_to_cls_sep_idx(response)
        
        save_label.append(int(label))
        save_context.append(context_cls_sep_idx)
        save_response.append(response_cls_sep_idx)       

                
    assert len(save_label)==len(save_context)==len(save_response)
     
    torch.save(save_label, '../data/{}_label.pt'.format(m))
    torch.save(save_context, '../data/{}_context.pt'.format(m))
    torch.save(save_response, '../data/{}_response.pt'.format(m))
    
    print('file saved : ../data/{}_sent.pt'.format(m))
    print('file saved : ../data/{}_label.pt'.format(m))
    print('file saved : ../data/{}_response.pt'.format(m))