from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.utils.data import Dataset
import constant as model_config
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ConveRTDataset(Dataset): 
    def __init__(self, dtype):
        
        # if add_response=='True': # concat response to intent (eg. 실행하기 앱을 실행합니다)
        self.context = torch.load('../data/{}_context.pt'.format(dtype))
        self.response = torch.load('../data/{}_response.pt'.format(dtype))
        self.label = torch.load('../data/{}_label.pt'.format(dtype))
        
        print('Load data from ../data/{}_sent.pt'.format(dtype))
        self.length = len(self.context)
        
    def __getitem__(self, idx):
        return self.context[idx], self.response[idx], self.label[idx]
    
    def __len__(self):
        return self.length
    

def convert_pad_collate(batch):

    (xx, yy, zz) = zip(*batch) # (sent, response)
    
    # Sentence process
    list_of_sent_len = [len(x) for x in xx]
    token_ids_sent = pad_sequences(xx, 
                              maxlen=max(list_of_sent_len), # model_config.maxlen 
                              value=model_config.pad_idx, 
                              padding='post',
                              dtype='long',
                              truncating='post') 
    # Response process
    list_of_response_len = [len(y) for y in yy]
    token_ids_response = pad_sequences(yy, 
                              maxlen=max(list_of_response_len), # model_config.maxlen 
                              value=model_config.pad_idx, 
                              padding='post',
                              dtype='long',
                              truncating='post')
    return token_ids_sent, token_ids_response, zz


class MappingDataset(Dataset): 
    def __init__(self):
        response_idx = torch.load('../data/response_idx.pt') # dict key:reponse label, value:cls_sep_response_idx
        self.response_label = []
        self.response_idx = []
        
        for k,v in response_idx.items():
            self.response_label.append(k)
            self.response_idx.append(v)
 
        self.length = len(self.response_idx)
        
    def __getitem__(self, idx):
        return self.response_label[idx], self.response_idx[idx]
    
    def __len__(self):
        return self.length
    

def mapping_pad_collate(batch):
    (xx, yy) = zip(*batch) # (sent, response)
    
    # Indicies process
    list_of_len = [len(y) for y in yy]
    token_ids = pad_sequences(yy, 
                              maxlen=max(list_of_len), # model_config.maxlen 
                              value=model_config.pad_idx, 
                              padding='post',
                              dtype='long',
                              truncating='post')
    
    return xx, token_ids


# class MappingDataset(Dataset): 
#     def __init__(self):

#         intent_mapping = torch.load('../data/aiet/intent_response_dict.pt')
#         # self.intent_text = [] # 실행하기
#         self.intent_label = [] # 0
#         # self.intent_indicies = [] # [2, 3056, 7789, 3]
#         self.response_to_idx= []
#         for k,v in intent_mapping.items():
#             # self.intent_text.append(v['text'])
#             self.intent_label.append(v['idx'])
#             # self.intent_indicies.append(v['tokenized'])
#             self.response_to_idx.append(v['response_to_idx'])
#         # print('Load data from ../data/aiet/{}/{}_sent.pt'.format(num_tr_sent, dtype))
#         self.length = len(self.response_to_idx)
        
#     def __getitem__(self, idx):
#         return self.intent_label[idx], self.response_to_idx[idx]
    
#     def __len__(self):
#         return self.length
    

# def mapping_pad_collate(batch):
#     (xx, yy) = zip(*batch) # (sent, response)
    
#     # Indicies process
#     list_of_len = [len(y) for y in yy]
#     token_ids = pad_sequences(yy, 
#                               maxlen=max(list_of_len), # model_config.maxlen 
#                               value=model_config.pad_idx, 
#                               padding='post',
#                               dtype='long',
#                               truncating='post')
    
#     return xx, token_ids



def _pad(sequence, maxlen):
    pad_idx = model_config.pad_idx
    if len(sequence)>=maxlen:
        return sequence[:maxlen]
    else:
        sequence = sequence + [model_config.pad_idx]*(maxlen-len(sequence)) 
        return sequence


def transform_to_bert_input(tokenized_idx_with_cls_sep, device):
        token_ids = pad_sequences(tokenized_idx_with_cls_sep, 
                                  maxlen=model_config.maxlen,
                                  value=model_config.pad_idx, 
                                  padding='post',
                                  dtype='long',
                                  truncating='post')

        valid_length = torch.tensor([len(tokenized_idx_with_cls_sep[0])]) # .long()
        segment_ids = [len(tokenized_idx_with_cls_sep[0])*[0]]

        # torch-compatible format
        token_ids = torch.tensor(tokenized_idx_with_cls_sep).float().to(device)
        valid_length = valid_length.clone().detach().to(device)
        segment_ids = torch.tensor(segment_ids).long().to(device)
        
        return token_ids, valid_length, segment_ids
    
    
    

