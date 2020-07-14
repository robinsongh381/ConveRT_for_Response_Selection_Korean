from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConveRTLoss(nn.Module):
    def __init__(self, config, loss_type):
        
        """
        calculate similarity matrix (CONTEXT_BATCH_SIZE, REPLY_BATCH_SIZE) between context and reply
            :param split_size: split matrix into fixed-size, defaults to 1
            :type split_size: int, optional
        """
        super().__init__()
        self.device = config.device
        self.loss_type = loss_type

    def forward(self, context_embed: torch.Tensor, reply_embed: torch.Tensor) -> torch.Tensor:
        
        """
        calculate context-reply matching loss
        """
                 
        assert context_embed.size(0) == reply_embed.size(0)
        true_label = torch.arange(context_embed.size(0)).to(self.device)
        total_size = context_embed.size(0)

        if self.loss_type=='dot':
            """
                No negative sampling applied
            """
            similarity = torch.matmul(context_embed, reply_embed.transpose(1,0))
            # true_label = torch.arange(sim.size(0))
            loss = F.cross_entropy(input=similarity, target=true_label)
            correct_count = similarity.argmax(-1).eq(true_label).sum().item()
            correct_count_for_recall = true_label.eq(similarity.argmax(-1)).sum().item()
            predict_label = similarity.argmax(-1).tolist()

            return loss, [correct_count, correct_count_for_recall], total_size, predict_label   
                 
#         if self.loss_type=='cosine':
  
#             cosine_sim =nn.CosineSimilarity(dim=-1)
#             cosine_sim_matrix = torch.tensor([])

#             for i in range(context_embed.size(0)): # context_embed.size(0) = batch_size
#                 cs = cosine_sim(context_embed[i], reply_embed)
#                 cosine_sim_matrix = torch.cat([cosine_sim_matrix, cs])
#             cosine_sim_matrix = cosine_sim_matrix.view(context_embed.size(0),-1)


#             # Postivie Sampling
#             positive_sample_sum = torch.diag(cosine_sim_matrix, 0).sum(0)
#             correct_count = cosine_sim_matrix.argmax(-1).eq(true_label).sum().item()
#             predict_label = cosine_sim_matrix.argmax(-1).tolist()

#             # Negative Sampling
#             negative_sample_matrix = cosine_sim_matrix.clone()#.fill_diagonal_(-1e10)
#             negative_sample_matrix = torch.exp(negative_sample_matrix)
#             negative_sample_matrix = negative_sample_matrix.clone()#.fill_diagonal_(1)
#             negative_sample_matrix = torch.log(negative_sample_matrix)
#             negative_sample_matrix = negative_sample_matrix.sum(1)

#             negative_sample_sum = negative_sample_matrix.sum(0)
#             loss = -(positive_sample_sum-negative_sample_sum)


#             return loss, correct_count, total_size, predict_label