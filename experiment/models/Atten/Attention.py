import numpy as np
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.embedding_size = args.embedding_size
        self.att_weight = nn.Linear(self.embedding_size,self.embedding_size)

    def forward(self,sent1,sent2):
        # sent1: bzs or 1,len1,embedding_size
        # sent2: bzs or 1,len2,embedding_size
        # weight: bzs,len1,len2
        weight = torch.matmul(self.att_weight(self.sent1),sent2.permute(0,2,1))
        # sent1_w bzs,len1
        sent1_w = torch.sum(weight,dim =-1).unsqueeze(-1)
        # sent2_w bzs,len2
        sent2_w = torch.sum(weight,dim =-2).unsqueeze(-1)
        
        return torch.mul(sent1_w,sent1),torch.mul(sent2_w,sent2)