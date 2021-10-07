import numpy as np
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.embedding_size = args.embedding_size
        self.hidden_size=  args.hidden_size
        self.att_weight = nn.Linear(4*self.hidden_size,4*self.hidden_size,bias = False)
        self.sfm = nn.Softmax(dim = -1)
        if args.cuda:
            self.att_weight = self.att_weight.cuda()

    def forward(self,sent1,sent2):
        # sent1: bzs or 1,len1,embedding_size
        # sent2: bzs or 1,len2,embedding_size
        # weight: bzs,len1,len2
        mask1 = torch.sum(sent1,dim = -1).bool()
        mask2 = torch.sum(sent2,dim = -1).bool()

        weight = torch.matmul(self.att_weight(sent1),sent2.permute(0,2,1))
        
        # sent1_w bzs,len1
        sent1_w = self.sfm(torch.sum(weight,dim =-1).masked_fill(~mask1,-float('inf'))).unsqueeze(-1)
        # sent2_w bzs,len2
        sent2_w = self.sfm(torch.sum(weight,dim =-2).masked_fill(~mask2,-float('inf'))).unsqueeze(-1)
        
        return sent1,torch.mul(sent2_w,sent2)