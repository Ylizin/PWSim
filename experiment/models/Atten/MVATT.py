import numpy as np
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.hidden_size =  args.hidden_size
        self.k = 10
        self.out_dim = 6
        self.att_weight = nn.Bilinear(2*self.hidden_size,2*self.hidden_size,self.out_dim,bias = False)
        self.sfm = nn.Softmax(dim = -1)
        if args.cuda:
            self.att_weight = self.att_weight.cuda()
        self.att_weight = self.att_weight.weight

    def forward(self,sent1,sent2):
        # sent1: bzs or 1,len1,embedding_size
        # sent2: bzs or 1,len2,embedding_size
        # weight: bzs,len1,len2
        mask1 = torch.sum(sent1,dim = -1).bool()
        mask2 = torch.sum(sent2,dim = -1).bool()
        
        # s1 -> (1,1,bzs,in1,len1)
        s1 = sent1.permute(0,2,1).unsqueeze(0).unsqueeze(0)
        # s2 -> (1,1,bzs,in2,len2)
        s2 = sent2.permute(0,2,1).unsqueeze(0).unsqueeze(0)

        # self.att_weight 
        # (out,in1,in2)
        # to (out,in1,1,1,in2)
        att_w = self.att_weight.unsqueeze(-2).unsqueeze(-2)
        
        # med (out,in1,bzs,1,len2)
        med = torch.matmul(att_w,s2)
        # med -> (out,len2,bzs,1,in1)
        # out = (out,len2,bzs,len1)
        # out -> (bzs,out,len1,len2)
        out = torch.matmul(med.permute(0,-1,2,3,1),s1).squeeze(-2).permute(2,0,-1,1)
        
        k_max = out.reshape(out.size(0),out.size(1),-1).topk(dim = -1,k=self.k)
        return k_max.values