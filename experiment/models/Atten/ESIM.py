import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.hidden_size =  args.hidden_size
        self.sfm = nn.Softmax(dim = -1)
        self.lstm = nn.LSTM(8*self.hidden_size,self.hidden_size,batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        if args.cuda:
            self.fc = self.fc.cuda()
            self.lstm = self.lstm.cuda()

    def att(self,mat1,mat2,mask1,mask2):
        attention = torch.matmul(mat1, mat2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))        
        
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1,mat2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, mat1)
        
        return x1_align, x2_align
    
    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)
    
    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)
    
    def forward(self,sent1,sent2):
        # sent1: bzs or 1,len1,embedding_size
        # sent2: bzs or 1,len2,embedding_size
        # weight: bzs,len1,len2
        mask1 = ~torch.sum(sent1,dim = -1).bool()
        mask2 = ~torch.sum(sent2,dim = -1).bool()

        att_mat1,att_mat2 = self.att(sent1,sent2,mask1,mask2)
        if sent1.size(0) == 1:
            att_mat1 = att_mat1.expand(att_mat2.size(0),*att_mat1.size()[1:])
            sent1 = sent1.expand(att_mat2.size(0),*att_mat1.size()[1:])
        comb1 = torch.cat([sent1,att_mat1,self.submul(sent1,att_mat1)],-1)
        comb2 = torch.cat([sent2,att_mat2,self.submul(sent2,att_mat2)],-1)
        
        f_mat1,_ = self.lstm(comb1)
        f_mat2,_ = self.lstm(comb2)
        
        f1 = self.apply_multiple(f_mat1)
        f2 = self.apply_multiple(f_mat2)
        
        f = torch.cat([f1,f2],dim = -1)
        
        return self.fc(f)