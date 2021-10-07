import numpy as np
import torch
from torch import nn
from .MVATT import Attention
from configs.LSTMConfigs import args as _args
from models.LSTM.LSTM import LSTMModule


class MVLSTM(nn.Module):
    def __init__(self,args=_args):
        super().__init__()
        self.args = args
        self.lstm = LSTMModule()
        self.att = Attention(self.args)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.pred = nn.Sequential(nn.Linear(self.att.k*self.att.out_dim,64),self.relu,
                                  nn.Linear(64,32),self.relu,
                                  nn.Linear(32,1),self.sig)
        if args.cuda:
            self.pred = self.pred.cuda()


    def forward(self,sents1,sents2):
        hidden1,length1 = self.lstm(sents1,ret_raw = True)
        hidden2,length2 = self.lstm(sents2,ret_raw = True)
        k_max = self.att(hidden1,hidden2)
        return self.pred(k_max.view(k_max.size(0),-1)).to(torch.float)

