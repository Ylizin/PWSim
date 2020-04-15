import numpy as np
import torch
from torch import nn
from .Attention import Attention
from configs.LSTMConfigs import args as _args
from models.LSTM.LSTM import LSTMModule


class AttentionLSTM(nn.Module):
    def __init__(self,args=_args):
        self.args = args
        self.lstm = LSTMModule()
        self.att = Attention(self.args)

    def forward(self,sents1,sents2):
        hidden1,length1 = self.lstm(sents1,ret_raw = True)
        hidden2,length2 = self.lstm(sents2,ret_raw = True)
        hidden1,hidden2 = self.att(hidden1,hidden2)
        return torch.div(torch.sum(hidden1,1),length1.view(-1,1)),torch.div(torch.sum(hidden2,1),length2.view(-1,1))

