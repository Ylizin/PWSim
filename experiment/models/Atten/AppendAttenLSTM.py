import numpy as np
import torch
from torch import nn
from .Attention import Attention
from configs.LSTMConfigs import args as _args
from models.LSTM.TopicAppendLSTM import LSTMModule


class AttentionLSTM(nn.Module):
    def __init__(self,args=_args):
        super().__init__()
        self.args = args
        self.lstm = LSTMModule()
        self.att = Attention(self.args)

    def forward(self,sents1,sents2,bows1,bows2,vae):
        hidden1,length1 = self.lstm(sents1,bows1,vae)
        hidden2,length2 = self.lstm(sents2,bows2,vae)
        hidden1,hidden2 = self.att(hidden1,hidden2)
        return torch.div(torch.sum(hidden1,1),length1.view(-1,1)),torch.sum(hidden2,1)

