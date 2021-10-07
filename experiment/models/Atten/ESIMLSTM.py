import numpy as np
import torch
from torch import nn
from .ESIM import Attention
from configs.LSTMConfigs import args as _args
from models.LSTM.LSTM import LSTMModule


class ESIMLSTM(nn.Module):
    def __init__(self,args=_args):
        super().__init__()
        self.args = args
        self.lstm = LSTMModule()
        self.att = Attention(self.args)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()


    def forward(self,sents1,sents2):
        hidden1,length1 = self.lstm(sents1,ret_raw = True)
        hidden2,length2 = self.lstm(sents2,ret_raw = True)
        k_max = self.att(hidden1,hidden2)
        return k_max

