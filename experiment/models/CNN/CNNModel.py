import numpy as np
import torch
from torch import nn
from models.LSTM.LSTM import LSTMModule
from configs.CNNConfigs import args

class CNN(nn.Module):
    def __init__(self,_args=args):
        super().__init__()
        self.mid_channel = 16 #args.mid_channel
        self.in_channel = 1  # args.out_channels
        self.kernel_size = _args.kernel_size  # here take a square kernel
        # self.max_length = args.max_length  # default length of the input doc
        self.cuda = _args.cuda
        
        self.padding = int(
            (self.kernel_size - 1) / 2
        )  # using this padding value can generate the same shape as input
        self.convs = nn.Sequential(
            nn.Conv2d(
                self.in_channel,
                self.mid_channel,
                self.kernel_size,
                padding=self.padding,
            ),
            nn.Conv2d(
                self.mid_channel,
                self.in_channel,
                self.kernel_size,
                padding=self.padding,
            ),
            nn.Tanh(),
        )
        self.rnn = LSTMModule()
        # self.rnn = nn.LSTM(300,512,1,bidirectional = True,batch_first=True)
        if self.cuda:
            self.convs = self.convs.cuda()


    def forward(self,in_feature):
        lstm_out,lengths = self.rnn(in_feature,ret_raw = True)
        cnn_out = self.convs(lstm_out.unsqueeze(1)).squeeze(1)
        # bzs, channel,length, embed
        # in_f = in_feature.unsqueeze(1)
        # lstm_in = self.convs(in_f).squeeze(1)
        # lstm_out,_ = self.rnn(lstm_in)
        if self.cuda:
            lengths = lengths.cuda()
        return torch.sum(cnn_out,dim = 1)/lengths.view(-1,1)
