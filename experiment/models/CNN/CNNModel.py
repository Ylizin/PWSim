import numpy as np
import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.mid_channel = 64 #args.mid_channel
        self.in_channel = 1  # args.out_channels
        self.kernel_size = args.kernel_size  # here take a square kernel
        self.max_length = args.max_length  # default length of the input doc
        self.cuda = args.cuda
        
        self.padding = int(
            (self.kernel_size - 1) / 2
        )  # using this padding value can generate the same shape as input
        self.convs = nn.Sequential(
            nn.Conv2d(
                self.in_channel,
                self.mid_channel,
                (3,3),
                padding=self.padding,
            ),
            nn.Conv2d(
                self.mid_channel,
                self.in_channel,
                (1,1),
                padding=self.padding,
            ),
            nn.Tanh(),
        )
        self.rnn = nn.LSTM(300,512,1,bidirectional = True,batch_first=True)
        if self.cuda:
            self.convs = self.convs.cuda()

    def forward(self,in_feature):
        # bzs, l, embed
        in_f = in_feature.unsqueeze(1)
        lstm_in = self.convs(in_f).squeeze(1)
        lstm_out,_ = self.rnn(lstm_in)
        return lstm_out
