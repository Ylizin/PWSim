import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.LSTM.LSTM import LSTMModule
from configs.CNNConfigs import args

class CNN(nn.Module):
    def __init__(self,_args=args):
        super().__init__()
        self.mid_channel = 16 #args.mid_channel
        self.in_channel = 1  # args.out_channels
        self.kernel_size = [2,3,4,5]  # here take a square kernel
        # self.max_length = args.max_length  # default length of the input doc
        self.cuda = _args.cuda
        
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.mid_channel, (k, args.embedding_size),padding=(k,0)) for k in self.kernel_size])
        self.dropout = nn.Dropout(args.dropout)
        

        # self.rnn = nn.LSTM(300,512,1,bidirectional = True,batch_first=True)
        self.rnn = LSTMModule()
        if self.cuda:
            self.convs = self.convs.cuda()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self,in_feature):
        lstm_out,lengths = self.rnn(in_feature,ret_raw = True)
        lstm_out = lstm_out.unsqueeze(1)
        out = [self.conv_and_pool(lstm_out,conv) for conv in self.convs]
        return torch.cat(out,1)
        # cnn_out = self.convs(lstm_out.unsqueeze(1)).squeeze(1)
        # bzs, channel,length, embed
        # in_f = in_feature.unsqueeze(1)
        # lstm_in = self.convs(in_f).squeeze(1)
        # lstm_out,_ = self.rnn(lstm_in)
        # if self.cuda:
            #lengths = lengths.cuda()
        #return torch.sum(cnn_out,dim = 1)/lengths.view(-1,1)
