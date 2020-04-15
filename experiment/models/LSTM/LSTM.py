from .LSTMModel import LSTM
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence
from configs.LSTMConfigs import args as _args
from models.Embedding.Embedding import WordEmbedding


class LSTMModule(nn.Module):
    def __init__(self,args=_args):
        super().__init__()
        self.lstm = LSTM(args)
        self.cuda = args.cuda
        self.embedding = WordEmbedding()

    def pack_input_sequence(self,seq):
        
        seq = [self.embedding(s) for s in seq]
        packed_input = pack_sequence(seq,enforce_sorted = False)
        if self.cuda:
            packed_input = packed_input.cuda()
        return packed_input

    def forward(self,seq,ret_raw=False):
        packed_seq = self.pack_input_sequence(seq)
        pad_out,hn,cn = self.lstm(packed_seq)
        lstm_hiddens = pad_out[0]
        lengths = pad_out[1]
        if self.cuda:
            lengths = lengths.cuda()
        sum_hiddens = torch.sum(lstm_hiddens,1)

        # return the averaged hiddens, raw hidden matrix
        if not ret_raw:
            return torch.div(sum_hiddens,lengths.view(-1,1))#,lstm_hiddens
        else:
            return lstm_hiddens,lengths.view(-1,1)
