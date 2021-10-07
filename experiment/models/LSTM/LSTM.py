from .LSTMModel import LSTM
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence,pad_sequence
from configs.LSTMConfigs import args as _args
from models.Embedding.Embedding import WordEmbedding


class LSTMModule(nn.Module):
    def __init__(self,suffix = '',args=_args):
        super().__init__()
        self.lstm = LSTM(args)
        self.cuda = args.cuda
        self.embedding = WordEmbedding(suffix)
        self.hidden_size = args.hidden_size
        self.idf = self.embedding.get_idf


    def pack_input_sequence(self,_seq):
        
        seq = [self.embedding(s) for s in _seq]
        # idfs = [self.idf(s) for s in _seq]
        packed_input = pack_sequence(seq,enforce_sorted = False)
        # pad_idfs = pad_sequence(idfs,batch_first = True)
        if self.cuda:
            packed_input = packed_input.cuda()
            # pad_idfs = pad_idfs.cuda()
        return packed_input,None

    def forward(self,seq,ret_raw=False,ret_idfs = False):
        packed_seq,pad_idfs = self.pack_input_sequence(seq)
        pad_out,hn,cn = self.lstm(packed_seq)
        lstm_hiddens = pad_out[0]
        lengths = pad_out[1].to(torch.float)

        
        if self.cuda:
            lengths = lengths.cuda()
        sum_hiddens = torch.sum(lstm_hiddens,1)
        # return the averaged hiddens, raw hidden matrix
        if not ret_raw:
            return torch.div(sum_hiddens,lengths.view(-1,1))#,lstm_hiddens
        else:
            if not ret_idfs:
                return lstm_hiddens,lengths.view(-1,1)
            else:
                return lstm_hiddens,lengths.view(-1,1),pad_idfs
