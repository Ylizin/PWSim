from .LSTMModel import LSTM
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence,pad_sequence
from configs.LSTMConfigs import args as _args
from models.Embedding.Embedding import WordEmbedding
from models.VAE.VAE import VAE,vae_loss

class LSTMModule(nn.Module):
    def __init__(self,args=_args):
        super().__init__()
        self.lstm = LSTM(args)
        self._cuda = args.cuda
        self.embedding = WordEmbedding()
        self.hidden_size = args.hidden_size
        self.embedding_size = args.embedding_size
        self.topic_size = args.topic_size
        self.idf = self.embedding.get_idf
        self.w = nn.Linear(self.topic_size, 2*self.hidden_size)
        self.u = nn.Linear(2*self.hidden_size,2*self.hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.sfm = nn.Softmax(dim = -2)
        
        self.v = nn.Linear(2*self.hidden_size, 1)
        
        if args.cuda:
            self = self.cuda()

    def pack_input_sequence(self,_seq):
        seq = [self.embedding(s) for s in _seq]
        idfs = [self.idf(s) for s in _seq]
        packed_input = pack_sequence(seq,enforce_sorted = False)
        pad_idfs = pad_sequence(idfs,batch_first = True)
        if self._cuda:
            packed_input = packed_input.cuda()
            pad_idfs = pad_idfs.cuda()
        return packed_input,pad_idfs

    def forward(self,seq,bows,vae_model=None):
        _,theta,loss,*_ = vae_model(bows)
        packed_seq,pad_idfs = self.pack_input_sequence(seq)
        pad_out,hn,cn = self.lstm(packed_seq)
        lstm_hiddens = pad_out[0]
        lengths = pad_out[1].to(torch.float).view(-1,1)
        out = pad_out[0]
        if self.cuda:
            lengths = lengths.cuda()
        _w_theta = self.w(theta).expand(out.shape[1], -1, -1).transpose(0, 1)
        out = torch.cat([out,_w_theta],-1)
        return out,lengths
