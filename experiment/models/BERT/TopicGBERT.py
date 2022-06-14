from .ALBERT import BERT

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence,pad_sequence
from configs.BERTConfigs import args as _args
from models.Embedding.Embedding import WordEmbedding
from models.VAE.VAE import VAE,vae_loss

class BERTModule(nn.Module):
    def __init__(self,suffix = '',args=_args):
        super().__init__()

        self.bert = BERT()
        self._cuda = args.cuda
        self.embedding = WordEmbedding(suffix)
        self.topic_size = args.topic_size
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.goal_size = args.goal_size
        self.goalEmbedding = WordEmbedding(suffix,embedding_size = self.goal_size)


        self.w = nn.Linear(self.topic_size, 2*self.hidden_size)
        # u for lstm_hidden out 
        self.u = nn.Linear(2*self.hidden_size+self.topic_size+2*self.hidden_size,2*self.hidden_size)
        # g for goal 
        self.g = nn.Linear(self.goal_size,2*self.hidden_size)
        # v for last out 
        self.v = nn.Linear(2*self.hidden_size, 1)
        self.o = nn.Linear(self.goal_size+self.topic_size+2*self.hidden_size,2*self.hidden_size)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.sfm = nn.Softmax(dim = -2)
        
        # self.rnn = nn.LSTM(2*self.input_size,self.hidden_size,1,bidirectional = args.bidirectional,batch_first =True)

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
    
    def forward(self,seq,bows,goals,vae_model=None):
        _,theta,loss,*_ = vae_model(bows)
        goal_vec_li = [torch.mean(self.g(self.goalEmbedding(goal)),0) for goal in goals]
        goal_vec = torch.stack(goal_vec_li,dim=0)
        pad_out = self.bert(seq)
        
#         out = pad_out[0]
#         lstm_out,(hn,cn) = self.rnn(pad_out[0])
        lstm_out = pad_out[0]
        lengths = torch.sum(pad_out[1],dim=-1).to(torch.float).view(-1,1)
        out = lstm_out
        if self.cuda:
            lengths = lengths.cuda()
        _w_theta = theta.expand(out.shape[1], -1, -1).transpose(0, 1)
        _goal_h = goal_vec.expand(out.shape[1], -1, -1).transpose(0, 1)
        out = torch.cat([out,_w_theta,_goal_h],-1)

        _w_theta = self.w(theta).expand(out.shape[1], -1, -1).transpose(0, 1)
        _u_h = self.u(out)
        _g = self.sigmoid(self.v(self.tanh(_w_theta + _u_h +_goal_h)))

        out = (
            lstm_out * _g
        )
        sum_hiddens = torch.sum(out,1)/lengths
        
        return sum_hiddens,loss
