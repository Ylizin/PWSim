from .LSTMModel import LSTM
from .GoalLSTMModel import GLSTM 
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence,pad_sequence
from configs.LSTMConfigs import args as _args
from models.Embedding.Embedding import WordEmbedding
from models.VAE.VAE import VAE,vae_loss

class LSTMModule(nn.Module):
    def __init__(self,suffix = '',args=_args):
        super().__init__()
        self.lstm = LSTM(args)
        self._cuda = args.cuda
        self.embedding = WordEmbedding(suffix)
        self.hidden_size = args.hidden_size
        self.topic_size = args.topic_size
        self.goal_size = args.goal_size
        self.goalEmbedding = WordEmbedding(suffix,embedding_size = self.goal_size)
        # self.idf = self.embedding.get_idf
        # w for topic
        self.w = nn.Linear(self.topic_size, 2*self.hidden_size)
        # u for lstm_hidden out 
        self.u = nn.Linear(self.goal_size+2*self.hidden_size+self.topic_size,2*self.hidden_size)
        # g for goal 
        self.g = nn.Linear(self.goal_size,2*self.hidden_size)
        # v for last out 
        self.v = nn.Linear(2*self.hidden_size, 1)
        self.glstm = GLSTM(args)


        self.o = nn.Linear(self.goal_size+self.topic_size+2*self.hidden_size,2*self.hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.sfm = nn.Softmax(dim = -2)
        
        
        if args.cuda:
            self = self.cuda()

    def pack_input_sequence(self,_seq):
        seq = [self.embedding(s) for s in _seq]
        # idfs = [self.idf(s) for s in _seq]
        packed_input = pack_sequence(seq,enforce_sorted = False)
        # pad_idfs = pad_sequence(idfs,batch_first = True)
        if self._cuda:
            packed_input = packed_input.cuda()
            # pad_idfs = pad_idfs.cuda()
        return packed_input,None

    def pack_goal_sequence(self,_goal):
        _goal = [self.g(self.goalEmbedding(s)) for s in _goal]
        # idfs = [self.idf(s) for s in _seq]
        packed_input = pack_sequence(_goal,enforce_sorted = False)
        # pad_idfs = pad_sequence(idfs,batch_first = True)
        if self._cuda:
            packed_input = packed_input.cuda()
            # pad_idfs = pad_idfs.cuda()
        return packed_input,None 
    
    def forward(self,seq,bows,goals,vae_model=None):
        _,theta,loss,*_ = vae_model(bows)
        goal_vec_li = [torch.mean(self.g(self.goalEmbedding(goal)),0) for goal in goals]
        goal_vec = torch.stack(goal_vec_li,dim=0)
        # packed_goal,pad_idfs = self.pack_input_sequence(seq)
        # pad_goal,hn,cn = self.lstm(packed_goal)
        # goal_vec = torch.sum(pad_goal[0],1)/torch.sum(pad_goal[1],dim = -1).to(torch.float).view(-1,1).cuda()

        packed_seq,pad_idfs = self.pack_input_sequence(seq)
        pad_out,hn,cn = self.lstm(packed_seq)
        lstm_hiddens = pad_out[0]

        lengths = torch.sum(pad_out[1],dim = -1).to(torch.float).view(-1,1)
        out = pad_out[0]
        if self.cuda:
            lengths = lengths.cuda()
        mask = torch.sum(out,dim = -1).to(torch.bool)
        # theta_cat_goal = torch.cat([theta,goal_vec],-1)
        _w_theta = theta.expand(out.shape[1], -1, -1).transpose(0, 1)
        _goal_h = goal_vec.expand(out.shape[1], -1, -1).transpose(0, 1)
        out = torch.cat([out,_w_theta,_goal_h],-1)
        _w_theta = self.w(theta).expand(out.shape[1], -1, -1).transpose(0, 1)
        _u_h = self.u(out)
        
        _g = self.sigmoid(self.v(self.tanh(_w_theta + _u_h +_goal_h)))

        _g = torch.masked_fill(_g,~mask.unsqueeze(-1),0) #mask is important
        out = (
            lstm_hiddens * _g
        )
        sum_hiddens = torch.sum(out,1)/lengths
        # sum_hiddens = torch.cat([sum_hiddens,theta,goal_vec],dim = -1 )
        return sum_hiddens,_g
