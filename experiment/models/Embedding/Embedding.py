import numpy as np
import torch
from torch import nn
from configs.EmbeddingConfigs import args as _args

class WordEmbedding(nn.Module):
    def __init__(self,args=_args):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.embedding_size
        self.pret_embeddings = args.pret_embeddings
        if not self.pret_embeddings:
            self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.load(self.pret_embeddings),freeze = False)
        
        self.cuda = args.cuda
        if self.cuda:
            self.embedding = self.embedding.cuda()
        
    def forward(self,idx):
        """
        forward embedding idxs
        
        :param idx: idx of words
        :type idx: tensor:[bzs,len]
        :return: embedding matrix
        :rtype: tensor:[bzs,len,embedding_size]
        """
        if self.cuda:
            idx = idx.cuda()
        return self.embedding(idx)