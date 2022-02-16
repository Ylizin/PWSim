import numpy as np
import torch
from torch import nn
from configs.EmbeddingConfigs import args as _args
import pickle

class WordEmbedding(nn.Module):
    def __init__(self,suffix = '',args=_args,embedding_size=None):
        super().__init__()
        self.vocab_size = args.vocab_size
        if not embedding_size:
            self.embedding_dim = args.embedding_size
        else:
            self.embedding_dim = embedding_size
        self.pret_embeddings = args.pret_embeddings# + suffix 
        self.suffix = suffix
        self.idf_path = args.idf
        self.sfm = nn.Softmax(dim = -2)
        
        if not self.pret_embeddings:
            self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        else:
            self.embedding = torch.load(self.pret_embeddings)
            # only embedding tensors
            if not isinstance(self.embedding,nn.Module):
                self.embedding = nn.Embedding.from_pretrained(self.embedding,freeze = False)

        self.init_idf()
        self.bn = nn.BatchNorm1d(self.embedding_dim)
        self.cuda = args.cuda
        if self.cuda:
            self.embedding = self.embedding.cuda()
            self.bn = self.bn.cuda()
        
    def save(self):
        torch.save(self.embedding.cpu(),self.pret_embeddings+self.suffix)
        self.embedding.cuda()
    
    def init_idf(self):
        pass
        # idf_d = pickle.load(open(self.idf_path,'rb'))
        # idf_v = torch.ones((self.vocab_size,1))
        # for k,v in idf_d.items():
        #     idf_v[k] = v
        # self.idf = nn.Embedding.from_pretrained(idf_v,freeze=True)
    
    def get_idf(self,idx):
        pass
        
    def forward(self,idx):
        """
        forward embedding idxs
        
        :param idx: idx of words
        :type idx: tensor:[bzs,len]
        :return: embedding matrix
        :rtype: tensor:[bzs,len,embedding_size]
        """
        if len(idx)==0:
            idx = torch.tensor([1])
        if self.cuda:
            if isinstance(idx,torch.Tensor) :
                idx = idx.cuda()
            elif isinstance(idx,(list,set,tuple)):
                idx = [i.cuda() for i in idx ]
        # idf_v = self.sfm(self.idf(idx))
        return self.embedding(idx)#.mul(idf_v)