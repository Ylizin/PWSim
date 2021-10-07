from .tag_generic import GeneralEval
import logging
import torch
from torch import nn
import numpy as np
from collections import Counter
from configs.globalConfigs import args as _args
import pandas as pd
from collections import defaultdict
from gensim.similarities import MatrixSimilarity
from models.Doc2Vec.train_d2v import *
import pickle

class D2VEval(GeneralEval):
    def __init__(self,args=_args):
        super().__init__()
        self.args = args
        self.train_pairs = self.data_set.train_pairs    
        self.test_keys = self.data_set.test_keys
        self.df_idx = self.data_set.df_idx
        self.df = self.data_set.tag_df
        self.di = self.data_set.di
        self.cos = nn.CosineSimilarity(dim = -1)
        
        self.init_tfidf()
        
    def init_tfidf(self):
        corp = self.df.tolist()
        corp = [[self.di.id2token[int(i)] for i in li] for li in corp]
        self.d2v = train_d2v(corp)
        self.sims = torch.tensor([self.d2v.infer_vector(c) for c in corp])


    def train(self):
        topks = [1,5,10,15,20,25,30]
        p,r,f,n = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
        pred = {}
        for test_k in self.test_keys:
            query = [self.di.id2token[int(i)] for i in self.data_set.pos[test_k][0]]
            
            query_vec = torch.tensor(self.d2v.infer_vector(query)).view(1,-1)
            sims = -self.cos(self.sims,query_vec)
            pos_ids = list(self.data_set.pos[test_k][1])
            sims_sort = np.argsort(sims,axis = -1)
            for tk in topks:
                    _p,_r,_f,_n = self.get_metrics(self.df_idx[sims_sort],pos_ids,tk) #s
                    p[tk].append(_p)
                    r[tk].append(_r)
                    f[tk].append(_f)
                    n[tk].append(_n)
        p = {k:np.mean(v) for k,v in p.items()}
        r = {k:np.mean(v) for k,v in r.items()}
        f = {k:np.mean(v) for k,v in f.items()}
        n = {k:np.mean(v) for k,v in n.items()}
        table = {'p':p,'r':r,'f':f,'n':n}
        print(pd.DataFrame(table).T,flush = True)