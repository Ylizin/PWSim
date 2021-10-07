from .tag_generic import GeneralEval
import logging
import torch
import numpy as np
from collections import Counter
from configs.globalConfigs import args as _args
from gensim.models import TfidfModel
from gensim.models.ldamodel import LdaModel
import pandas as pd
from collections import defaultdict
from gensim.similarities import MatrixSimilarity
import pickle

class TFIDFEval(GeneralEval):
    def __init__(self,args=_args):
        super().__init__(True)
        self.args = args
        self.train_pairs = self.data_set.train_pairs    
        self.test_keys = self.data_set.test_keys
        self.df_idx = self.data_set.df_idx
        self.df = self.data_set.bow_df

        self.init_tfidf()
        
    def init_tfidf(self):
        corp = self.df.tolist()
        self.tfidf = TfidfModel(corp)
        #pickle.dump(self.tfidf,open('./data/tag/tfidf','wb'))
        #self.tfidf = LdaModel(corp,id2word = self.data_set.di,num_topics=500,iterations = 3000)
        self.sims = MatrixSimilarity(self.tfidf[corp])


    def train(self):
        topks = [1,5,10,15,20,25,30]
        p,r,f,n = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
        pred = {}
        for test_k in range(len(self.data_set.pos)):
            query = self.data_set.filter_noise(list(Counter(self.data_set.pos[test_k][0]).items()))
            
            query_vec = self.tfidf[query]
            sims = -self.sims[query_vec]
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
        print(pd.DataFrame(table).mean().T)
