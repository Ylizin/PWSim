from .tag_generic import GeneralEval
import logging
import torch
import numpy as np
from collections import Counter
from configs.globalConfigs import args as _args
import pandas as pd
from collections import defaultdict
from models.WMD.WMD import cal_WMD
import pickle

class WMDEval(GeneralEval):
    def __init__(self,args=_args,train_test_id=0):
        super().__init__()
        self.args = args
        self.train_pairs = self.data_set.train_pairs    
        self.test_keys = self.data_set.test_keys
        self.df_idx = self.data_set.df_idx
        self.df = self.data_set.raw_ids
        self.di = self.data_set.di
        self.model_str = "WMD"

    def train(self):
        topks = [5,10,15,20]
        p,r,f,n = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
        pred = {}
        for test_k in self.test_keys:
            query = list(map(lambda x:self.di.id2token[int(x)],self.data_set.pos[test_k][0]))
            sims = []
            for i in self.df_idx:
                candidate = list(map(lambda x:self.di.id2token[int(x)], self.df[i]))
                sims.append(cal_WMD(query,candidate))
            
            pos_ids = list(self.data_set.pos[test_k][1])
            sims_sort = np.argsort(sims,axis = -1)
            for tk in topks:
                    _p,_r,_f,_n = self.get_metrics(self.df_idx[sims_sort],pos_ids,tk) #s
                    p[tk].append(_p)
                    r[tk].append(_r)
                    f[tk].append(_f)
                    n[tk].append(_n)
        table = {}
        for k,v in p.items():
            table.update({'p_'+str(k):v})
        for k,v in r.items():
            table.update({'r_'+str(k):v})
        for k,v in f.items():
            table.update({'f_'+str(k):v})
        for k,v in n.items():
            table.update({'n_'+str(k):v})

        res=pd.DataFrame(table).mean().T
        print(res)
        pd.DataFrame(res).T.to_csv('./out/'+self.model_str+str(self.train_test_id)+'.csv')