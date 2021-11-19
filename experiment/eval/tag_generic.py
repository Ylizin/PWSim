import os
from torch import nn, optim
from data_utils.dataloader import TagDataSet
import logging
import numpy as np
import torch
from collections import defaultdict,Counter

class GeneralEval:
    def __init__(self,BoW=False,train_test_id=0,raw = True):
        self.model_str = ''
        self.train_test_id = train_test_id
        self.feature_extractor = None
        self.data_set = None
        self.optim = None
        self.init_data_set(BoW,train_test_id,raw=raw)
        self.q_ext = self.data_set.q_ext

    def get_cv_mean(self):
        tables = [pd.read_csv('./out/'+self.model_str+str(i)) for i in range(5)]
        
        
    def init_fe(self):
        pass

    def init_data_set(self,BoW=False,train_test_id = 0,raw = True):
        self.data_set = TagDataSet(train_test_id,raw = raw)
        self.data_set.map_word2id(BoW)
        logging.info('{} load data set done.'.format(self.__class__.__name__))
    
    def query_ext(self,query):
        '''
            query -> [[]]
        '''
        
        return [list(map(int,self.q_ext[' '.join(map(str,q))].split())) for q in query]        
    
    def get_BoWs(self,seq):
        
        return [self.data_set.filter_noise(list(Counter(i).items())) for i in seq]
    
    def init_optim(self):
        pass

    def init_data_loader(self):
        pass
    
    def _cal_dcg(self,seq):
        seq = torch.tensor(seq).to(float)
        _pos = ((torch.arange(len(seq))+2).to(float).log2())
        pow = 2**seq -1 
        return torch.sum(pow/_pos).item()
    
    # f1,ndcg
    def get_metrics(self,sims_sort,pos_ids,topk,threshold = 0.5):
        # pos ids -> [(li,score)]

        # score^2大于阈值才进行计算，目的是加强对完全匹配的奖励，例如完全匹配:1 ,匹配3/4为 0.75^2, 3/6 为 0.5^2
        pos_ids = list(filter(lambda x: (x[1])**2 >=threshold,sorted(pos_ids,key = lambda x:x[0],reverse = True)))
        id2score = {_id:s for li,s in pos_ids for _id in li}
        sims_sort_np = np.array(sims_sort[:topk])
        pos_ids_np = np.array([idx for i in pos_ids for idx in i[0]]).reshape(-1)
        inter = np.intersect1d(sims_sort_np,pos_ids_np)
        precision = len(inter)/(len(pos_ids_np) if topk>len(pos_ids_np) else topk)
        recall = len(inter)/len(pos_ids_np)
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) >0 else 0
        _dcg = [id2score[i] if i in inter else 0 for i in sims_sort_np]
        _dcg.sort(reverse=True)
        dcg = self._cal_dcg(_dcg)
        _idcg = torch.zeros(topk)
        for i in range(len(pos_ids_np[:topk])):
            _idcg[i] = id2score[pos_ids_np[i]]
        _idcg,_ = torch.sort(_idcg,descending = True)
        idcg = self._cal_dcg(_idcg)
        return precision,recall,f1,dcg/idcg
    
