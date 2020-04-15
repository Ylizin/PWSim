from gensim.corpora import Dictionary
from .paths import embedding_path,dict_path,train_neg_path,train_pos_path,test_pos_path
from .process import load_texts
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from configs.globalConfigs import total_idx_len


def load_dict(path = dict_path):
    return Dictionary.load(dict_path)

def load_w2v(path = embedding_path):
    return torch.load(path)

def load_train_test():
    train_pos = np.load(train_pos_path)
    train_neg = np.load(train_neg_path)
    test_pos = np.load(test_pos_path)
    return train_pos,train_neg,test_pos

class WordsDataSet(Dataset):
    def __init__(self,train_test=None,di=None):
        super().__init__()
        if not train_test:
            train_test = load_train_test()
        if not di:
            di = load_dict()
        self.train_pos,self.train_neg,self.test_pos = train_test
        self.total_idx = total_idx_len
        self.words_df = load_texts()
        self.di = di
        self.id_df = pd.DataFrame()
        self.bow_df = pd.DataFrame()
        
    def map_word2id(self):
        d2i = lambda x: torch.tensor(self.di.doc2idx(x.strip().split()))
        for k in self.words_df:
            self.id_df[k]=self.words_df[k].apply(d2i)
            # setattr(self,k+'_id',k_id)
    
    def map_doc2bow(self):
        d2b = lambda x:self.di.doc2bow(x.strip().split())
        def to_sparse(x):
            t = torch.tensor(x)
            idx = t[:,0].view(-1,1)
            v = t[:,1]
            t = torch.cat([torch.zeros_like(idx),idx],dim = 1)
            return torch.sparse.FloatTensor(t.t(),v,torch.Size([1,len(self.di)]))

        for k in self.words_df:
            k_bow = self.words_df[k].apply(d2b)
            self.bow_df[k] = k_bow.apply(to_sparse)

            # setattr(self,k+'_bow',k_bow)

    def set_train_cols(self,cols):
        self.train_cols = cols

    def __getitem__(self,_id):
        return self.id_df[self.train_cols].loc[_id,:]
    
    def __len__(self):
        return len(self.id_df)

        

