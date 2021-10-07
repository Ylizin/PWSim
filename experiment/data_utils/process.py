from gensim.corpora import Dictionary
import pandas as pd
from .paths import tag_path,dict_path,pre_embedding_path,embedding_path
import numpy as np
from gensim.models import KeyedVectors
import pickle
import torch
from configs.globalConfigs import embedding_size

def load_texts(path = tag_path):
    df = pd.read_csv(path,header = None,index_col=0)
    return df

def load_dict(path = dict_path):
    return pickle.load(open(dict_path,'rb'))

def load_pre_w2v(path=pre_embedding_path):
    model = KeyedVectors.load_word2vec_format(path,binary=True)
    return model

def init_w2v(di,pre_model = None):
    len_di = len(di)
    w2v = torch.randn(len_di,embedding_size)
    if pre_model:
        # pre_w2i = {k:v.index for k,v in pre_model.items()}
        for i,k in di.items():
            if k in pre_model:
                w2v[i] = torch.from_numpy(pre_model[k])
    return w2v

def save_w2v(w2v):
    torch.save(w2v,embedding_path)

def init():
    print('init')
    d = load_dict()
    save_w2v(init_w2v(d,pre_model = load_pre_w2v()))

