from gensim.corpora import Dictionary
import pandas as pd
from .paths import eda_path,dict_path,pre_embedding_path,embedding_path
import numpy as np
from gensim.models import KeyedVectors
import torch
from configs.globalConfigs import embedding_size

def load_texts(path = eda_path):
    df = pd.read_csv(path)
    texts = df[['sr','ri','rs','rd','des']]
    return texts

def _make_dict(texts):
    t = texts.values.flatten()
    corp = [s.strip().split() for s in t]
    di = Dictionary(corp)
    return di

def init_dict():
    d = _make_dict(load_texts())
    return d

def save_dict(d,path = dict_path):
    d.save(path)

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

if __name__ == '__main__':
    d = init_dict()
    save_dict(d)
    save_w2v(init_w2v(d,pre_model = load_pre_w2v()))

