from gensim.corpora import Dictionary
from .paths import embedding_path,dict_path,neg_tag,pos_tag,train_pairs,test_pairs,query_ext,syn2raw,ori2syn
from .process import load_texts
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle

def parse_list(s):
    return eval(s)

def load_dict(path = dict_path):
    return Dictionary.load(dict_path)

def load_w2v(path = embedding_path):
    return torch.load(path)

def load_syn2raw(path = syn2raw):
    return pickle.load(open(path,'rb'))

def load_pos_neg():
    pos = pickle.load(open(pos_tag,'rb'))
    neg = pickle.load(open(neg_tag,'rb'))
    q_ext = pickle.load(open(query_ext,'rb'))
    q_syn = pickle.load(open(ori2syn,'rb'))

    keys = [inter[0] for inter in pos]
    return pos,neg,keys,q_ext,q_syn

def split_tags():
    ori_pos,ori_neg,*_ = load_pos_neg()
    pos = [(list(map(int,q.strip().split())),r) for q,r in ori_pos]
    neg = [(list(map(int,q.strip().split())),r) for q,r,s in ori_neg]
    
    raw_pos = [(list(map(int,q.strip().split())),r) for q,r in ori_pos]
    
    key_len = int(len(pos)/5)
    keys = list(range(len(pos)))
    
    for i in range(5):
        test_keys_li = keys[key_len*i:key_len*(i+1)]
        train_keys = keys[:key_len*i] + keys[key_len*(i+1):]
#         print(len(train_keys))
#         train_keys = np.random.choice(keys,size = int(0.7*key_len),replace = False)
#         test_keys = set(keys) - set(train_keys)    
        pos_pairs = [(pos[i][0],_id,score) for i in train_keys for ids,score in pos[i][1] for _id in ids]
        neg_pairs = [(neg[i][0],ids,0) for i in train_keys for ids in neg[i][1]]
        
        train_pairs_li = pos_pairs+neg_pairs
        
        pickle.dump(train_pairs_li,open(train_pairs+str(i),'wb'))
        pickle.dump(test_keys_li,open(test_pairs+str(i),'wb'))
        
class TagDataSet(Dataset):
    def __init__(self,train_test_id=0,di=None,raw = True):
        super().__init__()

        if not di:
            di = load_dict()
        self.ori_pos,self.ori_neg,self.keys,self.q_ext,self.q_syn = load_pos_neg()
        # ori df 的header:
        self.ori_df = load_texts()
        
        self.di = di
        self.di.id2token = {v:k for k,v in self.di.token2id.items()}
        self.noise = set(k if v<0 else -1 for k,v in di.cfs.items())
        self.pos = [(list(map(int,q.strip().split())),r) for q,r in self.ori_pos]
        self.neg = [(list(map(int,q.strip().split())),r) for q,r,s in self.ori_neg]
        self.init_train_test(train_test_id)
#         if raw:
#             self.syn2raw = load_syn2raw()
#             self.pos = [(list(map(int,self.syn2raw[q].strip().split())),r) for q,r in self.ori_pos]
#             train_pairs = [(self.syn2raw[' '.join(map(str,q))],ids,score) for q,ids,score in self.train_pairs]
#             self.train_pairs = [(list(map(int,q.strip().split())),ids,score) for q,ids,score in train_pairs]
        
    
    def filter_noise(self,seq):
        return seq
        if isinstance(seq[0],tuple):
            return list(filter(lambda x:x[0] not in self.noise,seq))
        else:
            return list(filter(lambda x:x not in self.noise,seq))
    
    def map_word2id(self,BoW = False):
        d2b = lambda x: self.filter_noise(self.di.doc2bow(x.strip().split()))
        d2i = lambda x: torch.tensor(self.filter_noise(self.di.doc2idx(x.strip().split())))
        # if BoW:
                                        # 1 for raw ,0 add chunks
        self.bow_df = self.ori_df.iloc[:,1].apply(d2b)# use enhanced 
        self.tag_ids = self.ori_df.iloc[:,3].apply(parse_list)
#         self.tag_df = self.ori_df.iloc[:,0].apply(d2i) # use enhanced 
        # 0 enhanced , 1 not enhanced
        # self.raw_ids = self.ori_df.iloc[:,0].apply(d2i)  
        self.raw_ids = self.ori_df.iloc[:,1].str.cat(self.ori_df.iloc[:,5],sep = ' ').apply(d2i)  
    
        # self.main_ids = self.ori_df.iloc[:,5].apply(lambda x:[(x,1)])
          
                                                                #2 for ext 0 for enrich
        # self.ext_df = self.ori_df.iloc[:,2].str.cat(self.ori_df.iloc[:,5],sep = ' ').apply(d2b)
        self.ext_df = self.ori_df.iloc[:,2].apply(d2b)

        self.chunk_ids = self.ori_df.iloc[:,4].apply(d2i)
        self.goal_ids = self.ori_df.iloc[:,5].apply(d2i)
        self.df_idx = self.raw_ids.index
        self.raw_df = self.ori_df.iloc[:,1]#.apply(d2i)

    def init_train_test(self,train_test_id = 0):
        self.train_pairs = pickle.load(open(train_pairs+str(train_test_id),'rb'))
        self.test_keys = pickle.load(open(test_pairs+str(train_test_id),'rb'))
        np.random.shuffle(self.train_pairs)

        
#     def generate_train_test(self):
#         key_len = len(self.pos)
#         keys = range(key_len)
#         # get 0.7 train queries
#         train_keys = np.random.choice(keys,size = int(0.7*key_len),replace = False)
#         self.test_keys = set(keys) - set(train_keys)
#         # train pairs->[(query,[(li,score)])]
#         pos_pairs = [(self.pos[i][0],_id,score) for i in train_keys for ids,score in self.pos[i][1] for _id in ids]
#         neg_pairs = [(self.neg[i][0],ids,0) for i in train_keys for ids in self.neg[i][1]]
#         self.train_pairs = pos_pairs+neg_pairs
#         pickle.dump(self.train_pairs,open(train_pairs,'wb'))
#         pickle.dump(self.test_keys,open(test_pairs,'wb'))

    def __getitem__(self,_id):
        return self.raw_ids[_id]

    def __len__(self):
        return len(self.raw_ids)