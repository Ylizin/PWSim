#%%
import pickle
from itertools import chain

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary

from eda import lem


def lem_sent(s):
    if not s:
        return ''
    return ' '.join(lem(s))
def parse_list(s):
    return eval(s)

def ext_servs(tags):
    t = [lem_sent(tag2explan[t]) if t in tag2explan else '' for t in tags]
    return ' '.join(t)

syn2raw = pickle.load(open('./syn2raw','rb'))
tag2explan = pickle.load(open('./explan','rb'))
tag_di = pickle.load(open('tag_di','rb'))
tag2servs = pickle.load(open('tag2servs','rb'))
df = pd.read_csv(r'./filtered_datas.csv',index_col=0)
#%%
# all servs presented
all_servs = set()
for tag in tag2servs.values():
    for servs,_ in tag:
        all_servs |= set(servs)
tag_servs = df.loc[all_servs]['internal_enrich'].apply(lem)
raw_servs = df.loc[all_servs]['Description'].apply(lem)
servs_tag = df.loc[all_servs]['Categories'].apply(parse_list)
servs_tag_ext = servs_tag.apply(ext_servs)
servs_tag_ids = servs_tag.apply(tag_di.doc2bow)

di = Dictionary(tag_servs.tolist())
di.add_documents(raw_servs)

tag_di.id2token = {v:k for k,v in tag_di.token2id.items()}
tag_servs = tag_servs.apply(' '.join)
raw_servs = raw_servs.apply(' '.join)

tag_servs = pd.DataFrame([tag_servs,raw_servs,servs_tag_ext,servs_tag_ids]).T

#%%
ori_para={}
extend_rel = {}
ori2raw = {}
for k,v in tag2servs.items():
    ori_sent = ' '.join(['{} '.format(lem_sent(tag_di.id2token[int(tag)])) for tag in k.split(',')])
    ext_sent = ' '.join([lem_sent(tag2explan[tag_di.id2token[int(tag)]]) for tag in k.split(',')])
    # ext_sent = ' '.join([lem_sent(tag2explan[tag]) for tag in syn2raw[k]])
    # ori2raw[ori_sent] = syn2raw[k]
    extend_rel[ori_sent] =ext_sent
    ori_para[ori_sent] = v

#%%
tag2servs_tup=[]
for k,v in ori_para.items():
    for li,score in v:
        tag2servs_tup.append((k,li,score))
        
ori2ext = {}
tag_texts = [v.split() for v in chain(extend_rel.values(),extend_rel.keys())] + list(ori2raw.values())
di.add_documents(tag_texts)

tag2servs_di = {}
for k,v,s in tag2servs_tup:
    _k = ' '.join(map(str,di.doc2idx(k.strip().split())))
    l = tag2servs_di.setdefault(_k,list())
    ori2ext[_k] = ' '.join(map(str,di.doc2idx(extend_rel[k].strip().split())))

    l.append((v,s))
# tag2servs_tup = [(di.doc2idx(k),v,s) for k,v,s in tag2servs_tup]
# in form of {tag:[([],score),([],score)]}
tag2servs_tup = list(tag2servs_di.items())

#%%
neg_tag2servs_tup = []
# neg-> [(tag,li,score)]
for k,v in tag2servs_di.items():
    seen_doc = set(chain(*[s[0] for s in v]))
    unseen_servs = all_servs-seen_doc
    neg_tag2servs_tup.append((k.replace(',',' '),np.random.choice(list(unseen_servs),size=20*len(v),replace = False),0))

#%%
d2idx = lambda x: ' '.join(map(str,di.doc2idx(x)))
ori2raw = {d2idx(k.strip().split()):d2idx(v) for k,v in ori2raw.items()}
tag_servs['main_cat'] = df.main_cat.apply(lambda x:[x])
tag_servs['main_ids'] = df.main_ids

tag_servs.to_csv('./tag_servs.csv',header=None)
pickle.dump(tag2servs_tup,open('./pos_tag2servs','wb'))
pickle.dump(neg_tag2servs_tup,open('./neg_tag2servs','wb'))
pickle.dump(di,open('./dict','wb'))
pickle.dump(ori2ext,open('./query_ext','wb'))
pickle.dump(ori2raw,open('./ori2raw','wb'))