#%%
import pandas as pd
from gensim import similarities
import collections
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import numpy as np
from itertools import combinations
from scipy.sparse import csr_matrix
#%%
df = pd.read_csv(r'./filtered_apis.csv')

def parse_list(s):
    return eval(s)

#%%
df['Categories'] = df['Categories'].apply(lambda x:parse_list(x))
df['Categories'] = df['Categories'].apply(lambda x:x[0])
len_ = df.groupby(['Categories']).count()
#选出长度为5-50的categories
len_ = len_[len_['Name'].apply(lambda x:5<x<51)]
ids = len_.index.drop('')
df = df.set_index('Categories').loc[ids]
d = [frame.index for l,frame in list(df.reset_index().groupby(['Categories']))]
#%%
# cat_corpus = df['Categories'].tolist()
#%%
# di = Dictionary(cat_corpus)
# docs = [di.doc2bow(d) for d in cat_corpus]
#%%
# model = TfidfModel(docs)
# cop = model[docs]
# sim_mat = similarities.MatrixSimilarity(cop)

#%%
# res = sim_mat[cop]
combs = [np.array(tuple(combinations(ids,2))) for ids in d]
c_r = np.concatenate(combs,axis=0)
#%%
col = c_r[:,0]
row = c_r[:,1]
data = np.ones_like(col)
res = csr_matrix((data,(col,row)),shape=(len(df),len(df))).todense()
for i in range(len(res)):
    res[i,i] = 1
#%%
# select 0.7 as train set and 0.3 docs as test set
index = np.arange(len(res))

train_idx = np.random.choice(a=len(res),size=int(0.7*(len(res))),replace = False)
train_idx.sort()
test_idx = np.setdiff1d(index,train_idx)
test_idx.sort()
#%%
# donot need triu, we calculate each pair twice, and we take the diag into consideration
# train_res = np.triu(res[train_idx][:,train_idx],1)
# test_res = np.triu(res[test_idx][:,test_idx],1)
#return the row,col of True elements for train set
# train_docs = np.argwhere(res>0.9)

#%%
#convert id back for argwhere
train_id_map = {i:v for i,v in enumerate(train_idx)}
test_id_map = {i:v for i,v in enumerate(test_idx)}
vec_train_map = np.vectorize(lambda x:train_id_map[x])
vec_test_map = np.vectorize(lambda x:test_id_map[x])
#%%
#in train set, we sample some 0 that total num equals to pos 
# select train matrix from res
train_docs = res[train_idx][:,train_idx]
train_pos  = np.argwhere(train_docs>0.9)
neg = np.argwhere(train_docs<1e-3)
# sample negs
train_neg_idx = np.random.choice(a=len(neg),size = len(train_pos),replace = False)
train_neg = neg[train_neg_idx]
# map back to id in df
train_pos=vec_train_map(train_pos)
train_neg=vec_train_map(train_neg)

#for test set we predict between every test doc and every doc in selected docs, not need to filter
#then sorted them
test_docs = res[test_idx]
test_pos = np.argwhere(test_docs>0.9)
test_pos[:,0]= vec_test_map(test_pos[:,0])

#%%
# selected_res = np.union1d(train_pos,test_pos)
# selected_res = np.union1d(selected_res,train_neg)
# reindex the df, select used text in it
# flat = selected_res.reshape(-1)
# uni = np.unique(flat)
# new_df = df.iloc[uni]
#new_
df.to_csv('./selected_docs.csv',index = False)

#%%
# re id for rows selected
# id2new_id = {v:i for i,v in enumerate(uni)}
# map_id2new_id = np.vectorize(lambda x: id2new_id[x])
#%%
# train_pos = map_id2new_id(train_pos)
# train_neg = map_id2new_id(train_neg)
# test_pos = map_id2new_id(test_pos)

np.save('train_pos',train_pos)
np.save('test_pos',test_pos)
np.save('train_neg',train_neg)
np.save('train_idx',train_idx)
np.save('test_idx',test_idx)
#%%
#for test
df = pd.read_csv('selected_docs.csv')
train_pos = np.load('./train_pos.npy')
train_neg = np.load('train_neg.npy')
test_pos = np.load('test_pos.npy')
train_idx = np.load('./train_idx.npy')
test_idx = np.load('./test_idx.npy')

# verify the diagonal
np.sum(np.apply_along_axis(lambda x:x[0]==x[1],1,test_pos))