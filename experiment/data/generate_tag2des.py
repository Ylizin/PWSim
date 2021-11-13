#%%
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from gensim.corpora import Dictionary
import pickle
from itertools import chain
from collections import Counter

tag2explan = pickle.load(open('./explan', 'rb'))
tag2explan = dict(filter(lambda x: x[1] is not None and len(
    x[1].strip().split()) > 3, tag2explan.items()))

df = pd.read_csv(r'./processed.csv')
f_t = set(['data', 'reference', 'other', '', 'tools',
           'application development', 'api'])

def parse_list(s):
    return list(filter(lambda x:x not in f_t,eval(s)))


def parse_set(s):
    f_t = set(['other','','api'])
    s = set(filter(lambda x:x not in f_t,eval(s)))
    return s


inp = df.Description.str.split().tolist()
d = Dictionary(inp)
d.id2token = {v: k for k, v in d.token2id.items()}
c = [d.doc2bow(l) for l in inp]

#%%
main_cat = df.Categories.apply(parse_list).apply(lambda x: x[0] if x else '')

df = df[main_cat.apply(lambda x:x != '')]


main_cat = main_cat[main_cat.apply(lambda x:x not in f_t)]
c = Counter(main_cat.to_list())
coun = pd.Series(c)
# main_coun = coun[coun > 122]
# sel_cats = set(main_coun.index)
df['main_cat'] = df.Categories.apply(lambda x: parse_list(x)[0])


cats = df.Categories.apply(parse_set)
# cat_len = cats.apply(lambda x: len(x))
# filter too much tags services
tag_di = Dictionary(cats.to_list())
tag_di.id2token = {v: k for k, v in tag_di.token2id.items()}
# tags = pd.Series(list(map(lambda x: tag_di.id2token[x], tag_di.dfs.keys())))
tag_freq = pd.Series(list(tag_di.dfs.values()))
# filter too frequent tags
# filtered_tags = set(tags[tag_freq < 500].tolist())
# cats = cats.apply(lambda x: x.intersection(filtered_tags))
#%%
des = df.loc[cats.index, 'Description']
cat_len = cats.apply(lambda x: len(x))
cats = cats[cat_len.apply(lambda x: 2 <= x < 7)]

cats_id = cats.apply(tag_di.doc2idx)
main_ids = df.main_cat.apply(lambda x: tag_di.token2id[x])
cats_id = cats_id.apply(set)
df = df.loc[cats.index]
df['cats_id'] = cats_id
#%%


def find_common_tags(i, j, cats_id, tags):
    # 找到所有满足要求的tags
    if len(cats_id.iloc[i]) < 3 or len(cats_id.iloc[j]) < 3:
        return
    common_ids = cats_id.iloc[i] & cats_id.iloc[j]
    if len(common_ids) >= 3:
        tags.add(','.join(map(str, common_ids)))


def mp_f(group_df):
    tags = set()
    cats_id = group_df.cats_id
    for i in range(len(cats_id)):
        for j in range(i+1,len(cats_id)):
            find_common_tags(i, j, cats_id, tags)
    return tags


tags = set()
with ProcessPoolExecutor(max_workers=20) as exc:
    for _,group_df in df.groupby('main_cat'):
        exc.submit(mp_f,group_df).add_done_callback(
            lambda x: tags.update(x.result()))
print("tags len:{}",len(tags))

# tag2servs = {query:{score:[]}}
# 拿到和tags匹配的所有des

def get_servs(tag, tag2servs):
    _tag = set(map(int, tag.split(',')))
    for i in range(len(cats_id)):
        same_len = len(_tag & cats_id.iloc[i])
        if same_len >= 3:
            rel_servs = tag2servs.setdefault(tag, dict())
            servs = rel_servs.setdefault(same_len/len(_tag), list())
            servs.append(cats_id.index[i])


def mp_get_servs(tag):
    tag2servs = {}
    get_servs(tag, tag2servs)
    return tag2servs


tag2servs = {}
# get all matched tag2des
with ProcessPoolExecutor(max_workers=20) as exc:
    for tag in tags:
        exc.submit(mp_get_servs, tag).add_done_callback(
            lambda x: tag2servs.update(x.result()))
# tag2servs = {k:[(li,s) for s,li in v.items()] for k,v in tag2servs.items()}
print("tag2servs len:{}",len(tags))

#%%
tag_di.id2token = {v: k for k, v in tag_di.token2id.items()}
items = [(map(int, q.split(',')), v) for q, v in tag2servs.items()]
# ave des len is 66 and ave q len is 3
querys = [(list(tag_di.id2token[k] for k in m), v) for m, v in items]

tag_di.add_documents([x[0] for x in querys])
tag2servs = {','.join(map(str, tag_di.doc2idx(k))): [
    (li, s) for s, li in v.items()] for k, v in querys}
pickle.dump(tag2servs, open('./tag2servs', 'wb'))
pickle.dump(tag_di, open('./tag_di', 'wb'))
tags = list(tag_di.token2id.keys()) + list(tag_di.token2id.keys())
#%%
# query长度不小于3， 正例数目不大于20
# filt = lambda x: list(chain(*[li if s>0.6 else [] for li,s in x]))
# filtered_tag2servs=list(filter(lambda x: len(x[0].split(','))>3 and len(filt(x[1]))>10,tag2servs.items()))
# filtered_tag2servs=list(filter(lambda x:len(filt(x[1]))>5,tag2servs.items()))
# _tag2servs = dict(filtered_tag2servs)
df['main_ids'] = main_ids
df.to_csv('./filtered_datas.csv')
pickle.dump(tag2servs, open('./filtered_tag2serv', 'wb'))
# %%
