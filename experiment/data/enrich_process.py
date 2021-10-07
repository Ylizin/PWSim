import pandas as pd
import spacy as sp
from eda import lem

vec_model= sp.load('en_core_web_lg')
df = pd.read_csv(r'./base_data.csv')
parse_list=lambda s:eval(s)
des = df.Description
cats = df.Categories.apply(parse_list)
from nltk import pos_tag_sents
pos_tags = pos_tag_sents(des.apply(lem))

nn_words = []
def dup_nn(li):
    res = ''
    nns = set()
    for w,att in li:
        if att.startswith('NN') or att.startswith('JJ'):
            res += w+' '
            nns.add(w)
        # res += w+' '
    nn_words.append(nns)
    return res
nn_sents = pd.Series([dup_nn(sent) for sent in pos_tags])

nn_docs = cats.apply(lambda x:' '.join(set(x))).apply(vec_model)
def get_most_sim(sent_doc,nns,topn = 3):
    words = []
    if not sent_doc:
        return []
    if len(sent_doc)<2:
        return []
    # words = [ (w,max([tag.similarity(vec_model.vocab[w]) for tag in sent_doc])) for w in nns]
    for w in nns:
        ve = vec_model.vocab[w]
        sim = sent_doc.similarity(ve)
        words.append((w,sim))
    # words.sort(key= lambda x:x[1])
    return words
nn_en = pd.Series([[(t,)*round(5*wei) for t,wei in get_most_sim(doc,nns)] for doc,nns in zip(nn_docs,nn_words)])

nn_en = nn_en.apply(lambda x: ' '.join([' '.join(en) for en in x]))
nn_sents = nn_sents+' '+nn_en
df['internal_enrich'] = nn_sents
nn_vecs = nn_sents.apply(vec_model)
rel_cats = [[(t,d.similarity(vec_model.vocab[t])) for t in tags] for d,tags in zip(nn_vecs,cats)]
df['internal_cats']= [set(x[0] for x in list(filter(lambda x:x[1]>0.5,li))) for li in rel_cats]
df.to_csv('processed.csv',index=None)