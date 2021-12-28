DATA_DIR = r'./input_data'
import os

tag_path = os.path.join(DATA_DIR,'tag_servs.csv')
dict_path = os.path.join(DATA_DIR,'dict')
idf_path = os.path.join(DATA_DIR,'idfs')
vae_path = os.path.join(DATA_DIR,'vae')
query_ext = os.path.join(DATA_DIR,'query_ext')

d2v_path = os.path.join(r'./data','d2v')

pre_embedding_path = os.path.join(DATA_DIR,'pre_w2v.tar.gz')
embedding_path = os.path.join(DATA_DIR,'w2v')
pos_tag = os.path.join(DATA_DIR,'pos_tag2servs')
ori2syn = os.path.join(DATA_DIR,'ori2syn')

neg_tag = os.path.join(DATA_DIR,'neg_tag2servs')
train_pairs = os.path.join(DATA_DIR,'train_pairs')
test_pairs = os.path.join(DATA_DIR,'test_pairs')
syn2raw = os.path.join(DATA_DIR,'ori2raw')