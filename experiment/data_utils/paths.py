DATA_DIR = r'./data'
import os

eda_path = os.path.join(DATA_DIR,'eda_dataset.csv')
dict_path = os.path.join(DATA_DIR,'dict')
pre_embedding_path = os.path.join(DATA_DIR,'pre_w2v.tar.gz')
embedding_path = os.path.join(DATA_DIR,'w2v')
train_pos_path = os.path.join(DATA_DIR,'train_pos.npy')
train_neg_path = os.path.join(DATA_DIR,'train_neg.npy')
test_pos_path = os.path.join(DATA_DIR,'test_pos.npy')