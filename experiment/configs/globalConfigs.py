embedding_size = 300
di_size = 18000
total_idx_len = 3888
test_idx_len = 1167 
train_idx_len = 2721
import torch
CUDA = torch.cuda.is_available()
from data_utils.paths import embedding_path,idf_path,vae_path

import argparse
glob_args = argparse.ArgumentParser('glob')
glob_args.add_argument("--batch_size", type=int, default=512)
glob_args.add_argument("--bert_batch_size", type=int, default=8)

glob_args.add_argument("--bertLr", type=float, default=5e-6)
glob_args.add_argument("--lr", type=float, default=3e-3)

glob_args.add_argument("--weight_decay", type=float, default=0)
glob_args.add_argument('--cuda',type=bool,default=CUDA)
glob_args.add_argument("--vae_path", type=str, default=vae_path)
glob_args.add_argument('--use_ext_query',type=bool,default=True)
glob_args.add_argument("--nepochs", type=int, default=30)
glob_args.add_argument("--topk", type=int, default=20)
glob_args.add_argument("-f", type=str, default='')


args = glob_args.parse_args()
