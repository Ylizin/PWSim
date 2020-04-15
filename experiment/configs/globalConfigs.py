embedding_size = 300
di_size = 13681
total_idx_len = 3888
test_idx_len = 1167 # in test_pos, there are also 6229 diagonal pairs, remember when cal metrics, subtract 1 for Numerator and Denominator 
train_idx_len = 2721
import torch
CUDA = torch.cuda.is_available()

import argparse
glob_args = argparse.ArgumentParser('LSTM')
glob_args.add_argument("--batch_size", type=int, default=128)
glob_args.add_argument("--lr", type=float, default=3e-4)
glob_args.add_argument("--weight_decay", type=float, default=1e-5)
glob_args.add_argument("--cuda", type=bool, default=CUDA)

glob_args.add_argument("--nepochs", type=int, default=1)
glob_args.add_argument("--topk", type=int, default=10)


args = glob_args.parse_args()
