import argparse
from .globalConfigs import CUDA

test_args = argparse.ArgumentParser('BERT')
test_args.add_argument('--cuda',type=bool,default=CUDA)
# test_args.add_argument("--embedding_size", type=int, default=500)
test_args.add_argument("--dropout", type=float, default=0.2)
test_args.add_argument("--hidden_size", type=int, default=64)
test_args.add_argument("--input_size", type=int, default=64)
test_args.add_argument("--topic_size", type=int, default=200)
test_args.add_argument("--bidirectional", type=bool, default=True)

test_args.add_argument("-f", type=str, default='')

args = test_args.parse_args()