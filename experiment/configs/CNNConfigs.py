import argparse
from .globalConfigs import CUDA

test_args = argparse.ArgumentParser('CNN')
test_args.add_argument('--cuda',type=bool,default=CUDA)
test_args.add_argument("--embedding_size", type=int, default=300)

test_args.add_argument("--dropout", type=float, default=0.2)
test_args.add_argument("--kernel_size", type=int, default=2)


test_args.add_argument("-f", type=str, default='')

args = test_args.parse_args()