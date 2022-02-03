import argparse
from .globalConfigs import CUDA,di_size
from data_utils.paths import embedding_path,idf_path,vae_path

test_args = argparse.ArgumentParser('VAE')
test_args.add_argument('--cuda',type=bool,default=CUDA)
test_args.add_argument("--embedding_size", type=int, default=300)
test_args.add_argument("--vocab_size", type=int, default=di_size)
test_args.add_argument("--dropout", type=float, default=0.0)
test_args.add_argument("--topic_size", type=int, default=300) # 200 dim 
test_args.add_argument("--idf_path", type=str, default=idf_path)
test_args.add_argument("--vae_path", type=str, default=vae_path)

test_args.add_argument("-f", type=str, default='')

args = test_args.parse_args()