from data_utils.paths import embedding_path
from .globalConfigs import di_size,CUDA

import argparse
test_args = argparse.ArgumentParser('Embedding')
test_args.add_argument("--embedding_size", type=int, default=300)
test_args.add_argument("--pret_embeddings", type=str, default=embedding_path)
test_args.add_argument("--vocab_size", type=int, default=di_size)
test_args.add_argument('--cuda',type=bool,default=CUDA)


args = test_args.parse_args()