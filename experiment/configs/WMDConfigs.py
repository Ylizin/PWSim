import argparse
from data_utils.paths import pre_embedding_path

test_args = argparse.ArgumentParser('WMD')
test_args.add_argument("--pret_embeddings", type=str, default=pre_embedding_path)
args = test_args.parse_args()