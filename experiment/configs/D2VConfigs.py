import argparse
from data_utils.paths import d2v_path

test_args = argparse.ArgumentParser('D2V')
test_args.add_argument("--d2v_path", type=str, default=d2v_path)
test_args.add_argument("--vector_size", type=int, default=1000)
test_args.add_argument("--eopchs", type=int, default=300)

test_args.add_argument("-f", type=str, default='')

args = test_args.parse_args()