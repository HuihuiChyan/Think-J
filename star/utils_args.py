import argparse

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/oss_bucket_0/huanghui/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="cot-judge-prompt"
    )
    parser.add_argument(
        "--critique-filt",
        type=str,
        choices=("False", "True"),
        default="False"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42
    )
    return parser