import argparse
from typing import List

import safetensors.torch as stt


def merge_safetensors(input_files: List[str], output_file: str):
    merged_tensors = {}
    for file in input_files:
        merged_tensors |= stt.load_file(file)
    stt.save_file(merged_tensors, output_file)
    print(f"Saved merged tensors to {output_file}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge multiple safetensors files into a single safetensor file.")
    parser.add_argument("-i", "--inputs", nargs="+", required=True, help="List of input files.")
    parser.add_argument("-o", "--output", required=False, help="Name of output file.", default="merged_model.safetensors")
    args = parser.parse_args()

    merge_safetensors(args.inputs, args.output)
