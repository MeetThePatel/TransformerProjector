import argparse
import safetensors.torch as stt
from torchtune.models.convert_weights import _FROM_HF
import re


def create_mapping_dict():
    # Create regex patterns for matching
    patterns = {}
    for src, tgt in _FROM_HF.items():
        if '{}' in src:
            # Convert the format string to regex pattern
            pattern = src.replace('.', r'\.').replace('{}', r'(\d+)')
            if tgt is not None:
                tgt = tgt.replace('{}', r'\1')  # Use the captured group
        else:
            pattern = src.replace('.', r'\.')
        patterns[pattern] = tgt

    return patterns


def convert_weight_names(weight_names):
    """Convert weight names from source format to target format."""
    patterns = create_mapping_dict()
    converted = {}

    for name in weight_names:
        matched = False
        for pattern, target in patterns.items():
            if re.match(f"^{pattern}$", name):
                if target is not None:
                    new_name = re.sub(f"^{pattern}$", target, name)
                    converted[name] = new_name
                matched = True
                break

        if not matched:
            converted[name] = None

    return converted


def rename_weights(input_file: str, output_file: str):
    _FROM_HF['model.layers.{}.self_attn.q_proj.bias'] = 'layers.{}.attn.q_proj.bias'
    _FROM_HF['model.layers.{}.self_attn.k_proj.bias'] = 'layers.{}.attn.k_proj.bias'
    _FROM_HF['model.layers.{}.self_attn.v_proj.bias'] = 'layers.{}.attn.v_proj.bias'

    input_weights = stt.load_file(input_file)
    mapping = convert_weight_names(input_weights.keys())

    renamed_weights_dict = {}
    for key, value in input_weights.items():
        new_key = mapping[key]
        renamed_weights_dict[new_key] = value

    stt.save_file(renamed_weights_dict, output_file)
    print(f"Renamed weight tensors to {output_file}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert from HuggingFace weight convention to TorchTune convention.")
    parser.add_argument('-i', '--inputs', required=True, help="Input weights file.")
    parser.add_argument('-o', '--output', help='Name of output file.', default="renamed_weights.safetensors")
    args = parser.parse_args()

    rename_weights(args.inputs, args.output)
