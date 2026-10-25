"""Fetch only the pinned public ProtT5 inference assets, not alternate checkpoints."""
import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
args = parser.parse_args()
snapshot_download('Rostlab/prot_t5_xl_uniref50',
                  revision='973be27c52ee6474de9c945952a8008aeb2a1a73',
                  local_dir=args.output,
                  allow_patterns=['config.json', 'pytorch_model.bin', 'spiece.model',
                                  'tokenizer_config.json', 'special_tokens_map.json'])
