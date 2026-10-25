"""Verify a capture against the retained original runtime manifest."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--modernised', action='store_true')
    args = p.parse_args()
    a = json.loads(args.reference.read_text())
    b = json.loads(args.candidate.read_text())
    source_keys = [x for x in a['source_hashes'] if x.startswith('src/')] if args.modernised else list(a['source_hashes'])
    errors = [f'source mismatch: {x}' for x in source_keys if a['source_hashes'][x] != b['source_hashes'].get(x)]
    if a['weights'] != b['weights']:
        errors.append('checkpoint identity mismatch')
    if not args.modernised:
        for key in ['versions', 'openmm_full_version', 'openmm_git_revision', 'reduce_binary_sha256', 'reduce_dictionary_sha256']:
            if a[key] != b[key]:
                errors.append(f'{key} mismatch')
    print(json.dumps({'passed': not errors, 'source_files': len(source_keys),
                      'checkpoints': len(a['weights']), 'errors': errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
