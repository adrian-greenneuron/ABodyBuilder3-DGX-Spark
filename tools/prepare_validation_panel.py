"""Validate the separately selected published validation split and make CLI input.

The 150 upstream validation targets were not used in implementation tuning.
They are not an unseen model test: the model authors used this validation split.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads((args.data / 'selection_261026.json').read_text())
    truth = args.data / 'truth'
    truth.mkdir(exist_ok=False)
    records = []
    with args.csv.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'heavy', 'light', 'target_id'])
        writer.writeheader()
        for name in selection['targets']:
            path = args.data / (name + '.pt')
            if hashlib.sha256(path.read_bytes()).hexdigest() != selection['manifest'][name]['sha256']:
                raise ValueError(f'Validation tensor checksum differs: {name}')
            # The archive is the checksum-verified published tensor release.
            data = torch.load(path, map_location='cpu', weights_only=False)
            sequence = data['sequence']
            h, n = int(data['is_heavy'].sum()), len(data['aatype'])
            expected_heavy = torch.tensor([1] * h + [0] * (n - h))
            if (data['structure'] != name or len(sequence) != n or not 0 < h < n
                    or not torch.equal(data['is_heavy'].long(), expected_heavy)
                    or not torch.equal(data['residue_index'], torch.cat((torch.arange(h), torch.arange(n - h) + 500)))):
                raise ValueError(f'Validation chain/sequence/index contract failed: {name}')
            pos = data['all_atom_positions'].numpy()
            mask = data['all_atom_mask'].numpy().astype(bool)
            if not np.isfinite(pos[mask]).all():
                raise ValueError(f'Nonfinite observed truth atoms: {name}')
            regions = np.asarray(data['region'])
            np.savez(truth / (name + '.npz'), positions=pos, masks=mask,
                     regions=regions, heavy=data['is_heavy'].numpy().astype(bool),
                     aatype=data['aatype'].numpy())
            writer.writerow(dict(name=name, heavy=sequence[:h], light=sequence[h:], target_id=name))
            records.append({'name': name, 'length': n, 'heavy_length': h,
                            'cdrh3_length': int((regions == 'cdrh3').sum()),
                            'observed_atoms': int(mask.sum()),
                            'tensor_keys': sorted(data)})
    report = {'completed': True, 'targets': len(records),
              'distinct_pdb_entries': len({row['name'].split('_')[0] for row in records}),
              'scope': selection['scope'], 'targets_metadata': records,
              'csv_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest()}
    (args.data / 'validation_261026.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'targets_metadata'}), flush=True)


if __name__ == '__main__':
    main()
