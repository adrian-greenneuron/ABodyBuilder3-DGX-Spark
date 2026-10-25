"""Prepare the public repeat workload and freeze the CLI used for measurements."""
import argparse
import csv
import json
import hashlib
import random
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--counts', type=int, nargs='+', default=[5, 20, 100, 1000])
parser.add_argument('--synthetic-unique', action='store_true', help='Deterministic distinct substitutions; implementation stress inputs, not biological accuracy data.')
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
fixtures = json.loads((repo / 'tests/benchmark_antibodies.json').read_text())
args.output.mkdir(parents=True, exist_ok=False)
(args.output / 'antibodies.json').write_text(json.dumps(fixtures, indent=2) + '\n')
shutil.copy2(repo / 'scripts/predict.py', args.output / 'predict_main.py')
shutil.copy2(repo / 'scripts/structure_preparation.py', args.output / 'structure_preparation.py')
shutil.copy2(repo / 'scripts/batch_pipeline.py', args.output / 'batch_pipeline.py')
for count in args.counts:
    if count < 1:
        raise ValueError('Count must be positive')
    seen = set()
    rng = random.Random(251026)
    with (args.output / f'antibodies_{count}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'heavy', 'light'])
        writer.writeheader()
        for i in range(count):
            row = fixtures[i % len(fixtures)]
            heavy, light = row['heavy'], row['light']
            if args.synthetic_unique:
                while True:
                    h, l = list(row['heavy']), list(row['light'])
                    for chain in (h,l):
                        for p in rng.sample([j for j,a in enumerate(chain) if a != 'C'],2):
                            chain[p] = rng.choice([a for a in 'ADEFGHIKLMNPQRSTVWY' if a != chain[p]])
                    heavy, light = ''.join(h), ''.join(l)
                    if (heavy,light) not in seen:break
            seen.add((heavy,light))
            writer.writerow({'name': f'{i:04d}_{row["name"]}', 'heavy': heavy, 'light': light})
    path=args.output / f'antibodies_{count}.csv'
    (args.output/f'inputs_{count}_251026.json').write_text(json.dumps({
        'count':count,'unique_sequence_pairs':len(seen),'synthetic_substitutions':args.synthetic_unique,
        'seed':251026 if args.synthetic_unique else None,
        'purpose':'implementation/throughput stress; not an experimental-structure accuracy dataset',
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+'\n')
