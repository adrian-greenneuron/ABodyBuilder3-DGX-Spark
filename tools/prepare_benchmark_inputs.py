"""Prepare the public repeat workload and freeze the CLI used for measurements."""
import argparse
import csv
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--counts', type=int, nargs='+', default=[5, 20, 100, 1000])
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
fixtures = json.loads((repo / 'tests/benchmark_antibodies.json').read_text())
args.output.mkdir(parents=True, exist_ok=False)
(args.output / 'antibodies.json').write_text(json.dumps(fixtures, indent=2) + '\n')
shutil.copy2(repo / 'scripts/predict.py', args.output / 'predict_main.py')
shutil.copy2(repo / 'scripts/structure_preparation.py', args.output / 'structure_preparation.py')
for count in args.counts:
    if count < 1:
        raise ValueError('Count must be positive')
    with (args.output / f'antibodies_{count}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'heavy', 'light'])
        writer.writeheader()
        for i in range(count):
            row = fixtures[i % len(fixtures)]
            writer.writerow({'name': f'{i:04d}_{row["name"]}', 'heavy': row['heavy'], 'light': row['light']})
