"""Run bounded, paired reference/deferred jobs in fresh unchanged images."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--component', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--repetitions', type=int, default=2)
    args = parser.parse_args()
    root, component, inputs = args.study.resolve(), args.component.resolve(), args.inputs.resolve()
    rows = []
    expected = sorted(row['name'] + '.pdb' for row in csv.DictReader((inputs / f'antibodies_{args.count}.csv').open()))
    for model in ('base', 'plddt'):
        for rep in range(1, args.repetitions + 1):
            order = ('reference', 'deferred') if (model == 'base') == (rep % 2 == 1) else ('deferred', 'reference')
            folders = {}
            for mode in order:
                out = root / f'n{args.count}' / model / mode / f'run{rep}'
                out.mkdir(parents=True, exist_ok=False); folders[mode] = out
                cmd = ['docker', 'run', '--rm', '--network', 'none', '--gpus', 'all', '--ipc=host',
                       '--user', '1000:1000', '-e', 'HOME=/tmp', '-e', 'PYTHONHASHSEED=0',
                       '-v', f'{component}:/component:ro', '-v', f'{inputs}:/inputs:ro',
                       '-v', f'{out}:/results', args.image, 'python', '-u', '/component/tools/measure_pipeline.py',
                       '--mode', mode, '--input', f'/inputs/antibodies_{args.count}.csv',
                       '--output', '/results', '--model', model]
                (out / 'command_261026.json').write_text(json.dumps(cmd, indent=2) + '\n')
                print('START', args.count, model, mode, rep, flush=True)
                start = time.perf_counter()
                with (out / 'inference.log').open('w') as output:
                    result = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT)
                if result.returncode:
                    print((out / 'inference.log').read_text()[-6000:], flush=True)
                    raise RuntimeError(f'Comparison failed: {out}')
                elapsed = time.perf_counter() - start
                metrics = json.loads((out / 'metrics_261026.json').read_text())
                if sorted(path.name for path in (out / 'pdbs').glob('*.pdb')) != expected:
                    raise ValueError('Output names/counts differ.')
                rows.append({'count': args.count, 'model': model, 'mode': mode, 'replicate': rep,
                             'container_seconds': elapsed, 'metrics': metrics, 'output': str(out.relative_to(root))})
                (root / f'measurements_n{args.count}_261026.json').write_text(json.dumps(rows, indent=2) + '\n')
                print('DONE', args.count, model, mode, rep, round(elapsed, 3), flush=True)
            left, right = folders['reference'], folders['deferred']
            if json.loads((left / 'fingerprints_261026.json').read_text()) != json.loads((right / 'fingerprints_261026.json').read_text()):
                raise ValueError(f'Raw fingerprints differ: {model} repetition {rep}')
            for name in expected:
                if (left / 'pdbs' / name).read_bytes() != (right / 'pdbs' / name).read_bytes():
                    raise ValueError(f'PDB differs: {model} {name}')
            print('EXACT_RAW_AND_PDB', model, rep, args.count, flush=True)


if __name__ == '__main__':
    main()
