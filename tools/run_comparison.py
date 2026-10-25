"""Run isolated, sequential deployment benchmarks without deleting containers."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

IMAGES = {
    'original': 'abodybuilder3-spark:latest',
    'rebuilt': 'abodybuilder3-spark:rebuild-251026',
    'modernised': 'abodybuilder3-official-spark:modernised-251026',
    'dev': 'abodybuilder3-spark:latest',
}


def execute(args, condition, model, count, mode='sequential', batch=3):
    label = f'{condition}_{model}_n{count}_{mode}_b{batch}'
    dest = args.output / label
    dest.mkdir(parents=True, exist_ok=False)
    name = 'abb3-' + label.replace('_', '-') + '-' + args.study
    image = subprocess.check_output(['docker', 'image', 'inspect', IMAGES[condition], '--format', '{{.Id}}'], text=True).strip()
    cmd = ['docker', 'run', '--name', name, '--network', 'none', '--gpus', 'all', '--ipc=host',
           '-e', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True', '-e', 'PYTHONHASHSEED=0',
           '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'HF_HOME=/tmp/hf',
           '-v', f'{args.tools}:/measurement:ro', '-v', f'{args.frozen}:/frozen:ro',
           '-v', f'{dest}:/output', '-v', f'{args.prott5}:/prott5:ro']
    if condition == 'dev':
        cmd += ['-v', f'{args.dev_source}:/dev_source:ro', '-e', 'PYTHONPATH=/dev_source/src']
    cmd += [image, 'python', '/measurement/benchmark_runtime.py', '--model', model,
            '--mode', mode, '--count', str(count), '--batch-size', str(batch), '--output', '/output/predictions']
    (dest / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
    start = time.perf_counter()
    with (dest / 'inference.log').open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    row = {'condition': condition, 'model': model, 'count': count, 'mode': mode,
           'batch_size': batch, 'image': image, 'container': name,
           'complete_process_seconds': time.perf_counter() - start, 'exit_code': result.returncode,
           'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (dest / 'process.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row), flush=True)
    if result.returncode:
        raise RuntimeError(f'Failed: {dest / "inference.log"}')
    metrics = json.loads((dest / 'predictions/measurements.json').read_text())
    if not metrics['completed']:
        raise RuntimeError(f'Incomplete measurement: {dest}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--conditions', nargs='+', choices=IMAGES, default=['original', 'rebuilt', 'modernised'])
    p.add_argument('--models', nargs='+', choices=['base', 'plddt', 'language'], default=['base', 'plddt', 'language'])
    p.add_argument('--count', type=int, default=5)
    p.add_argument('--mode', choices=['sequential', 'dev_batch', 'dev_language_adapter'], default='sequential')
    p.add_argument('--batch-size', type=int, default=3)
    p.add_argument('--study', default='251026')
    p.add_argument('--tools', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--prott5', type=Path, required=True)
    p.add_argument('--dev-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    for attr in ['tools', 'frozen', 'prott5', 'dev_source', 'output']:
        setattr(args, attr, getattr(args, attr).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for condition in args.conditions:
            execute(args, condition, model, args.count, args.mode, args.batch_size)


if __name__ == '__main__':
    main()
