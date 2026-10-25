"""Measure the unchanged eight-worker CLI, checking every expected PDB output."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

from run_comparison import IMAGES


def cuda_summary(folder):
    events = []
    peaks = []
    for path in folder.glob('*.json'):
        peaks.append(json.loads(path.read_text()))
    for path in folder.glob('*.csv'):
        with path.open() as f:
            for r in csv.DictReader(f):
                events.append((float(r['monotonic_seconds']), path.stem,
                               int(r['cuda_allocated_bytes']), int(r['cuda_reserved_bytes'])))
    current = {}
    allocated = reserved = 0
    for t, pid, a, r in sorted(events):
        current[pid] = (a, r)
        allocated = max(allocated, sum(x[0] for x in current.values()))
        reserved = max(reserved, sum(x[1] for x in current.values()))
    return {'sampled_concurrent_cuda_allocated_peak_bytes': allocated,
            'sampled_concurrent_cuda_reserved_peak_bytes': reserved,
            'per_process_cuda_peaks': peaks}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition', choices=['original', 'rebuilt', 'modernised'], required=True)
    p.add_argument('--model', choices=['base', 'plddt', 'language'], required=True)
    p.add_argument('--image', help='Override the retained runtime image for a matched comparison.')
    p.add_argument('--label', help='Unique container-name label for a new comparison.')
    p.add_argument('--replicate', type=int, choices=[0, 1, 2], help='Run one replicate, permitting interleaved comparisons.')
    p.add_argument('--prott5', type=Path, help='Pinned offline ProtT5 snapshot for language CLI runs.')
    p.add_argument('--batch-size', help='Optional dev batch size; omitted for historical/main CLIs.')
    p.add_argument('--cpu-workers', type=int, default=2)
    p.add_argument('--inference-mode',choices=['lookup','lookup-final','optimised'],help='Measurement-only ablation of the new dev inference path.')
    p.add_argument('--fingerprint',action='store_true',help='Hash every raw coordinate/mask/confidence output for strict agreement checks.')
    p.add_argument('--count', type=int, default=1000)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.frozen = args.frozen.resolve()
    args.output = args.output.resolve()
    tools = Path(__file__).resolve().parent
    image = subprocess.check_output(['docker', 'image', 'inspect', args.image or IMAGES[args.condition], '--format', '{{.Id}}'], text=True).strip()
    records_path = args.output / 'measurements.json'
    rows = json.loads(records_path.read_text()) if records_path.exists() else []
    for rep in ([args.replicate] if args.replicate is not None else range(3)):
        dest = args.output / f'run{rep}'
        dest.mkdir(parents=True, exist_ok=False)
        name = f'abb3-cli-{args.label or args.condition}-{args.model}-n{args.count}-w{args.workers}-r{rep}-251026'
        cmd = ['docker', 'run', '--name', name, '--network', 'none', '--gpus', 'all', '--ipc=host',
               '-e', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True', '-e', 'PYTHONHASHSEED=0',
               '-e', 'PYTHONPATH=/measurement/instrumentation:/measurement', '-e', 'ABB3_RESOURCE_DIR=/output/resources',
               '-v', f'{tools}:/measurement:ro', '-v', f'{args.frozen}:/frozen:ro', '-v', f'{dest}:/output',
               image, 'python', '/measurement/measure_cli.py', 'predict',
               '--csv', f'/frozen/antibodies_{args.count}.csv', '--output', '/output/pdbs/',
               '--model', args.model, '--workers', str(args.workers), '--seed', '42', '--quiet']
        insertion=cmd.index(image)
        extra=[]
        if args.inference_mode:
            extra += ['-e',f'ABB3_BENCHMARK_INFERENCE_MODE={args.inference_mode}']
        if args.fingerprint:
            extra += ['-e','ABB3_FINGERPRINT_PATH=/output/prediction_fingerprints.json']
        cmd[insertion:insertion]=extra
        if args.batch_size is not None:
            cmd += ['--batch-size', args.batch_size, '--cpu-workers', str(args.cpu_workers),
                    '--batch-report', '/output/batching.json']
        if args.prott5:
            # Match the CLI's default relative model directory without patching
            # its loading code or downloading weights during measurement.
            index = cmd.index(image)
            cmd[index:index] = ['-v', f'{args.prott5.resolve()}:/opt/abodybuilder3/Rostlab/prot_t5_xl_uniref50:ro', '-e', 'HF_HOME=/tmp/hf',
                                '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1']
        (dest / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
        start = time.perf_counter()
        with (dest / 'inference.log').open('w') as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        wall = time.perf_counter() - start
        if r.returncode:
            raise RuntimeError(f'CLI failed: {dest}')
        expected = []
        with (args.frozen / f'antibodies_{args.count}.csv').open() as f:
            expected = [x['name'] + '.pdb' for x in csv.DictReader(f)]
        actual = sorted(x.name for x in (dest / 'pdbs').glob('*.pdb'))
        if actual != sorted(expected):
            raise ValueError(f'Missing/extra predictions: {dest}')
        fingerprints=json.loads((dest/'prediction_fingerprints.json').read_text()) if args.fingerprint else None
        if fingerprints is not None and len(fingerprints)!=args.count:
            raise ValueError('Missing raw prediction fingerprints')
        row = {'label': args.label or args.condition, 'batch_size': args.batch_size, 'cpu_workers': args.cpu_workers,
               'inference_mode':args.inference_mode,'raw_predictions_fingerprinted':len(fingerprints) if fingerprints is not None else None,
               'condition': args.condition, 'model': args.model, 'count': args.count,
               'workers': args.workers, 'replicate': rep, 'warmup': rep == 0,
               'complete_process_seconds': wall, 'antibodies_per_minute': 60 * args.count / wall,
               'image': image, 'container': name,
               **json.loads((dest / 'cli_metrics.json').read_text()),
               **cuda_summary(dest / 'resources')}
        if not row['completed'] or row['sampled_concurrent_cuda_allocated_peak_bytes'] <= 0:
            raise ValueError('Incomplete CLI resource measurements')
        rows.append(row)
        (args.output / 'measurements.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
