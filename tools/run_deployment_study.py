"""Run finite, interleaved installed-CLI comparisons; retain commands and logs."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--suite', choices=['smoke', 'performance', 'accuracy'], required=True)
    parser.add_argument('--counts', type=int, nargs='+')
    args = parser.parse_args()
    study = args.study.resolve()
    tools = Path(__file__).resolve().parent
    image = subprocess.check_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    records = []
    counts = args.counts or ([1, 5] if args.suite == 'smoke' else ([150] if args.suite == 'accuracy' else [100, 1000]))
    for count in counts:
        for model in ('base', 'plddt'):
            for rep in ([1] if args.suite == 'smoke' else [1, 2]):
                order = ['eager', 'aot'] if (model == 'base') == (rep == 1) else ['aot', 'eager']
                for execution in order:
                    dest = study / args.suite / f'n{count}' / model / execution / f'run{rep}'
                    if dest.exists():
                        if (dest / 'metrics_261026.json').exists():
                            raise FileExistsError(f'Completed result already exists: {dest}')
                        attempt = 2
                        while dest.with_name(f'run{rep}_attempt{attempt}').exists():
                            attempt += 1
                        dest = dest.with_name(f'run{rep}_attempt{attempt}')
                    dest.mkdir(parents=True, exist_ok=False)
                    cmd = ['docker', 'run', '--rm', '--network', 'none', '--gpus', 'all', '--ipc=host',
                           '--user', '1000:1000', '-e', 'HOME=/tmp', '-e', 'PYTHONHASHSEED=0',
                           '-e', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
                           '-v', f'{tools}:/measurement:ro', '-v', f'{study}/inputs:/inputs:ro',
                           '-v', f'{dest}:/results', image, 'python', '-u', '/measurement/measure_deployment.py',
                           '--input', f'/inputs/antibodies_{count}.csv', '--output', '/results',
                           '--model', model, '--execution', execution, '--precision', 'strict',
                           '--writer', 'native', '--batch-size', 'auto',
                           '--capture-limit', str(min(count, 250))]
                    # The smoke control includes the unchanged Python writer;
                    # performance also records it as the previous dev output path.
                    if execution == 'eager':
                        cmd[cmd.index('--writer') + 1] = 'python'
                    (dest / 'command_261026.json').write_text(json.dumps(cmd, indent=2) + '\n')
                    print('START', args.suite, count, model, execution, rep, flush=True)
                    started = time.perf_counter()
                    with (dest / 'inference.log').open('w') as output:
                        result = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT)
                    if result.returncode:
                        print((dest / 'inference.log').read_text()[-6000:], flush=True)
                        raise RuntimeError(f'Installed CLI failed: {dest}')
                    elapsed = time.perf_counter() - started
                    metrics = json.loads((dest / 'metrics_261026.json').read_text())
                    batches = json.loads((dest / 'batching.json').read_text())
                    if not metrics['completed'] or batches['saved'] != count or batches['oom_retries']:
                        raise ValueError(f'Incomplete or unexpected retry: {dest}')
                    if execution == 'aot' and (metrics['native_writer_calls'] != count
                            or batches['deployment']['aot_batches'] + batches['deployment']['eager_shape_fallback_batches'] != len(batches['batches'])
                            or metrics['dynamo_counters'].get('stats', {}).get('unique_graphs', 0)):
                        raise ValueError(f'AOT/native dispatch or no-JIT check failed: {dest}')
                    if execution == 'aot' and args.suite != 'accuracy' and batches['deployment']['eager_shape_fallback_batches']:
                        raise ValueError(f'Unexpected shape fallback: {dest}')
                    row = {'count': count, 'model': model, 'execution': execution, 'replicate': rep,
                           'image': image, 'container_seconds': elapsed,
                           'output': str(dest.relative_to(study)), 'metrics': metrics, 'batching': batches}
                    records.append(row)
                    (study / f'{args.suite}_measurements_261026.json').write_text(json.dumps(records, indent=2) + '\n')
                    print('DONE', args.suite, count, model, execution, rep, round(elapsed, 3), flush=True)


if __name__ == '__main__':
    main()
