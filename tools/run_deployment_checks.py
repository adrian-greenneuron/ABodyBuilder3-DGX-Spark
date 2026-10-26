"""Finite failure, sustained-job and optional-path deployment qualification."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--prott5', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.study.resolve()
    repo = Path(__file__).resolve().parents[1]
    records = json.loads((root / 'deployment_checks_261026.json').read_text()) if args.resume else []

    def run(label, cmd):
        previous = [row for row in records if row['label'] == label]
        if args.resume and previous and previous[-1]['exit_code'] == 0:
            if json.loads((root / f'{label}_command_261026.json').read_text()) != cmd:
                raise ValueError(f'Cannot reuse changed command: {label}')
            print('REUSE', label, flush=True)
            return
        print('START', label, flush=True)
        if previous:
            attempt = len(previous)
            for suffix in ('_command_261026.json', '_261026.log'):
                source = root / (label + suffix)
                shutil.copy2(source, root / (label + f'_attempt{attempt}' + suffix))
        (root / f'{label}_command_261026.json').write_text(json.dumps(cmd, indent=2) + '\n')
        started = time.perf_counter()
        with (root / f'{label}_261026.log').open('w') as output:
            result = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT)
        row = {'label': label, 'exit_code': result.returncode, 'seconds': time.perf_counter() - started}
        records.append(row)
        (root / 'deployment_checks_261026.json').write_text(json.dumps(records, indent=2) + '\n')
        if result.returncode:
            print((root / f'{label}_261026.log').read_text()[-6000:], flush=True)
            raise RuntimeError(f'Qualification failed: {label}')
        print('DONE', label, round(row['seconds'], 3), flush=True)

    common = ['docker', 'run', '--rm', '--network', 'none', '--gpus', 'all', '--ipc=host',
              '--user', '1000:1000', '-e', 'HOME=/tmp',
              '-e', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
              '-v', f'{repo}/tests:/tests:ro', '-v', f'{repo}/tools:/measurement:ro',
              '-v', f'{root}:/results']
    run('optional_preparation_and_missing_package', common + [args.image, 'python', '-u',
        '/tests/check_deployment_failures.py', '/results/optional_compatibility_261026.json'])
    run('interruption_controller', ['python3', str(repo / 'tools/check_job_interruption.py'),
        '--study', str(root), '--image', args.image])
    (root / 'language_logs').mkdir(exist_ok=True)
    language = common + ['-v', f'{root}/language_logs:/opt/abodybuilder3/lightning_logs',
        '-v', f'{args.prott5.resolve()}:/opt/abodybuilder3/Rostlab/prot_t5_xl_uniref50:ro',
        '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1', args.image,
        'python', 'scripts/predict.py', 'predict', '--csv', '/results/inputs/antibodies_1.csv',
        '--output', '/results/language_compatibility.pdb', '--model', 'language', '--workers', '1',
        '--batch-size', '1', '--precision', 'strict', '--seed', '42', '--quiet']
    run('language_compatibility', language)
    text = (root / 'language_compatibility.pdb').read_text()
    if not text.endswith('END\n') or not text.startswith('PARENT N/A'):
        raise ValueError('Language CLI did not produce a complete PDB.')
    (root / 'sustained').mkdir(exist_ok=False)
    run('sustained', common + [args.image, 'python', '-u', '/measurement/measure_deployment.py',
        '--input', '/results/inputs/antibodies_20000.csv', '--output', '/results/sustained',
        '--model', 'plddt', '--execution', 'auto', '--precision', 'strict', '--writer', 'native',
        '--capture-limit', '0', '--memory-trace'])
    metrics = json.loads((root / 'sustained/metrics_261026.json').read_text())
    batching = json.loads((root / 'sustained/batching.json').read_text())
    if (not metrics['completed'] or not batching['completed'] or batching['saved'] != 20000
            or metrics['native_writer_calls'] != 20000 or batching['oom_retries']):
        raise ValueError('Sustained output/resource accounting failed.')
    print('ALL_CHECKS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
