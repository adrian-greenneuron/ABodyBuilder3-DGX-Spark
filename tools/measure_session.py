"""Compare consecutive real CLI requests with optional process-local reuse."""
import time
START = time.perf_counter()
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import threading

import torch
from benchmark_runtime import Memory
from prediction_fingerprint import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['base', 'plddt'], required=True)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--resident', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, '/opt/abodybuilder3/scripts')
    spec = importlib.util.spec_from_file_location('session_cli', '/opt/abodybuilder3/scripts/predict.py')
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    from abodybuilder3 import aot_runtime
    session = None
    if args.resident:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'candidate'))
        from resident_session import ResidentSession
        session = ResidentSession(cli, aot_runtime); session.install()
    import batch_pipeline
    original_pipeline = batch_pipeline.run_pipeline
    def pipeline(batches, prepare, infer, write, **options):
        def prediction(items, prepared):
            result = infer(items, prepared)
            for (name, _, _), _, inp in result:
                inp['_session_name'] = name
            return result
        return original_pipeline(batches, prepare, prediction, write, **options)
    batch_pipeline.run_pipeline = pipeline
    original_pdb = cli.generate_pdb
    fingerprints = []
    lock = threading.Lock()
    def pdb(output, inp):
        row = {'name': inp['_session_name'], **fingerprint(output, inp)}
        with lock: fingerprints.append(row)
        return original_pdb(output, inp)
    cli.generate_pdb = pdb
    rows = list(csv.DictReader(args.input.open()))
    expected = sorted(row['name'] + '.pdb' for row in rows)
    measurements = []
    with Memory() as memory:
        for job in range(args.jobs):
            fingerprints.clear()
            out = args.output / f'job{job}'; (out / 'pdbs').mkdir(parents=True)
            sys.argv = ['predict.py', 'predict', '--csv', str(args.input), '--output', str(out / 'pdbs'),
                        '--model', args.model, '--workers', '1', '--seed', '42', '--quiet',
                        '--batch-report', str(out / 'batching.json')]
            started = time.perf_counter()
            try:
                cli.app()
            except SystemExit as error:
                if error.code not in (0, None): raise
            duration = time.perf_counter() - started
            if len(fingerprints) != len(rows) or sorted(p.name for p in (out / 'pdbs').glob('*.pdb')) != expected:
                raise ValueError('Resident request lost predictions or output identities.')
            measurements.append({'job': job, 'cli_seconds': duration,
                                 'deployment': json.loads((out / 'batching.json').read_text())['deployment']})
            (out / 'fingerprints_261026.json').write_text(json.dumps(sorted(fingerprints, key=lambda row: row['name']), indent=2) + '\n')
            print('JOB_COMPLETE', job, duration, flush=True)
        finished = time.perf_counter()
    if session:
        hits = dict(session.hits); session.restore()
    else:
        hits = None
    result = {'completed': True, 'model': args.model, 'jobs': args.jobs, 'inputs_per_job': len(rows),
              'resident': args.resident, 'script_to_complete_seconds': finished - START,
              'requests': measurements, 'cache_counts': hits,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), **memory.result()}
    (args.output / 'metrics_261026.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
