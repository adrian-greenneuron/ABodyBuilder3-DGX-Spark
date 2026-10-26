"""Measure the installed CLI without changing its model, writer or scheduler.

Optional raw captures are saved after the timed CLI call. The small capture and
fingerprint overhead inside inference is shared by comparison conditions.
"""
import time
START = time.perf_counter()
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from benchmark_runtime import Memory
from prediction_fingerprint import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['base', 'plddt'], required=True)
    parser.add_argument('--execution', choices=['auto', 'aot', 'eager'], default='auto')
    parser.add_argument('--precision', choices=['strict', 'tf32'], default='strict')
    parser.add_argument('--writer', choices=['native', 'python'], default='native')
    parser.add_argument('--batch-size', default='auto')
    parser.add_argument('--capture-limit', type=int, default=100)
    parser.add_argument('--memory-trace', action='store_true')
    parser.add_argument('--legacy', action='store_true', help='Measure the retained pre-AOT CLI with explicit precision.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'pdbs').mkdir(exist_ok=False)
    rows = list(csv.DictReader(args.input.open()))
    by_name = {row['name']: row.get('target_id') or row['name'] for row in rows}
    sys.path.insert(0, '/opt/abodybuilder3/scripts')
    spec = importlib.util.spec_from_file_location('installed_deployment_cli', '/opt/abodybuilder3/scripts/predict.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    captured, fingerprints = {}, []

    import batch_pipeline
    original_pipeline = batch_pipeline.run_pipeline

    def measured_pipeline(batches, prepare, infer, write, **options):
        def measured_infer(items, prepared):
            records = infer(items, prepared)
            first_batch = not fingerprints
            for (name, _, _), output, inp in records:
                fingerprints.append({'name': name, 'source': by_name[name], **fingerprint(output, inp)})
                source = by_name[name]
                if source not in captured and len(captured) < args.capture_limit:
                    captured[source] = {
                        'positions': np.array(output['atom37'], copy=True),
                        'masks': np.array(output['atom37_atom_exists'], copy=True),
                        'confidence_logits': np.array(output.get('plddt', np.empty(0, np.float32)), copy=True),
                        'heavy': inp['is_heavy'].numpy().astype(bool).copy(),
                        'aatype': inp['aatype'].numpy().copy()}
            if first_batch:
                print(json.dumps({'event': 'first_batch_complete', 'count': len(records)}), flush=True)
            return records
        return original_pipeline(batches, prepare, measured_infer, write, **options)

    batch_pipeline.run_pipeline = measured_pipeline
    if args.legacy:
        torch.set_float32_matmul_precision('highest' if args.precision == 'strict' else 'high')
        torch.backends.cuda.matmul.allow_tf32 = args.precision != 'strict'
        torch.backends.cudnn.allow_tf32 = args.precision != 'strict'
    sys.argv = ['predict.py', 'predict', '--csv', str(args.input), '--output', str(args.output / 'pdbs'),
                '--model', args.model, '--workers', '1', '--seed', '42', '--quiet',
                '--batch-size', args.batch_size, '--batch-report', str(args.output / 'batching.json')]
    if not args.legacy:
        sys.argv += ['--execution', args.execution, '--precision', args.precision, '--pdb-writer', args.writer]
    with Memory() as memory:
        try:
            cli.app()
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
        cli_end = time.perf_counter()
    if len(fingerprints) != len(rows):
        raise ValueError(f'Expected {len(rows)} raw predictions, captured {len(fingerprints)}.')
    if sorted(path.name for path in (args.output / 'pdbs').glob('*.pdb')) != sorted(row['name'] + '.pdb' for row in rows):
        raise ValueError('Installed CLI output identities/counts differ from input.')
    raw = args.output / 'raw'
    raw.mkdir(exist_ok=True)
    for name, values in captured.items():
        np.savez(raw / (name + '.npz'), **values)
    if args.memory_trace:
        with (args.output / 'memory_samples_261026.csv').open('w') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['seconds_since_script_start', 'process_tree_rss_bytes'])
            writer.writerows((stamp - START, rss) for stamp, rss in memory.samples)
    try:
        from abodybuilder3 import _pdb_native
        native_calls = _pdb_native.calls()
    except ImportError:
        native_calls = 0
    from torch._dynamo.utils import counters
    result = {'completed': True, 'model': args.model, 'execution': args.execution,
              'precision': args.precision, 'writer': args.writer, 'legacy': args.legacy,
              'input_count': len(rows), 'captured_targets': len(captured),
              'script_to_cli_complete_seconds': cli_end - START,
              'evidence_writing_seconds': time.perf_counter() - cli_end,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
              'native_writer_calls': native_calls, 'dynamo_counters': {str(k): dict(v) for k, v in counters.items()},
              'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
              'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
              'float32_matmul_precision': torch.get_float32_matmul_precision(),
              **memory.result()}
    (args.output / 'metrics_261026.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'fingerprints_261026.json').write_text(json.dumps(fingerprints, indent=2) + '\n')
    print(json.dumps({'completed': True, 'count': len(rows),
                      'seconds': result['script_to_cli_complete_seconds'], 'native_calls': native_calls}), flush=True)


if __name__ == '__main__':
    main()
