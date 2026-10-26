"""Measure original/deferred adapters with identical capture in output workers."""
import time
START = time.perf_counter()
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import threading

import numpy as np
import torch
from benchmark_runtime import Memory
from prediction_fingerprint import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['reference', 'deferred'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['base', 'plddt'], required=True)
    parser.add_argument('--aot-root')
    parser.add_argument('--batch-size', default='auto')
    parser.add_argument('--batch-memory-gib', default='6')
    parser.add_argument('--fixed-bank-root')
    args = parser.parse_args()
    (args.output / 'pdbs').mkdir()
    rows = list(csv.DictReader(args.input.open()))
    by_name = {row['name']: row.get('target_id') or row['name'] for row in rows}
    sys.path.insert(0, '/opt/abodybuilder3/scripts')
    if args.fixed_bank_root:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'candidate'))
        import shape_banks
        from abodybuilder3 import aot_runtime, batched_inference
        original_prepare = batched_inference.prepare_cpu_batch
        def pad_prepare(*values, **options):
            return shape_banks.pad_prepared(original_prepare(*values, **options))
        batched_inference.prepare_cpu_batch = pad_prepare
        aot_runtime.AOTForward = lambda root, model, checkpoint, fallback: shape_banks.ShapeBankForward(
            args.fixed_bank_root, root, model, checkpoint, fallback)
    if args.mode == 'deferred':
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'candidate'))
        import deferred_inference
        import abodybuilder3.batched_inference as batching
        prepare = batching.prepare_cpu_batch
        def pinned_prepare(*values, **options):
            return deferred_inference.pin_prepared(prepare(*values, **options))
        batching.prepare_cpu_batch = pinned_prepare
        batching.infer_prepared_batch = deferred_inference.infer_prepared_batch
    spec = importlib.util.spec_from_file_location('pipeline_component_cli', '/opt/abodybuilder3/scripts/predict.py')
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    import batch_pipeline
    original_pipeline = batch_pipeline.run_pipeline
    fingerprints, captured = [], {}
    lock = threading.Lock()
    def pipeline(batches, prepare, infer, write, **options):
        def prediction(items, prepared):
            result = infer(items, prepared)
            for (name, _, _), _, inp in result:
                inp['_comparison_name'] = name
            return result
        return original_pipeline(batches, prepare, prediction, write, **options)
    batch_pipeline.run_pipeline = pipeline
    original_pdb = cli.generate_pdb
    def generate_pdb(output, inp):
        name = inp['_comparison_name']
        fp = {'name': name, 'source': by_name[name], **fingerprint(output, inp)}
        with lock:
            fingerprints.append(fp)
            if by_name[name] not in captured and len(captured) < 150:
                captured[by_name[name]] = {'positions': np.array(output['atom37'], copy=True),
                    'masks': np.array(output['atom37_atom_exists'], copy=True),
                    'confidence_logits': np.array(output.get('plddt', np.empty(0, np.float32)), copy=True),
                    'heavy': inp['is_heavy'].numpy().astype(bool).copy(), 'aatype': inp['aatype'].numpy().copy()}
        return original_pdb(output, inp)
    cli.generate_pdb = generate_pdb
    sys.argv = ['predict.py', 'predict', '--csv', str(args.input), '--output', str(args.output / 'pdbs'),
                '--model', args.model, '--workers', '1', '--seed', '42', '--quiet',
                '--batch-report', str(args.output / 'batching.json'), '--batch-size', args.batch_size,
                '--batch-memory-gib', args.batch_memory_gib]
    if args.aot_root:
        sys.argv += ['--aot-root', args.aot_root]
    with Memory() as memory:
        try:
            cli.app()
        except SystemExit as error:
            if error.code not in (0, None):
                raise
        finished = time.perf_counter()
    if len(fingerprints) != len(rows):
        raise ValueError('Missing component comparison fingerprints.')
    raw = args.output / 'raw'; raw.mkdir()
    for name, value in captured.items():
        np.savez(raw / (name + '.npz'), **value)
    (args.output / 'fingerprints_261026.json').write_text(json.dumps(sorted(fingerprints, key=lambda row: row['name']), indent=2) + '\n')
    result = {'completed': True, 'model': args.model, 'mode': args.mode, 'count': len(rows),
              'script_to_cli_complete_seconds': finished - START,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), **memory.result()}
    (args.output / 'metrics_261026.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
