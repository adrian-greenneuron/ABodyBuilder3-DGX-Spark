"""Isolated measurement adapter; never installed in the production CLI.

Run the frozen dev CLI and its unmodified overlapping CPU/GPU pipeline. Only
the requested model arithmetic/compiled forward is changed in this process.
"""
import time
SCRIPT_START = time.perf_counter()
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from benchmark_runtime import Memory
from prediction_fingerprint import fingerprint


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['eager', 'compiled', 'combined', 'graph', 'aot'], required=True)
    parser.add_argument('--precision', choices=['normal', 'strict'], required=True)
    parser.add_argument('--model', choices=['base', 'plddt'], required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--output', type=Path, default=Path('/output'))
    parser.add_argument('--date-code', default='251026')
    parser.add_argument('--batch-size',default='auto')
    parser.add_argument('--cpu-workers',type=int,default=2)
    parser.add_argument('--batch-memory-gib',type=float,default=6.)
    parser.add_argument('--aot-root',type=Path,help='Qualified AOT packages; only used by the isolated AOT condition.')
    args = parser.parse_args()
    output = args.output
    cli = module_from_file('measured_pipeline_cli', '/frozen/predict_main.py')
    if args.precision == 'strict':
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
    initial_precision = torch.get_float32_matmul_precision()
    initial_tf32 = torch.backends.cuda.matmul.allow_tf32
    source = list(csv.DictReader(Path(f'/frozen/antibodies_{args.count}.csv').open()))
    identities = json.loads(Path('/frozen/input_identities_251026.json').read_text())
    by_name = identities[str(args.count)]
    original_load = cli.load_model
    forwards = []
    fingerprints = []
    captured = {}
    load_records = []

    def load_model(*a, **kw):
        start = time.perf_counter()
        model = original_load(*a, **kw)
        load_records.append({'checkpoint_loading_seconds': time.perf_counter()-start})
        if args.mode == 'graph':
            from graph_runtime import GraphForward
            model.forward=GraphForward(model.forward,(16 if args.count<=128 else 32) if args.batch_size=='auto' else int(args.batch_size),
                max(len(row['heavy'])+len(row['light']) for row in source))
            load_records[-1]['graph_forward']=model.forward.events
        elif args.mode != 'eager':
            candidate = module_from_file('pipeline_candidate', '/component/candidate/structure_module.py')
            model.__class__ = candidate.StructureModule
            for layer in model.ipa_layers:
                layer.__class__ = candidate.InvariantPointAttention
            original_forward = model.forward
            if args.mode == 'aot':
                if args.precision != 'strict' or args.aot_root is None:
                    raise ValueError('AOT comparison requires strict FP32 and --aot-root.')
                from aot_model import AOTForward
                execute = AOTForward(args.aot_root,args.model)
                load_records[-1]['aot_package_loading'] = execute.events
            else:
                execute = torch.compile(original_forward, mode='default', dynamic=True)

            def forward(*values, **options):
                if args.mode == 'combined':
                    options['_inference_point_matmul'] = True
                start = time.perf_counter()
                result = execute(*values, **options)
                # Atom conversion in the unchanged caller synchronises later.
                forwards.append({'host_forward_seconds': time.perf_counter()-start,
                                 'batch_size': int(values[1].shape[0]),
                                 'padded_length': int(values[1].shape[1])})
                return result
            model.forward = forward
        print(json.dumps({'event': 'model_loaded', 'model': args.model,
                          'mode': args.mode, 'precision': args.precision}), flush=True)
        return model

    cli.load_model = load_model
    # The CLI imports this module when invoking its real run_batched_pipeline.
    import batch_pipeline
    original_pipeline = batch_pipeline.run_pipeline

    def pipeline(batches, prepare, infer, write, **options):
        def measured_infer(items, prepared):
            start = time.perf_counter()
            records = infer(items, prepared)
            duration = time.perf_counter()-start
            for (name, heavy, light), result, inputs in records:
                fingerprints.append({'name': name, 'source': by_name[name],
                                     **fingerprint(result, inputs)})
                truth = by_name[name]
                if truth not in captured:
                    captured[truth] = {'positions': np.array(result['atom37'], copy=True),
                                      'masks': np.array(result['atom37_atom_exists'], copy=True),
                                      'confidence_logits': np.array(result.get('plddt', np.empty((0,), np.float32)), copy=True),
                                      'heavy': inputs['is_heavy'].numpy().astype(bool).copy(),
                                      'aatype': inputs['aatype'].numpy().copy()}
            if len(fingerprints) == len(items):
                print(json.dumps({'event': 'first_batch_complete', 'items': len(items),
                                  'inference_and_conversion_seconds': duration}), flush=True)
            return records
        return original_pipeline(batches, prepare, measured_infer, write, **options)

    batch_pipeline.run_pipeline = pipeline
    completed = False
    with Memory() as memory:
        sys.argv = ['/frozen/predict_main.py', 'predict', '--csv', f'/frozen/antibodies_{args.count}.csv',
                    '--output', str(output/'pdbs')+'/', '--model', args.model, '--workers', '1',
                    '--batch-size', args.batch_size, '--cpu-workers', str(args.cpu_workers),
                    '--batch-memory-gib',str(args.batch_memory_gib),'--seed', '42', '--quiet',
                    '--batch-report', str(output/'batching.json')]
        try:
            cli.app()
            completed = True
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
            completed = True
        cli_complete = time.perf_counter()
    assert completed and len(fingerprints) == args.count
    assert len(captured) == 100
    assert sorted(p.name for p in (output/'pdbs').glob('*.pdb')) == sorted(r['name']+'.pdb' for r in source)
    assert torch.get_float32_matmul_precision() == initial_precision
    assert torch.backends.cuda.matmul.allow_tf32 == initial_tf32
    evidence_start = time.perf_counter()
    evidence = output/'raw'; evidence.mkdir()
    for name, fields in captured.items():
        np.savez(evidence/(name+'.npz'), **fields)
    (output/'fingerprints.json').write_text(json.dumps(sorted(fingerprints, key=lambda x:x['name']), indent=2)+'\n')
    evidence_seconds = time.perf_counter()-evidence_start
    from torch._dynamo.utils import counters
    metadata = {'completed': completed, 'model': args.model, 'mode': args.mode, 'precision': args.precision,
                'count': args.count, 'distinct_experimental_targets': len(captured),
                'requested_batch_size':args.batch_size,'cpu_workers':args.cpu_workers,'batch_memory_gib':args.batch_memory_gib,
                'script_to_cli_complete_seconds': cli_complete-SCRIPT_START,
                'raw_evidence_writing_seconds': evidence_seconds,
                'allow_tf32': initial_tf32, 'float32_matmul_precision': initial_precision,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'compile': {'mode': 'default', 'dynamic': True} if args.mode in ('compiled','combined') else None,
                'aot_root':str(args.aot_root) if args.mode=='aot' else None,
                'checkpoint_loading': load_records, 'forward_calls': forwards,
                'dynamo_counters': {str(k): dict(v) for k,v in counters.items()},
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                **memory.result()}
    cache_manifest=Path('/opt/abodybuilder3_compilation_cache_manifest.json')
    metadata['packaged_cache_manifest_sha256'] = hashlib.sha256(cache_manifest.read_bytes()).hexdigest() if cache_manifest.exists() else None
    (output/f'metrics_{args.date_code}.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'event': 'completed', 'count': args.count, 'mode': args.mode,
                      'cli_seconds': metadata['script_to_cli_complete_seconds'],
                      'raw_evidence_seconds': evidence_seconds}), flush=True)


if __name__ == '__main__':
    main()
