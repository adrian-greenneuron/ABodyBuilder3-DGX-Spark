"""Profile the existing batched predictor without changing model settings.

Run inside the pinned deployment image. The labelled, serial stages identify
operators and allocations; they are not a timing breakdown of the overlapping
production pipeline. Unprofiled complete-job measurements remain authoritative.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from benchmark_runtime import Memory


def load_cli():
    path = Path('/opt/abodybuilder3/scripts/predict.py')
    spec = importlib.util.spec_from_file_location('profile_cli', path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cli.VERBOSE = False
    return cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['base', 'plddt'], default='plddt')
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--final-only', action='store_true')
    parser.add_argument('--reuse-pair-buffer', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.count, args.batch_size, args.repeats) < 1:
        parser.error('Counts must be positive.')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cli = load_cli()
    from abodybuilder3.batched_inference import prepare_cpu_batch, batch_add_atom37_to_output
    sys.path.insert(0, '/opt/abodybuilder3/scripts')
    from batch_pipeline import BatchPolicy, plan_batches
    fixtures = json.loads(args.fixtures.read_text())
    items = [(str(i), fixtures[i % len(fixtures)]['heavy'], fixtures[i % len(fixtures)]['light'])
             for i in range(args.count)]
    groups = list(plan_batches(items, BatchPolicy(args.batch_size)))
    model = cli.load_model(args.model, torch.device('cuda'))

    def execute():
        for group in groups:
            with torch.profiler.record_function('stage.cpu_features'):
                cpu, inputs = prepare_cpu_batch([(h, l) for _, h, l in group], compact=True) if args.compact else prepare_cpu_batch([(h, l) for _, h, l in group])
            with torch.profiler.record_function('stage.to_device'):
                batch = {k: v.to('cuda') for k, v in cpu.items()}
            with torch.no_grad(), torch.profiler.record_function('stage.model'):
                options = {}
                if args.compact:
                    batch['pair'] = projector(batch.pop('pair_code'))
                    options['_inference_pair_projected'] = True
                if args.final_only:
                    options['_inference_final_only'] = True
                if args.reuse_pair_buffer:
                    options['_inference_pair_buffer'] = True
                output = model({'single': batch['single'], 'pair': batch['pair']},
                               batch['aatype'], batch['seq_mask'], **options)
            with torch.profiler.record_function('stage.atoms_and_cpu_transfer'):
                result = batch_add_atom37_to_output(output, batch['aatype'],
                                                    [len(x['aatype']) for x in inputs])
            with torch.profiler.record_function('stage.format_pdb'):
                for out, inp in zip(result, inputs):
                    cli.generate_pdb(out, inp)
            del cpu, batch, inputs, result, output

    projector = None
    if args.compact:
        from abodybuilder3.inference_features import PairProjection
        projector = PairProjection(model.linear_in_edge)
    execute()
    torch.cuda.synchronize()
    measurements = []
    for repeat in range(args.repeats):
        torch.cuda.reset_peak_memory_stats()
        with Memory() as memory:
            started = time.perf_counter()
            execute()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        measurements.append({'repeat': repeat, 'seconds': elapsed,
                             'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                             'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                             **memory.result()})

    # Forward hooks label existing operations, rather than implementing another
    # forward path. Hooks exist only in the profiled pass, after timed controls.
    with ExitStack() as stack:
        def label(module, name):
            ranges = []
            def before(*_):
                context = torch.profiler.record_function(name)
                context.__enter__()
                ranges.append(context)
            def after(*_):
                ranges.pop().__exit__(None, None, None)
            stack.callback(module.register_forward_pre_hook(before).remove)
            stack.callback(module.register_forward_hook(after).remove)
        label(model.linear_in_edge, 'model.pair_projection')
        label(model.linear_in_node, 'model.node_projection')
        for i, module in enumerate(model.ipa_layers):
            label(module, f'model.ipa.{i}')
        for i, module in enumerate(model.angle_resnet_layers):
            label(module, f'model.sidechain_angles.{i}')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as profile:
            execute()
            torch.cuda.synchronize()
    if args.trace:
        profile.export_chrome_trace(str(args.output / 'trace.json'))
    events = []
    for event in profile.key_averages():
        events.append({'name': event.key, 'device_type': str(event.device_type), 'count': event.count,
                       'self_cpu_us': event.self_cpu_time_total,
                       'cpu_us': event.cpu_time_total,
                       'self_device_us': event.self_device_time_total,
                       'device_us': event.device_time_total,
                       'self_cpu_memory_bytes': event.self_cpu_memory_usage,
                       'self_device_memory_bytes': event.self_device_memory_usage})
    root = Path('/opt/abodybuilder3')
    paths = ['src/abodybuilder3/openfold/model/structure_module.py',
             'src/abodybuilder3/batched_inference.py', 'scripts/predict.py']
    report = {'model': args.model, 'count': args.count,
              'compact': args.compact, 'final_only': args.final_only,
              'reuse_pair_buffer': args.reuse_pair_buffer,
              'batch_sizes': [len(g) for g in groups],
              'scope': 'serial stage/operator profile; production stages overlap',
              'measurements': measurements,
              'median_seconds': statistics.median(x['seconds'] for x in measurements),
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'float32_matmul_precision': torch.get_float32_matmul_precision(),
              'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
              'source_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
              'events': sorted(events, key=lambda e: e['self_device_us'], reverse=True)}
    (args.output / 'profile_251026.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'completed': True, 'median_seconds': report['median_seconds'],
                      'top_device_operators': report['events'][:12]}), flush=True)


if __name__ == '__main__':
    main()
