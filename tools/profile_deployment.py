"""Stage/NVTX instrumentation for the installed AOT CLI; no prediction changes.

Use with and without Nsight to measure collection overhead. CUDA events measure
stream elapsed time; the Nsight kernel trace distinguishes kernels from idle
stream intervals and host synchronisation. Nested/overlapping times are not
additive fractions of complete-job wall time.
"""
import functools
import importlib.util
import json
from pathlib import Path
import threading
import time

import torch

START = time.perf_counter()
events, cuda_events, queues = [], [], []
lock = threading.Lock()
queue_counts = {'preparing': 0, 'ready': 0, 'waiting_to_write': 0, 'writing': 0}


def queue_event(**updates):
    with lock:
        for name, value in updates.items():
            queue_counts[name] += value
        queues.append({'seconds': time.perf_counter() - START, **queue_counts})


def stage(name, function, gpu=False):
    @functools.wraps(function)
    def call(*args, **kwargs):
        started = time.perf_counter()
        torch.cuda.nvtx.range_push('ABB3.' + name)
        first = last = None
        if gpu:
            first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            first.record()
        try:
            return function(*args, **kwargs)
        finally:
            if gpu:
                last.record()
            torch.cuda.nvtx.range_pop()
            with lock:
                events.append({'stage': name, 'start': started - START,
                               'end': time.perf_counter() - START,
                               'thread': threading.get_native_id()})
                if gpu:
                    cuda_events.append((name, first, last))
    return call


def main():
    import sys
    sys.path.insert(0, '/opt/abodybuilder3/scripts')
    import batch_pipeline
    from abodybuilder3 import aot_runtime, batched_inference, inference_features
    aot_runtime.AOTForward.__init__ = stage('package_loading', aot_runtime.AOTForward.__init__)
    aot_runtime.AOTForward.__call__ = stage('aot_forward', aot_runtime.AOTForward.__call__, gpu=True)
    inference_features.PairProjection.__call__ = stage('pair_projection', inference_features.PairProjection.__call__, gpu=True)
    batched_inference.batch_add_atom37_to_output = stage('atom_conversion_and_download', batched_inference.batch_add_atom37_to_output, gpu=True)
    import abodybuilder3.native_pdb as formatter
    formatter.to_pdb_fast = stage('pdb_formatting', formatter.to_pdb_fast)
    original_pipeline = batch_pipeline.run_pipeline

    def pipeline(batches, prepare, infer, write, **options):
        def prep(items):
            queue_event(preparing=1)
            try:
                return stage('feature_construction', prepare)(items)
            finally:
                queue_event(preparing=-1, ready=1)
        def prediction(items, prepared):
            queue_event(ready=-1)
            result = stage('inference_and_conversion', infer)(items, prepared)
            queue_event(waiting_to_write=1)
            return result
        def output(records):
            queue_event(waiting_to_write=-1, writing=1)
            try:
                return stage('output_batch', write)(records)
            finally:
                queue_event(writing=-1)
        return original_pipeline(batches, prep, prediction, output, **options)

    batch_pipeline.run_pipeline = pipeline
    spec = importlib.util.spec_from_file_location('deployment_measurement', Path(__file__).with_name('measure_deployment.py'))
    measurement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measurement)
    # Its import helper loads the installed CLI. Instrument only the loader's
    # return function, preserving the source, actual model and CLI selection.
    original_loader = importlib.util.spec_from_file_location
    def traced_spec(name, path, *args, **kwargs):
        spec = original_loader(name, path, *args, **kwargs)
        if str(path) == '/opt/abodybuilder3/scripts/predict.py':
            original_exec = spec.loader.exec_module
            def exec_module(module):
                original_exec(module)
                module.load_model = stage('checkpoint_loading', module.load_model)
            spec.loader.exec_module = exec_module
        return spec
    importlib.util.spec_from_file_location = traced_spec
    measurement.main()
    torch.cuda.synchronize()
    out = Path(sys.argv[sys.argv.index('--output') + 1]) if '--output' in sys.argv else Path('/results')
    # The CLI rewrites argv to --output /results/pdbs; keep stage evidence beside
    # its batch report rather than inside the structure directory.
    if out.name == 'pdbs':
        out = out.parent
    stream = [{'stage': name, 'stream_elapsed_ms': first.elapsed_time(last)}
              for name, first, last in cuda_events]
    result = {'completed': True, 'events': events, 'cuda_events': stream,
              'queue_events': queues,
              'timing_note': 'GPU stream elapsed includes launch gaps; use the kernel trace for kernel-only time.'}
    if any(queue_counts.values()):
        raise ValueError(f'Queue accounting did not drain: {queue_counts}')
    (out / 'stage_profile_261026.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
