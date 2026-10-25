"""Measure the frozen CLI; worker CUDA samples come from optional sitecustomize."""
import json
import os
from pathlib import Path
import runpy
import sys
import time
from benchmark_runtime import Memory

start = time.perf_counter()
completed = False
fingerprints=[]
try:
    with Memory() as mem:
        mode=os.environ.get('ABB3_BENCHMARK_INFERENCE_MODE')
        if mode or os.environ.get('ABB3_FINGERPRINT_PATH'):
            import abodybuilder3.batched_inference as batching
            from prediction_fingerprint import fingerprint
            original=batching.infer_prepared_batch
            def measured(*args,**kwargs):
                if mode:
                    kwargs.update(final_only=mode!='lookup',reuse_pair_buffer=mode=='optimised')
                result=original(*args,**kwargs)
                if os.environ.get('ABB3_FINGERPRINT_PATH'):
                    fingerprints.extend(fingerprint(out,inp) for out,inp in zip(*result))
                return result
            batching.infer_prepared_batch=measured
        sys.argv = ['/frozen/predict_main.py', *sys.argv[1:]]
        try:
            runpy.run_path('/frozen/predict_main.py', run_name='__main__')
            completed = True
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
            completed = True
finally:
    if os.environ.get('ABB3_FINGERPRINT_PATH'):
        Path(os.environ['ABB3_FINGERPRINT_PATH']).write_text(json.dumps(fingerprints,indent=2)+'\n')
    Path('/output/cli_metrics.json').write_text(json.dumps({
        'completed': completed, 'cli_seconds': time.perf_counter() - start, **mem.result(),
    }, indent=2) + '\n')
