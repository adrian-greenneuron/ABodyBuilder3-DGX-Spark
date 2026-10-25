"""Measure the frozen CLI; worker CUDA samples come from optional sitecustomize."""
import json
from pathlib import Path
import runpy
import sys
import time
from benchmark_runtime import Memory

start = time.perf_counter()
completed = False
try:
    with Memory() as mem:
        sys.argv = ['/frozen/predict_main.py', *sys.argv[1:]]
        try:
            runpy.run_path('/frozen/predict_main.py', run_name='__main__')
            completed = True
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
            completed = True
finally:
    Path('/output/cli_metrics.json').write_text(json.dumps({
        'completed': completed, 'cli_seconds': time.perf_counter() - start, **mem.result(),
    }, indent=2) + '\n')
