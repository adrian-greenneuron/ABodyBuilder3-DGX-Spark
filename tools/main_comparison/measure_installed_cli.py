"""Measure the installed CLI with its unchanged per-version defaults."""
import json,runpy,sys,time
from pathlib import Path
from benchmark_runtime import Memory
start=time.perf_counter();completed=False
try:
    with Memory() as memory:
        sys.argv=['/opt/abodybuilder3/scripts/predict.py',*sys.argv[1:]]
        try:
            runpy.run_path('/opt/abodybuilder3/scripts/predict.py',run_name='__main__')
            completed=True
        except SystemExit as error:
            if error.code not in (0,None):raise
            completed=True
finally:
    Path('/results/metrics_261026.json').write_text(json.dumps({'completed':completed,'wrapped_cli_seconds':time.perf_counter()-start,**memory.result()},indent=2)+'\n')
