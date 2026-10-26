"""Host controller for the four/eight-worker installed-CLI screen."""
import json
from pathlib import Path
import subprocess
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root.resolve()
records = []
for rep in (1, 2):
    for mode in ("openmm", "reduce"):
        for workers in (4, 8) if rep == 1 else (8, 4):
            relative = Path("worker_cap") / mode / f"workers{workers}" / f"run{rep}"
            dest = root / relative; dest.parent.mkdir(parents=True, exist_ok=True)
            command = ["docker", "run", "--rm", "--network", "none", "--gpus", "all", "--ipc=host",
                       "--user", "1000:1000", "-e", "HOME=/tmp", "-e", "OPENMM_CPU_THREADS=1",
                       "-v", f"{root}:/results", "abodybuilder3-dev-spark:parallel-preparation-candidate-261026",
                       "python", "-u", "/results/frozen_tools/measure_preparation_cli.py",
                       "--input", "/results/inputs/antibodies_8.csv", "--output", "/results/" + str(relative),
                       "--mode", mode, "--candidate", "--preparation-workers", str(workers)]
            start = time.perf_counter()
            with dest.with_suffix(".log").open("w") as log:
                p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            duration = time.perf_counter() - start
            if p.returncode:
                raise RuntimeError(f"Failed {dest}")
            records.append({"mode": mode, "workers": workers, "replicate": rep,
                            "container_seconds": duration,
                            "metrics": json.loads((dest / "metrics_261026.json").read_text())})
            (root / "worker_cap_measurements_261026.json").write_text(json.dumps(
                {"completed": len(records) == 8, "rows": records}, indent=2) + "\n")
            print(mode, workers, rep, duration, flush=True)
