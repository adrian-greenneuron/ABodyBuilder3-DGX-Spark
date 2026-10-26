"""Queue the frozen installed-CLI comparison; intended for the host."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    old = "abodybuilder3-dev-spark:language-lifecycle-261026"
    new = "abodybuilder3-dev-spark:parallel-preparation-candidate-261026"
    rows = []
    specs = [(mode, candidate, rep, False) for rep in (1, 2)
             for mode in ("openmm", "reduce")
             for candidate in ([False, True] if rep == 1 else [True, False])]
    specs += [("openmm", False, 1, True), ("openmm", True, 1, True)]
    for mode, candidate, rep, eager in specs:
        condition = "parallel" if candidate else "serial"
        suffix = "eager2" if eager else "aot8"
        relative = Path(suffix) / mode / condition / f"run{rep}"
        directory = args.root / relative; directory.parent.mkdir(parents=True, exist_ok=True)
        command = ["docker", "run", "--rm", "--network", "none", "--gpus", "all", "--ipc=host",
                   "--user", "1000:1000", "-e", "HOME=/tmp", "-e", "OPENMM_CPU_THREADS=1",
                   "-v", f"{args.root}:/results", new if candidate else old,
                   "python", "-u", "/results/frozen_tools/measure_preparation_cli.py",
                   "--input", f"/results/inputs/antibodies_{2 if eager else 8}.csv",
                   "--output", "/results/" + str(relative), "--mode", mode]
        if candidate:
            command += ["--candidate"]
        if eager:
            command += ["--eager"]
        started = time.perf_counter()
        with directory.with_suffix(".log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        if result.returncode:
            raise RuntimeError(f"Failed {directory}: {result.returncode}")
        record = {"condition": condition, "mode": mode, "replicate": rep, "eager": eager,
                  "container_seconds": elapsed, "image": new if candidate else old,
                  "metrics": json.loads((directory / "metrics_261026.json").read_text())}
        rows.append(record)
        (args.root / "measurements_261026.json").write_text(json.dumps(
            {"completed": len(rows) == len(specs), "rows": rows}, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "metrics"}), flush=True)


if __name__ == "__main__":
    main()
