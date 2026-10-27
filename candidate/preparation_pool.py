"""Bounded, process-isolated hydrogen preparation after raw prediction finishes.

The controller runs as a separate small Python program so spawned workers do
not import the neural-network CLI or initialise its CUDA runtime. OpenMM's CPU
hydrogenation stays single-threaded in each worker. Relaxation remains serial.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
import time
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys


def worker_count(value, jobs, relaxation):
    if value == "auto":
        if relaxation != "none" or os.environ.get("OPENMM_CPU_THREADS", "1") != "1":
            return 1
        return min(8, jobs, os.cpu_count() or 1)
    try:
        count = int(value)
    except ValueError:
        raise ValueError("--preparation-workers must be auto or an integer from 1 to 20.") from None
    if count < 1 or count > 20:
        raise ValueError("--preparation-workers must be auto or an integer from 1 to 20.")
    if count > 1 and relaxation != "none":
        raise ValueError("Relaxation uses serial preparation; use --preparation-workers 1 or auto.")
    if count > 1 and os.environ.get("OPENMM_CPU_THREADS", "1") != "1":
        raise ValueError("Parallel preparation requires OPENMM_CPU_THREADS=1; extra engine threads change hydrogen coordinates.")
    return min(count, jobs)


def prepare_paths(paths, options, workers):
    """Return small per-path records; preserve raw files and chemistry audits."""
    paths = [str(Path(path).resolve()) for path in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate preparation paths are not allowed.")
    if not paths:
        return []
    payload = {"paths": paths, "options": asdict(options), "workers": workers}
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                            input=json.dumps(payload), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Preparation worker controller failed ({result.returncode}): {result.stderr[-2000:]}")
    records = json.loads(result.stdout)
    failed = [row for row in records if row["status"] == "failed"]
    if failed:
        details = "; ".join(f"{Path(row['path']).name}: {row['error']}" for row in failed[:3])
        raise RuntimeError(f"Preparation failed for {len(failed)} structures (raw files retained): {details}")
    return records


def _initialise():
    os.environ["OPENMM_CPU_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def _prepare_one(job):
    started = time.monotonic()
    result = _prepare_one_impl(job)
    result.update(started_monotonic=started, finished_monotonic=time.monotonic())
    return result


def _prepare_one_impl(job):
    path, options = job
    from structure_preparation import Options, PreparationError, prepare_structure
    path = Path(path)
    try:
        prepared, relaxed, report = prepare_structure(path.read_text(), Options(**options))
        if relaxed is not None:
            raise ValueError("Parallel preparation does not support relaxation.")
        path.with_suffix(".prepared.pdb").write_text(prepared)
        path.with_suffix(".preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        return {"path": str(path), "status": report["status"], "warnings": report["warnings"]}
    except PreparationError as exc:
        path.with_suffix(".preparation.json").write_text(json.dumps(exc.report, indent=2) + "\n")
        return {"path": str(path), "status": "failed", "error": str(exc), "warnings": []}
    except Exception as exc:
        return {"path": str(path), "status": "failed", "error": str(exc), "warnings": []}


def _bounded_map(pool, jobs, limit):
    iterator = iter(enumerate(jobs))
    pending = {}
    results = {}
    while True:
        while len(pending) < limit:
            try:
                index, job = next(iterator)
            except StopIteration:
                break
            pending[pool.submit(_prepare_one, job)] = index
        if not pending:
            break
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            results[pending.pop(future)] = future.result()
    return [results[index] for index in sorted(results)]


def main():
    payload = json.load(sys.stdin)
    paths, options, workers = payload["paths"], payload["options"], payload["workers"]
    if not 1 <= workers <= 20 or options["relax"] != "none":
        raise ValueError("Invalid parallel preparation request.")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate preparation paths.")
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialise) as pool:
        records = _bounded_map(pool, ((path, options) for path in paths), 2 * workers)
    print(json.dumps(records))


if __name__ == "__main__":
    main()
