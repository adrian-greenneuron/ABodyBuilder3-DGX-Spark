"""Finite, process-isolated optional-stage comparisons; no default changes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, "/opt/abodybuilder3/scripts")


def initialise_worker(threads):
    os.environ["OPENMM_CPU_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = "1"


def prepare_one(job):
    import structure_preparation as sp
    source, destination, mode = job
    options = sp.Options(ph=7., seed=42,
                         hydrogen_method="reduce" if mode == "reduce" else "openmm",
                         sidechain_flips=mode == "reduce",
                         relax="restrained" if mode == "restrained" else "none",
                         relax_platform="CUDA" if mode == "restrained" else "CPU")
    started = time.perf_counter()
    prepared, relaxed, report = sp.prepare_structure(Path(source).read_text(), options)
    path = Path(destination); path.mkdir(parents=True)
    (path / "prepared.pdb").write_text(prepared)
    if relaxed:
        (path / "relaxed.pdb").write_text(relaxed)
    (path / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"source": Path(source).name, "mode": mode,
            "seconds": time.perf_counter() - started,
            "prepared_sha256": hashlib.sha256(prepared.encode()).hexdigest(),
            "relaxed_sha256": hashlib.sha256(relaxed.encode()).hexdigest() if relaxed else None,
            "status": report["status"], "options": asdict(options)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--study", choices=["threads", "workers", "relax_workers"], required=True)
    args = parser.parse_args()
    import psutil
    all_inputs = sorted(args.inputs.glob("*.pdb"))
    if args.study == "threads":
        sources = all_inputs[:2]
        specs = [("openmm", 1, threads) for threads in (1, 2, 4)]
    elif args.study == "workers":
        sources = all_inputs
        specs = [(mode, workers, 1) for mode in ("openmm", "reduce") for workers in (1, 4)]
    else:
        sources = all_inputs[:5]
        specs = [("restrained", workers, 1) for workers in (1, 2)]
    records = []
    for rep in (1, 2):
        for mode, workers, threads in specs if rep == 1 else list(reversed(specs)):
            output = args.root / f"{mode}_workers{workers}_threads{threads}_run{rep}"
            output.mkdir(parents=True)
            jobs = [(str(p), str(output / p.stem), mode) for p in sources]
            process = psutil.Process()
            stop = threading.Event(); samples = []; container_samples = []

            def sample():
                while not stop.is_set():
                    try:
                        samples.append(sum(p.memory_info().rss for p in [process] + process.children(recursive=True)
                                           if p.is_running()))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    cgroup = Path("/sys/fs/cgroup/memory.current")
                    if cgroup.exists():
                        container_samples.append(int(cgroup.read_text()))
                    stop.wait(.1)

            thread = threading.Thread(target=sample, daemon=True); thread.start()
            started = time.perf_counter()
            try:
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                         initializer=initialise_worker, initargs=(threads,)) as pool:
                    rows = list(pool.map(prepare_one, jobs))
                elapsed = time.perf_counter() - started
            finally:
                stop.set(); thread.join()
            record = {"mode": mode, "workers": workers, "threads": threads, "replicate": rep,
                      "inputs": len(sources), "seconds_including_process_startup_and_writing": elapsed,
                      "process_tree_rss_peak_bytes": max(samples), "outputs": rows}
            record["container_memory_peak_bytes"] = max(container_samples) if container_samples else None
            records.append(record)
            (args.root / "measurements_261026.json").write_text(json.dumps(
                {"completed": len(records) == 2 * len(specs), "study": args.study, "rows": records}, indent=2) + "\n")
            print(json.dumps({k: v for k, v in record.items() if k != "outputs"}), flush=True)


if __name__ == "__main__":
    main()
