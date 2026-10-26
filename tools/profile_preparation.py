"""Profile the unchanged optional pipeline; retain profiles and atom outputs locally."""
import argparse
import cProfile
import contextlib
import functools
import hashlib
import json
import os
from pathlib import Path
import pstats
import sys
import threading
import time

os.environ.setdefault("OPENMM_CPU_THREADS", "1")
sys.path.insert(0, "/opt/abodybuilder3/scripts")
import structure_preparation as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    import psutil
    stages = []
    stack = []

    def timed(name, function):
        @functools.wraps(function)
        def wrapped(*a, **kw):
            start = time.perf_counter()
            parent = stack[-1] if stack else None
            stack.append(name)
            try:
                return function(*a, **kw)
            finally:
                stack.pop()
                stages.append({"stage": name, "parent": parent,
                               "seconds": time.perf_counter() - start})
        return wrapped

    # Warm imports, then mark application and third-party boundaries without
    # changing any arguments, return values, platform or chemical protocol.
    np, mm, app, unit, fixer = sp._runtime()
    for name in ("_repair", "_reduce", "_geometry", "_changes", "_minimise", "_pdb"):
        setattr(sp, name, timed(name, getattr(sp, name)))
    app.Modeller.addHydrogens = timed("OpenMM.addHydrogens", app.Modeller.addHydrogens)
    app.ForceField.createSystem = timed("OpenMM.createSystem", app.ForceField.createSystem)
    fixer.findMissingAtoms = timed("PDBFixer.findMissingAtoms", fixer.findMissingAtoms)
    fixer.addMissingAtoms = timed("PDBFixer.addMissingAtoms", fixer.addMissingAtoms)
    original_run = sp.subprocess.run

    @functools.wraps(original_run)
    def subprocess_run(command, *a, **kw):
        return timed("subprocess." + ("reduce_version" if "-Version" in command else "reduce"),
                     original_run)(command, *a, **kw)

    sp.subprocess.run = subprocess_run
    jobs = [("openmm", "6yio_example", False),
            ("openmm", "6yio_example", True),
            ("reduce", "6yio_example", True),
            ("restrained", "6yio_example", True),
            ("openmm", "adalimumab_like", True),
            ("reduce", "adalimumab_like", True),
            ("restrained", "adalimumab_like", True)]
    rows = []
    for index, (mode, target, profile) in enumerate(jobs):
        directory = args.root / f"{index:02d}_{target}_{mode}"
        directory.mkdir()
        raw = (args.root / "inputs" / f"{target}.pdb").read_text()
        options = sp.Options(ph=7., seed=42,
                             hydrogen_method="reduce" if mode == "reduce" else "openmm",
                             sidechain_flips=mode == "reduce",
                             relax="restrained" if mode == "restrained" else "none",
                             relax_platform="CUDA" if mode == "restrained" else "CPU")
        stages.clear()
        stop = threading.Event()
        memory = []
        process = psutil.Process()

        def sample():
            while not stop.is_set():
                memory.append(process.memory_info().rss)
                stop.wait(.05)

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        profiler = cProfile.Profile()
        started = time.perf_counter()
        try:
            with profiler if profile else contextlib.nullcontext():
                prepared, relaxed, report = sp.prepare_structure(raw, options)
            elapsed = time.perf_counter() - started
        finally:
            stop.set(); thread.join()
        (directory / "prepared.pdb").write_text(prepared)
        if relaxed:
            (directory / "relaxed.pdb").write_text(relaxed)
        (directory / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        functions = []
        if profile:
            profiler.dump_stats(str(directory / "profile.pstats"))
            for (filename, line, function), (primitive, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
                functions.append({"file": filename, "line": line, "function": function,
                                  "calls": calls, "self_seconds": own, "cumulative_seconds": cumulative})
            functions.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
        row = {"mode": mode, "target": target, "profiled": profile,
               "seconds": elapsed, "rss_peak_bytes": max(memory),
               "stages": list(stages), "functions": functions[:65],
               "prepared_sha256": hashlib.sha256(prepared.encode()).hexdigest(),
               "status": report["status"]}
        rows.append(row)
        (args.root / "profile_measurements_261026.json").write_text(
            json.dumps({"completed": len(rows) == len(jobs), "rows": rows}, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k not in ("stages", "functions")}), flush=True)


if __name__ == "__main__":
    main()
