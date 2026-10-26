"""Repeat the previously variable input with serial and concurrent repair.

This diagnostic captures full precision before PDB serialization. It is not a
complete-CLI performance benchmark and uses different counts per worker group.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time


def worker(job):
    source, destination, policy = job
    os.environ["OPENMM_CPU_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["AB3_EXPERIMENTAL_REPAIR"] = policy
    sys.path.insert(0, "/opt/abodybuilder3/scripts")
    import structure_preparation as sp
    import numpy as np
    destination = Path(destination)
    destination.mkdir(parents=True)
    arrays = {}
    identities = {}
    contexts = []
    original_repair = sp._repair
    _, mm, app, unit, _ = sp._runtime()
    original_hydrogens = app.Modeller.addHydrogens
    original_context_init = mm.Context.__init__
    original_set_positions = mm.Context.setPositions

    def context_init(self, *args, **kwargs):
        original_context_init(self, *args, **kwargs)
        platform = self.getPlatform()
        contexts.append({"platform": platform.getName(), "properties": {
            key: platform.getPropertyValue(self, key) for key in platform.getPropertyNames()}})

    def set_positions(self, positions):
        if len(contexts) == 1 and "repair_initial_nm" not in arrays:
            arrays["repair_initial_nm"] = sp._positions(positions).copy()
        return original_set_positions(self, positions)

    def repair(*args, **kwargs):
        original, fixer = original_repair(*args, **kwargs)
        arrays["repair_nm"] = sp._positions(fixer.positions).copy()
        identities["repair"] = sp._keys(fixer.topology)
        return original, fixer

    def hydrogens(self, *args, **kwargs):
        result = original_hydrogens(self, *args, **kwargs)
        arrays["hydrogens_nm"] = sp._positions(self.positions).copy()
        identities["hydrogens"] = sp._keys(self.topology)
        return result

    sp._repair = repair
    app.Modeller.addHydrogens = hydrogens
    mm.Context.__init__ = context_init
    mm.Context.setPositions = set_positions
    started = time.perf_counter()
    try:
        prepared, _, audit = sp.prepare_structure(Path(source).read_text(), sp.Options(seed=42, ph=7.))
    finally:
        sp._repair = original_repair
        app.Modeller.addHydrogens = original_hydrogens
        mm.Context.__init__ = original_context_init
        mm.Context.setPositions = original_set_positions
    elapsed = time.perf_counter() - started
    np.savez(destination / "stages.npz", **arrays)
    (destination / "prepared.pdb").write_text(prepared)
    (destination / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (destination / "identities.json").write_text(json.dumps(identities))
    record = {"directory": str(destination), "policy": policy, "seconds": elapsed,
              "prepared_sha256": hashlib.sha256(prepared.encode()).hexdigest(),
              "contexts": contexts, "stages_sha256": {
                  key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()}}
    (destination / "trace.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", default=["reference", "cuda_deterministic"])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    records = []
    specs = [(policy, workers, rep) for rep in (1, 2)
             for policy in (args.policies if rep == 1 else args.policies[::-1])
             for workers in ([1, 8] if rep == 1 else [8, 1])]
    for policy, workers, rep in specs:
        directory = args.root / f"{policy}_w{workers}_r{rep}"
        count = 4 if workers == 1 else 16
        jobs = [(str(args.source), str(directory / f"copy_{i:02d}"), policy) for i in range(count)]
        print("START", directory.name, flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            outputs = list(pool.map(worker, jobs))
        row = {"policy": policy, "workers": workers, "replicate": rep, "outputs": outputs,
               "pdb_variants": len({x["prepared_sha256"] for x in outputs}),
               "repair_variants": len({x["stages_sha256"]["repair_nm"] for x in outputs}),
               "hydrogen_variants": len({x["stages_sha256"]["hydrogens_nm"] for x in outputs})}
        records.append(row)
        (args.root / "screen_measurements_261026.json").write_text(json.dumps(
            {"completed": len(records) == len(specs), "rows": records}, indent=2) + "\n")
        print("DONE", directory.name, {k: row[k] for k in ("pdb_variants", "repair_variants", "hydrogen_variants")}, flush=True)


if __name__ == "__main__":
    main()
