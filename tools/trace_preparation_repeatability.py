"""Local float-array trace locating optional-stage repeatability differences."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, "/opt/abodybuilder3/scripts")


def initialise():
    os.environ["OPENMM_CPU_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def trace(job):
    source, directory = map(Path, job)
    import structure_preparation as sp
    import numpy as np
    directory.mkdir(parents=True)
    stages = {}
    original_repair = sp._repair
    np, mm, app, unit, _ = sp._runtime()
    original_hydrogens = app.Modeller.addHydrogens

    def repair(*a, **kw):
        original, fixer = original_repair(*a, **kw)
        stages["repair_positions_nm"] = sp._positions(fixer.positions).copy()
        return original, fixer

    def hydrogens(self, *a, **kw):
        result = original_hydrogens(self, *a, **kw)
        stages["hydrogenated_positions_nm"] = sp._positions(self.positions).copy()
        return result

    sp._repair = repair
    app.Modeller.addHydrogens = hydrogens
    try:
        prepared, _, audit = sp.prepare_structure(source.read_text(), sp.Options(seed=42, ph=7.))
    finally:
        sp._repair = original_repair
        app.Modeller.addHydrogens = original_hydrogens
    np.savez(directory / "stages.npz", **stages)
    (directory / "prepared.pdb").write_text(prepared)
    return {"directory": directory.name, "prepared_sha256": hashlib.sha256(prepared.encode()).hexdigest(),
            "stage_sha256": {key: hashlib.sha256(array.tobytes()).hexdigest() for key, array in stages.items()}}


def main():
    source, root = map(Path, sys.argv[1:])
    records = []
    for workers, count in ((1, 4), (4, 8), (8, 8)):
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialise) as pool:
            values = list(pool.map(trace, [(str(source), str(root / f"workers{workers}_{i}")) for i in range(count)]))
        records.append({"workers": workers, "outputs": values})
        (root / "repeatability_261026.json").write_text(json.dumps(records, indent=2) + "\n")
        print(workers, "PDB variants", len(set(v["prepared_sha256"] for v in values)),
              "repair variants", len(set(v["stage_sha256"]["repair_positions_nm"] for v in values)), flush=True)


if __name__ == "__main__":
    main()
