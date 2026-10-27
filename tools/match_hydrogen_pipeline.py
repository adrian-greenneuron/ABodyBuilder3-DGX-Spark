"""Compare CPU/GPU hydrogen positions allowing same-parent label permutations.

This deliberately generous label-matching diagnostic never moves heavy atoms,
aligns structures, minimises coordinates or rewrites predictions. Parent and
residue identities remain local; exported records contain generic target IDs.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from openmm import app

sys.path.insert(0, "/opt/abodybuilder3/scripts")
import structure_preparation as sp


def load(path):
    pdb = app.PDBFile(str(path))
    atoms = list(pdb.topology.atoms())
    keys = sp._keys(pdb.topology)
    coordinates = dict(zip(keys, sp._positions(pdb.positions)*10))
    groups = {}
    for a, b in pdb.topology.bonds():
        if a.element.symbol == "H":
            a, b = b, a
        if b.element.symbol == "H":
            groups.setdefault(keys[a.index], []).append(keys[b.index])
    assert sum(map(len, groups.values())) == sum(a.element.symbol == "H" for a in atoms)
    return coordinates, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for route in ("openmm", "reduce"):
        base = args.study/"cli"/(route+"_cpu20_phase_r1")/"pdbs"
        candidate = args.study/"cli"/(route+"_gpu20_phase_r1")/"pdbs"
        for path in sorted(base.glob("*.prepared.pdb")):
            aa, groups_a = load(path)
            bb, groups_b = load(candidate/path.name)
            assert aa.keys() == bb.keys() and groups_a.keys() == groups_b.keys()
            named, matched = [], []
            for parent, members in groups_a.items():
                assert set(members) == set(groups_b[parent]) and len(members) <= 3
                a = np.array([aa[k] for k in members]); b = np.array([bb[k] for k in members])
                named.extend(np.linalg.norm(a-b, axis=1))
                order = min(itertools.permutations(range(len(members))),
                            key=lambda p: float(np.sum((a-b[list(p)])**2)))
                matched.extend(np.linalg.norm(a-b[list(order)], axis=1))
            rows.append({"route": route, "target": path.stem.removesuffix(".prepared"),
                         "H_count": len(named), "max_by_name_A": float(max(named)),
                         "rms_by_name_A": float(np.sqrt(np.mean(np.square(named)))),
                         "max_after_matching_A": float(max(matched)),
                         "rms_after_matching_A": float(np.sqrt(np.mean(np.square(matched))))})
        print("MATCHED", route, flush=True)
    args.output.write_text(json.dumps({"completed": True, "protocol": __doc__, "rows": rows}, indent=2)+"\n")


if __name__ == "__main__":
    main()
