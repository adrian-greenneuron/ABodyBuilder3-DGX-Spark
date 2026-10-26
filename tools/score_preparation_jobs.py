"""Match atom identities and compare chemistry candidates to their own repeats."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def atoms(path):
    result = {}
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            key = (line[21:22], line[22:27], line[17:20], line[12:16])
            if key in result:
                raise ValueError(f"Duplicate atom identity: {key}")
            result[key] = (np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                           line[76:78].strip())
    return result


def coordinate_change(left, right, name):
    a, b = atoms(left / name), atoms(right / name)
    if a.keys() != b.keys():
        return {"atom_identities_equal": False, "left_atoms": len(a), "right_atoms": len(b)}
    changes = {}
    for label, elements in (("all", None), ("hydrogen", {"H"}), ("heavy", {"C", "N", "O", "S"})):
        d = [float(np.linalg.norm(a[k][0] - b[k][0])) for k in a
             if elements is None or a[k][1] in elements]
        changes[label + "_coordinate_rms_angstrom"] = float(np.sqrt(np.mean(np.square(d)))) if d else None
        changes[label + "_max_displacement_angstrom"] = max(d, default=None)
    return {"atom_identities_equal": True, "atoms": len(a),
            "pdb_bytes_equal": (left / name).read_bytes() == (right / name).read_bytes(), **changes}


def compare(left, right):
    a = json.loads((left / "audit.json").read_text())
    b = json.loads((right / "audit.json").read_text())
    result = {"prepared": coordinate_change(left, right, "prepared.pdb"),
              "protonation_variants_equal": a["protonation_variants"] == b["protonation_variants"],
              "warnings_equal": a["warnings"] == b["warnings"],
              "prepared_overlaps": [a["prepared_geometry"]["severe_heavy_atom_overlaps"],
                                     b["prepared_geometry"]["severe_heavy_atom_overlaps"]],
              "max_original_heavy_displacement_angstrom": [a["preparation_changes"]["max_original_heavy_displacement_angstrom"],
                                                            b["preparation_changes"]["max_original_heavy_displacement_angstrom"]]}
    if (left / "relaxed.pdb").exists():
        result["relaxed"] = coordinate_change(left, right, "relaxed.pdb")
        for stage in ("hydrogen_minimisation", "restrained_minimisation"):
            result[stage] = {key: [a[stage][key], b[stage][key]] for key in
                             ("initial_energy_kj_mol", "final_energy_kj_mol", "final_physical_energy_kj_mol",
                              "rms_force_kj_mol_nm", "converged")}
        result["relaxed_overlaps"] = [a["relaxed_geometry"]["severe_heavy_atom_overlaps"],
                                      b["relaxed_geometry"]["severe_heavy_atom_overlaps"]]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for study in ("thread_screen", "workers", "relax_workers"):
        data = json.loads((args.root / study / "measurements_261026.json").read_text())
        groups = {}
        for row in data["rows"]:
            group = f"{row['mode']}_workers{row['workers']}_threads{row['threads']}"
            for output in row["outputs"]:
                name = Path(output["source"]).stem
                groups.setdefault((row["mode"], name), []).append(
                    (group, row["replicate"], args.root / study / f"{group}_run{row['replicate']}" / name))
        for (mode, target), values in groups.items():
            for (left_group, left_rep, left), (right_group, right_rep, right) in itertools.combinations(values, 2):
                # Retain both within-condition repeats and all matched-rep candidate comparisons.
                if left_group != right_group and left_rep != right_rep:
                    continue
                rows.append({"study": study, "mode": mode, "target": target,
                             "left": left_group, "left_replicate": left_rep,
                             "right": right_group, "right_replicate": right_rep,
                             "within_condition": left_group == right_group, **compare(left, right)})
    (args.root / "paired_preparation_differences_261026.json").write_text(
        json.dumps({"completed": True, "coordinate_precision": "PDB rounded to 0.001 angstrom", "rows": rows}, indent=2) + "\n")
    for study in ("thread_screen", "workers", "relax_workers"):
        for within in (True, False):
            selection = [r for r in rows if r["study"] == study and r["within_condition"] == within]
            print(study, "within" if within else "between", "pairs", len(selection),
                  "identities equal", all(r["prepared"]["atom_identities_equal"] for r in selection),
                  "protonation equal", all(r["protonation_variants_equal"] for r in selection),
                  "prepared exact", sum(r["prepared"]["pdb_bytes_equal"] for r in selection),
                  "max heavy/H", max(r["prepared"]["heavy_max_displacement_angstrom"] for r in selection),
                  max(r["prepared"]["hydrogen_max_displacement_angstrom"] for r in selection))


if __name__ == "__main__":
    main()
