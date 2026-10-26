"""Paired coordinate, topology/audit and system-memory summaries; no alignment."""
import argparse
import itertools
import json
from pathlib import Path
import statistics

import numpy as np


def atoms(path):
    rows = [line for line in path.read_text().splitlines() if line.startswith("ATOM  ")]
    keys = [(line[21], line[22:26].strip(), line[26], line[17:20].strip(), line[12:16].strip()) for line in rows]
    assert len(set(keys)) == len(keys)
    return {key: (np.array([float(line[k:k+8]) for k in (30, 38, 46)]), line[76:78].strip())
            for key, line in zip(keys, rows)}


def compare(left, right, raw):
    a, b, original = atoms(left), atoms(right), atoms(raw)
    assert set(a) == set(b), (left, right, "atom identity differs")
    assert all(a[key][1] == b[key][1] for key in a)
    groups = {"all_atoms": list(a),
              "original_heavy": [key for key in a if key in original],
              "added_heavy": [key for key in a if key not in original and a[key][1] != "H"],
              "hydrogens": [key for key in a if a[key][1] == "H"],
              "backbone": [key for key in a if key in original and key[-1] in ("N", "CA", "C", "O")]}
    result = {"left": str(left), "right": str(right), "raw": str(raw),
              "byte_identical": left.read_bytes() == right.read_bytes(), "identities_equal": True}
    for group, keys in groups.items():
        delta = np.array([a[key][0] - b[key][0] for key in keys])
        norms = np.linalg.norm(delta, axis=1) if len(keys) else np.array([])
        result[group] = {"atoms": len(keys), "changed_atoms": int(np.count_nonzero(norms)),
                         "max_displacement_angstrom": float(norms.max(initial=0.)),
                         "rms_displacement_angstrom": float(np.sqrt(np.mean(norms**2))) if len(keys) else 0.}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    root = args.study.resolve()
    data = json.loads((root / "cli_measurements_261026.json").read_text())
    assert data["completed"] and len(data["rows"]) == 20
    summaries = []
    comparisons = []
    audit_comparisons = []
    conditions = ["reference_serial", "cpu_serial", "cpu_parallel", "double_serial", "double_parallel"]
    for route in ("openmm", "reduce"):
        rows = [row for row in data["rows"] if row["route"] == route]
        reference = next(row for row in rows if row["condition"] == "reference_serial" and row["replicate"] == 1)
        reference_directory = root / "cli" / reference["label"] / "pdbs"
        names = [name[:-4] for name in reference["outputs"] if not name.endswith(".prepared.pdb")]
        reference_median = statistics.median(row["complete_container_seconds"] for row in rows if row["condition"] == "reference_serial")
        for condition in conditions:
            matched = [row for row in rows if row["condition"] == condition]
            times = [row["complete_container_seconds"] for row in matched]
            ram = [row["peak_additional_unavailable_bytes"]/2**30 for row in matched]
            summary = {"route": route, "condition": condition, "times_seconds": times,
                       "median_seconds": statistics.median(times), "speedup": reference_median/statistics.median(times),
                       "peak_additional_ram_gib": max(ram), "ram_repeats_gib": ram,
                       "ram_sensitivity_gib": [min(row["ram_sensitivity_bytes"][0] for row in matched)/2**30,
                                               max(row["ram_sensitivity_bytes"][1] for row in matched)/2**30],
                       "baseline_drift_flags": sum(row["baseline_drift_flag"] for row in matched)}
            repeat_comparisons = []
            first, second = sorted(matched, key=lambda row: row["replicate"])
            for name in names:
                left = root / "cli" / first["label"] / "pdbs" / f"{name}.prepared.pdb"
                right = root / "cli" / second["label"] / "pdbs" / f"{name}.prepared.pdb"
                item = compare(left, right, reference_directory / f"{name}.pdb")
                item.update(kind="within_condition_repeat", route=route, condition=condition, target=name)
                repeat_comparisons.append(item)
                comparisons.append(item)
            summary["repeat_identical_pdbs"] = sum(item["byte_identical"] for item in repeat_comparisons)
            summary["repeat_targets"] = len(names)
            summaries.append(summary)
        # Every condition/replicate vs reference, plus all condition pairs at matched rep.
        for replicate in (1, 2):
            matched = [row for row in rows if row["replicate"] == replicate]
            for first, second in itertools.combinations(matched, 2):
                a = root / "cli" / first["label"] / "pdbs"
                b = root / "cli" / second["label"] / "pdbs"
                for name in names:
                    item = compare(a / f"{name}.prepared.pdb", b / f"{name}.prepared.pdb", reference_directory / f"{name}.pdb")
                    item.update(kind="between_condition", route=route, first=first["condition"], second=second["condition"],
                                replicate=replicate, target=name)
                    comparisons.append(item)
                    aa = json.loads((a / f"{name}.preparation.json").read_text())
                    bb = json.loads((b / f"{name}.preparation.json").read_text())
                    assert aa["protonation_variants"] == bb["protonation_variants"]
                    assert aa["repair"] == bb["repair"]
                    for key in ("atoms", "hydrogens", "residues", "severe_heavy_atom_overlaps", "peptide_cn_over_1_8_angstrom"):
                        assert aa["prepared_geometry"][key] == bb["prepared_geometry"][key], (key, name, a, b)
                    assert aa["prepared_geometry"]["disulfides"] == bb["prepared_geometry"]["disulfides"]
                    assert aa["warnings"] == bb["warnings"]
                    if route == "reduce":
                        assert aa["reduce"]["initial_hydrogens"] == bb["reduce"]["initial_hydrogens"]
                    audit_comparisons.append({"route": route, "target": name, "first": first["condition"], "second": second["condition"],
                                              "replicate": replicate, "protonation_geometry_and_warnings_equal": True})
        # Raw model output must be exact across all conditions and technical repeats.
        for name in names:
            raw = (reference_directory / f"{name}.pdb").read_bytes()
            assert all((root / "cli" / row["label"] / "pdbs" / f"{name}.pdb").read_bytes() == raw for row in rows)
    screen_summaries = []
    float_comparisons = []
    representatives = {}
    for directory in ("screen", "screen2"):
        screen = json.loads((root / directory / "screen_measurements_261026.json").read_text())
        assert screen["completed"]
        policies = sorted({row["policy"] for row in screen["rows"]})
        for policy in policies:
            outputs = [item for row in screen["rows"] if row["policy"] == policy for item in row["outputs"]]
            summary = {"policy": policy, "preparations": len(outputs),
                       "pdb_variants": len({item["prepared_sha256"] for item in outputs}),
                       "repair_variants": len({item["stages_sha256"]["repair_nm"] for item in outputs}),
                       "hydrogen_variants": len({item["stages_sha256"]["hydrogens_nm"] for item in outputs})}
            screen_summaries.append(summary)
            variants = {}
            for item in outputs:
                # Container paths are translated without touching the saved originals.
                relative = Path(item["directory"]).relative_to("/study")
                variants.setdefault(item["stages_sha256"]["hydrogens_nm"], root / relative)
            representatives[policy] = list(variants.values())
    for first, second in itertools.combinations_with_replacement(representatives, 2):
        values = []
        for a in representatives[first]:
            for b in representatives[second]:
                aa, bb = np.load(a / "stages.npz"), np.load(b / "stages.npz")
                assert json.loads((a/"identities.json").read_text()) == json.loads((b/"identities.json").read_text())
                for key in ("repair_nm", "hydrogens_nm"):
                    delta = (aa[key] - bb[key]) * 10
                    values.append({"stage": key, "max_displacement_angstrom": float(np.linalg.norm(delta,axis=1).max()),
                                   "rms_displacement_angstrom": float(np.sqrt(np.mean(np.sum(delta*delta,axis=1))))})
        float_comparisons.append({"first": first, "second": second, "variants": values})
    result = {"completed": True, "summaries": summaries, "comparisons": comparisons,
              "audit_comparisons": audit_comparisons, "screen_summaries": screen_summaries,
              "float_comparisons": float_comparisons,
              "aggregate": {"timed_cases":20, "included_prepared_structures":160,
                            "all_raw_model_outputs_identical":True,
                            "audit_comparison_count":len(audit_comparisons),
                            "baseline_drift_flags":sum(row["baseline_drift_flag"] for row in data["rows"]),
                            "max_abs_drift_gib":max(abs(row["post_idle_drift_bytes"]) for row in data["rows"])/2**30,
                            "swap_change_bytes":max(row["swap_change_bytes"] for row in data["rows"])}}
    (root / "analysis_261026.json").write_text(json.dumps(result,indent=2)+"\n")
    for summary in summaries:
        print(summary)
    print("SCREEN",screen_summaries)
    print("AGGREGATE",result["aggregate"])


if __name__ == "__main__":
    main()
