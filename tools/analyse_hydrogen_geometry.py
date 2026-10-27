"""Paired, sequence-free analysis of the final hydrogen geometry experiment."""
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import statistics
import sys

import numpy as np

from match_hydrogen_pipeline import load


def coordinates(left, right):
    a, groups_a = load(left)
    b, groups_b = load(right)
    assert a.keys() == b.keys() and groups_a.keys() == groups_b.keys()
    hs = set(itertools.chain.from_iterable(groups_a.values()))
    heavy = [k for k in a if k not in hs]
    heavy_max = max(np.linalg.norm(a[k]-b[k]) for k in heavy)
    named, matched = [], []
    for parent, members in groups_a.items():
        assert set(members) == set(groups_b[parent])
        aa = np.array([a[k] for k in members]); bb = np.array([b[k] for k in members])
        named.extend(np.linalg.norm(aa-bb, axis=1))
        order = min(itertools.permutations(range(len(members))), key=lambda p: float(np.sum((aa-bb[list(p)])**2)))
        matched.extend(np.linalg.norm(aa-bb[list(order)], axis=1))
    return {"max_heavy_displacement_A": float(heavy_max), "H_count": len(named),
            "H_rms_by_name_A": float(np.sqrt(np.mean(np.square(named)))),
            "H_rms_after_matching_A": float(np.sqrt(np.mean(np.square(matched)))),
            "max_H_after_matching_A": float(max(matched)),
            "H_displacements_over_0_1_A": int(np.sum(np.array(matched) > .1)),
            "H_displacements_over_0_5_A": int(np.sum(np.array(matched) > .5)),
            "prepared_bytes_identical": left.read_bytes() == right.read_bytes()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); root = args.root
    selected = []
    for stage in ("planar127", "adaptive127", "adaptive382", "adaptive382_followup"):
        data = json.loads((root/stage/"cli_measurements_271026.json").read_text())
        for row in data["rows"]:
            if stage == "planar127" and row["geometry"] != "off":
                continue
            assert row["exit_code"] == 0
            selected.append({**row, "stage": stage, "cohort": "127" if stage.endswith("127") else "382"})
    assert len(selected) == 20
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["cohort"], row["route"], row["condition"]].append(row)
    summaries = []
    for (cohort, route, condition), rows in grouped.items():
        times = [r["complete_container_seconds"] for r in rows]
        summaries.append({"cohort": cohort, "count": int(cohort), "route": route, "condition": condition,
                          "workers": rows[0]["workers"], "n": len(rows), "seconds": times,
                          "median_seconds": statistics.median(times),
                          "structures_per_minute": int(cohort)*60/statistics.median(times),
                          "peak_additional_ram_gib": max(r["peak_additional_unavailable_bytes"] for r in rows)/2**30,
                          "ram_sensitivity_gib": [min(min(r["ram_sensitivity_bytes"]) for r in rows)/2**30,
                                                  max(max(r["ram_sensitivity_bytes"]) for r in rows)/2**30],
                          "baseline_drift_flags": sum(r["baseline_drift_flag"] for r in rows),
                          "max_swap_change_bytes": max(r["swap_change_bytes"] for r in rows),
                          "images": sorted(set(r["image_id"] for r in rows)),
                          "records": [r["stage"]+"/"+r["label"] for r in rows]})
    raw_checks = []
    for cohort in ("127", "382"):
        rows = [r for r in selected if r["cohort"] == cohort]
        reference = {k: v for k, v in rows[0]["outputs"].items() if not k.endswith(".prepared.pdb")}
        for row in rows:
            raw = {k: v for k, v in row["outputs"].items() if not k.endswith(".prepared.pdb")}
            assert raw == reference, (cohort, row["label"])
        raw_checks.append({"cohort": cohort, "jobs": len(rows), "predictions": len(reference)*len(rows), "all_raw_bytes_identical": True})
    geometry = []
    for stage in ("planar127", "adaptive127", "adaptive382", "adaptive382_followup"):
        data = json.loads((root/stage/"chemical_geometry_271026.json").read_text())
        assert not data["failed"]
        geometry.extend({**r, "stage": stage} for r in data["rows"] if stage != "planar127" or "original" in r["condition"])
    probes = []
    for filename in ("probe_planar_validation_271026.json", "probe_adaptive_validation_271026.json"):
        data = json.loads((root/filename).read_text())
        probes.extend(r for r in data["rows"] if r["stage"] != "planar127" or "original" in r["condition"])
    probes = [r for r in probes if r["stage"] != "planar382"]
    checks = ["H_bond_errors_over_0_10_A", "H_angle_errors_over_30_deg", "alpha_H_wrong_side",
              "tetrahedral_H_wrong_side", "tetrahedral_H_signed_volume_inversions", "planar_H_over_1_deg", "inward_planar_H_over_1_deg"]
    geometry_summary = []
    for s in summaries:
        labels = {v.split("/", 1)[1] for v in s["records"]}
        rows = [r for r in geometry if r["label"] in labels and (r["stage"] == "planar127") == ("original" in s["condition"])]
        # Different cohort sizes have the same candidate labels; select by stage.
        stages = {v.split("/", 1)[0] for v in s["records"]}
        rows = [r for r in rows if r["stage"] in stages]
        assert len(rows) == s["count"]
        ps = [r for r in probes if r["stage"] in stages and r["label"] in labels]
        assert len(ps) == s["count"]
        geometry_summary.append({"cohort": s["cohort"], "route": s["route"], "condition": s["condition"],
                                 "structures": len(rows), "structures_with_any_checked_defect": sum(any(r[k] for k in checks) for r in rows),
                                 "counts": {k: sum(r[k] for r in rows) for k in checks},
                                 "max_H_bond_error_A": max(r["max_H_bond_error_A"] for r in rows),
                                 "max_H_angle_error_deg": max(r["max_H_angle_error_deg"] for r in rows),
                                 "max_planar_H_deviation_deg": max(r["max_planar_H_deviation_deg"] for r in rows),
                                 "probe_H_pairs": sum(r["H_clash_pairs"] for r in ps),
                                 "probe_all_pairs": sum(r["all_clash_pairs"] for r in ps)})
    differences = []
    def compare(cohort, route, left_condition, right_condition, left_rep=1, right_rep=1, label="CPU versus GPU"):
        left = next(r for r in selected if (r["cohort"],r["route"],r["condition"],r["replicate"]) == (cohort,route,left_condition,left_rep))
        right = next(r for r in selected if (r["cohort"],r["route"],r["condition"],r["replicate"]) == (cohort,route,right_condition,right_rep))
        a = root/left["stage"]/"cli"/left["label"]/"pdbs"
        b = root/right["stage"]/"cli"/right["label"]/"pdbs"
        for filename in sorted(k for k in left["outputs"] if k.endswith(".prepared.pdb")):
            metric = coordinates(a/filename, b/filename)
            assert metric["max_heavy_displacement_A"] == 0, (filename, metric)
            differences.append({"cohort": cohort, "route": route, "comparison": label, "target": filename, **metric})
        print("COMPARED", cohort, route, label, flush=True)
    for route in ("openmm", "reduce"):
        for cohort in ("127", "382"):
            compare(cohort, route, "cpu_adaptive20", "gpu_adaptive12")
        for condition in ("cpu_adaptive20", "gpu_adaptive12"):
            compare("127", route, condition, condition, 1, 2, condition+" repeat")
        compare("127", route, "gpu_original20", "gpu_adaptive12", label="Old versus corrected GPU")
        compare("127", route, "cpu_original20", "cpu_adaptive20", label="Old versus corrected CPU")
    attempts = []
    for row in selected:
        if row["geometry"] == "off" or row["replicate"] != 1:
            continue
        for audit in row["audits"]:
            opt = audit["hydrogen_optimisation"]
            attempts.append({"stage": row["stage"], "route": row["route"], "condition": row["condition"], "target": audit["name"],
                             "accepted_steric_k": opt["parameters"]["steric_k"], "attempt_count": len(opt["steric_attempts"]),
                             "optimiser_success": opt["success"], "iterations": opt["iterations"]})
    preflight = json.loads((root/"planar_preflight_271026.json").read_text())
    rejection_checks = [json.loads(p.read_text()) for p in (root/"planar_rejections").glob("*_adaptive/rejection_checks_271026.json")]
    assert len(rejection_checks) == 4 and all(r["completed"] for r in rejection_checks)
    for row in selected:
        assert not row["baseline_drift_flag"] and row["swap_change_bytes"] == 0
    result = {"completed": True, "scope": "Final accepted-coordinate candidate, not enabled on main or dev. Bounded explicit steric back-off; different chemistry objective from legacy Amber H minimisation.",
              "coverage": {"total_native_pairs": len(preflight["rows"]), "compatible": sum(r["passed"] for r in preflight["rows"]),
                           "fixed_heavy_rejections": [r for r in preflight["rows"] if not r["passed"]]},
              "summaries": summaries, "raw_identity": raw_checks, "geometry_summary": geometry_summary,
              "geometry": geometry, "probe": probes, "differences": differences, "optimisation_attempts": attempts,
              "rejection_checks": rejection_checks,
              "failed_attempts": [{"stage": "adaptive382", "label": "openmm_gpu_adaptive12_r1", "cause": "Two CUDA initialisation failures; all 382 attempts retained; whole job retried unchanged once. Not an independent replicate or a successful timing."}],
              "coordinate_comparison_note": "No alignment or coordinate changes. Same-parent H permutations are a generous diagnostic, not a claim of chemical equivalence for all labels. H RMS values describe prepared-coordinate differences, not experimental accuracy."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
