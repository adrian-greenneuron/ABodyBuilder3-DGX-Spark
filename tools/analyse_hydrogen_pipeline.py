"""Summarise frozen full-pipeline runs without changing any structure.

Only aggregate counts and generic target IDs leave the local study. PDB date
headers, atom coordinates and raw neural predictions are compared separately.
"""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

import numpy as np

STUDIES = ["engine", "smoke", "pipeline64", "pipeline64_followup", "confirmation128"]
METRICS = ["H_bond_errors_over_0_10_A", "H_angle_errors_over_30_deg",
           "alpha_H_wrong_side", "severe_H_clashes"]


def load(path):
    return json.loads(path.read_text())


def pdb_atoms(path):
    atoms = {}
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            key = (line[21:22], line[22:26], line[26:27], line[17:20],
                   line[12:16].strip(), line[16:17])
            assert key not in atoms
            atoms[key] = (np.array([float(line[k:k+8]) for k in (30, 38, 46)]),
                          line[76:78].strip())
    return atoms


def without_date(data):
    return re.sub(rb"(?m)^(REMARK   1 CREATED WITH OPENMM [^\n]*), \d{4}-\d{2}-\d{2}$",
                  rb"\1, <DATE>", data)


def compare(a, b):
    left, right = a.read_bytes(), b.read_bytes()
    aa, bb = pdb_atoms(a), pdb_atoms(b)
    common = aa.keys() & bb.keys()
    def delta(keys):
        values = [float(np.linalg.norm(aa[k][0]-bb[k][0])) for k in keys]
        return {"max_A": max(values, default=0),
                "rms_A": float(np.sqrt(np.mean(np.square(values)))) if values else 0}
    h = [k for k in common if aa[k][1] == "H"]
    heavy = [k for k in common if aa[k][1] != "H"]
    atom_records = lambda data: [s for s in data.splitlines() if s.startswith((b"ATOM  ", b"HETATM"))]
    return {"bytes_identical": left == right,
            "date_only_difference": left != right and without_date(left) == without_date(right),
            "atom_records_identical": atom_records(left) == atom_records(right),
            "same_atom_keys": aa.keys() == bb.keys(),
            "coordinate_identical": aa.keys() == bb.keys() and all(np.array_equal(aa[k][0], bb[k][0]) for k in aa),
            "hydrogens": delta(h), "heavy_atoms": delta(heavy)}


def counts(records):
    fields = ["bytes_identical", "date_only_difference", "atom_records_identical",
              "same_atom_keys", "coordinate_identical"]
    return {"targets": len(records), **{k: sum(r[k] for r in records) for k in fields},
            "max_H_displacement_A": max((r["hydrogens"]["max_A"] for r in records), default=0),
            "max_heavy_displacement_A": max((r["heavy_atoms"]["max_A"] for r in records), default=0)}


def defect_sets(detail):
    def atom_key(row):
        return tuple(tuple(x) for x in row["atoms"])
    return {
        METRICS[0]: {atom_key(x) for x in detail["bad_bonds"] if abs(x["error_angstrom"]) > .1},
        METRICS[1]: {atom_key(x) for x in detail["bad_angles"] if abs(x["error_degrees"]) > 30},
        METRICS[2]: {tuple(x["residue"]) for x in detail["alpha_H"] if x.get("wrong_side", False)},
        METRICS[3]: {atom_key(x) for x in detail["severe_H_clashes"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    performance, summaries, repeatability, comparisons, phases, geometry = [], [], [], [], [], []
    raw_reference = {}
    for study in STUDIES:
        root = args.study / study
        measurements = load(root/"cli_measurements_261026.json")
        assert measurements["completed"]
        groups = defaultdict(list)
        cases = measurements["rows"]
        for case in cases:
            directory = root/"cli"/case["label"]
            assert case["exit_code"] == 0
            batch = load(directory/"batching.json")
            count = measurements["protocol"]["inputs"][case["input"]]["count"]
            assert batch["completed"] and batch["saved"] == count and not batch["oom_retries"]
            p = {k: case[k] for k in ["label", "route", "condition", "workers", "schedule", "hydrogen", "replicate", "image_id",
                 "complete_container_seconds", "baseline_available_bytes", "min_job_available_bytes",
                 "peak_additional_unavailable_bytes", "post_idle_drift_bytes", "baseline_drift_flag",
                 "ram_sensitivity_bytes", "swap_change_bytes"]}
            p.update(study=study, count=count, host_cpu_busy_fraction=case.get("host_cpu_busy_fraction"))
            p["structures_per_minute"] = 60*count/p["complete_container_seconds"]
            performance.append(p)
            groups[(case["route"], case["condition"])].append(p)
            paths = directory/"pdbs"
            raw = sorted(name for name in case["outputs"] if not name.endswith(".prepared.pdb"))
            identities = batch.get("deployment", {})
            assert identities.get("precision") == "strict_fp32"
            # Same target within a panel, irrespective of route, workers or replicate.
            raw_mismatches = 0
            for name in raw:
                key = (count, name)
                digest = hashlib.sha256((paths/name).read_bytes()).hexdigest()
                raw_reference.setdefault(key, digest)
                raw_mismatches += digest != raw_reference[key]
            p["raw_prediction_mismatches"] = raw_mismatches
            p["deployment"] = identities
            if "phase_events" in batch:
                events = batch["phase_events"]
                first = events["first_raw_write"]
                end = events["all_preparation_finished"]
                stream = batch.get("preparation_stream")
                timing = {"study": study, "label": case["label"], "route": case["route"],
                          "condition": case["condition"], "replicate": case["replicate"],
                          "raw_write_span_seconds": events["last_raw_write"]-first,
                          "post_prediction_tail_seconds": end-events["prediction_pipeline_finished"],
                          "span_from_first_raw_seconds": end-first,
                          "events_from_first_raw_seconds": {k: v-first for k, v in events.items()}}
                if stream:
                    assert stream["accepted"] == count and stream["completed"] == count and stream["error"] is None
                    assert stream["max_inflight"] <= stream["inflight_limit"]
                    jobs = stream["results"]
                    begin = min(x["started_monotonic"] for x in jobs)
                    timing.update(first_preparation_relative_seconds=begin-first,
                                  overlap_with_raw_write_seconds=max(0, events["last_raw_write"]-begin),
                                  max_inflight=stream["max_inflight"], inflight_limit=stream["inflight_limit"],
                                  actual_workers=stream["workers"],
                                  jobs=[{"start": x["started_monotonic"]-first, "end": x["finished_monotonic"]-first,
                                         "worker_pid": x["worker_pid"]} for x in jobs])
                phases.append(timing)
        for (route, condition), rows in groups.items():
            times = [x["complete_container_seconds"] for x in rows]
            summaries.append({"study": study, "route": route, "condition": condition,
                              "count": rows[0]["count"], "workers": rows[0]["workers"],
                              "schedule": rows[0]["schedule"], "hydrogen": rows[0]["hydrogen"],
                              "repetitions": len(rows), "times_seconds": times,
                              "median_seconds": statistics.median(times),
                              "structures_per_minute": 60*rows[0]["count"]/statistics.median(times),
                              "peak_additional_ram_gib": max(x["peak_additional_unavailable_bytes"] for x in rows)/2**30,
                              "ram_repeats_gib": [x["peak_additional_unavailable_bytes"]/2**30 for x in rows],
                              "ram_sensitivity_gib": [min(x["ram_sensitivity_bytes"][0] for x in rows)/2**30,
                                                      max(x["ram_sensitivity_bytes"][1] for x in rows)/2**30],
                              "drift_flags": sum(x["baseline_drift_flag"] for x in rows),
                              "host_cpu_busy_fraction": statistics.mean(x["host_cpu_busy_fraction"] for x in rows)
                              if rows[0]["host_cpu_busy_fraction"] is not None else None})
            if len(rows) == 2:
                a = root/"cli"/rows[0]["label"]/"pdbs"
                b = root/"cli"/rows[1]["label"]/"pdbs"
                details = [{"target": path.stem.removesuffix(".prepared"), **compare(path, b/path.name)}
                           for path in sorted(a.glob("*.prepared.pdb"))]
                repeatability.append({"study": study, "route": route, "condition": condition,
                                      **counts(details), "details": details})
        # Compare first repetitions to their own panel/route CPU control.
        controls = {}
        for case in cases:
            if case["replicate"] == 1 and case["hydrogen"] == "cpu1" and case["schedule"] == "phased":
                controls.setdefault(case["route"], case)
        for case in cases:
            if case["replicate"] != 1 or case["route"] not in controls:
                continue
            control = controls[case["route"]]
            a = root/"cli"/control["label"]/"pdbs"
            b = root/"cli"/case["label"]/"pdbs"
            details = []
            for path in sorted(a.glob("*.prepared.pdb")):
                target = path.stem.removesuffix(".prepared")
                audit_a = load(a/(target+".preparation.json"))
                audit_b = load(b/(target+".preparation.json"))
                details.append({"target": target, **compare(path, b/path.name),
                                "protonation_identical": audit_a["protonation_variants"] == audit_b["protonation_variants"],
                                "warnings_identical": audit_a["warnings"] == audit_b["warnings"],
                                "original_heavy_displacement_A": audit_b["preparation_changes"]["max_original_heavy_displacement_angstrom"]})
            comparisons.append({"study": study, "route": case["route"], "condition": case["condition"],
                                "control": control["condition"], **counts(details),
                                "protonation_identical": sum(x["protonation_identical"] for x in details),
                                "details": details})
        geo_path = root/"chemical_geometry_261026.json"
        if geo_path.exists():
            geo = load(geo_path)["rows"]
            by_case = defaultdict(dict)
            for row in geo:
                by_case[(row["route"], row["condition"])][row["target"]] = row
            for (route, condition), targets in by_case.items():
                control = controls[route]["condition"]
                baseline = by_case[(route, control)]
                paired = []
                unassessable = eligible = 0
                for target, row in targets.items():
                    ref = baseline[target]
                    detail_a = load(root/"cli"/ref["label"]/"pdbs"/(target.removesuffix(".pdb")+".geometry.json"))
                    detail_b = load(root/"cli"/row["label"]/"pdbs"/(target.removesuffix(".pdb")+".geometry.json"))
                    defects_a, defects_b = defect_sets(detail_a), defect_sets(detail_b)
                    eligible += len(detail_b["alpha_H"])
                    unassessable += sum(not x["assessable"] for x in detail_b["alpha_H"])
                    paired.append({"target": target, "baseline": {k: ref[k] for k in METRICS},
                                   "candidate": {k: row[k] for k in METRICS},
                                   "delta": {k: row[k]-ref[k] for k in METRICS},
                                   "new_defects": {k: len(defects_b[k]-defects_a[k]) for k in METRICS},
                                   "resolved_defects": {k: len(defects_a[k]-defects_b[k]) for k in METRICS}})
                geometry.append({"study": study, "route": route, "condition": condition, "count": len(targets),
                                 "totals": {k: sum(x[k] for x in targets.values()) for k in METRICS},
                                 "targets_with_flags": {k: sum(x[k] > 0 for x in targets.values()) for k in METRICS},
                                 "worse_equal_better": {k: {"worse": sum(x["delta"][k] > 0 for x in paired),
                                                           "equal": sum(x["delta"][k] == 0 for x in paired),
                                                           "better": sum(x["delta"][k] < 0 for x in paired)} for k in METRICS},
                                 "max_H_bond_error_A": max(x["max_H_bond_error_A"] for x in targets.values()),
                                 "max_H_angle_error_deg": max(x["max_H_angle_error_deg"] for x in targets.values()),
                                 "alpha_H_eligible": eligible, "alpha_H_unassessable": unassessable,
                                 "alpha_H_near_plane": sum(x["alpha_H_near_plane"] for x in targets.values()),
                                 "H_bonds": sum(x["H_bonds"] for x in targets.values()),
                                 "H_angles": sum(x["H_angles"] for x in targets.values()),
                                 "new_defects": {k: sum(x["new_defects"][k] for x in paired) for k in METRICS},
                                 "resolved_defects": {k: sum(x["resolved_defects"][k] for x in paired) for k in METRICS},
                                 "paired": paired})
    data = {"generation_date": "2026-10-27", "performance": performance, "summaries": summaries,
            "repeatability": repeatability, "comparisons": comparisons, "phases": phases, "geometry": geometry,
            "geometry_note": "Counts are screening diagnostics on fixed PDB-rounded coordinates, not experimental structural accuracy or a validated clashscore. No minimisation or output rewriting."}
    assert not any(x["raw_prediction_mismatches"] for x in performance)
    assert len({x["deployment"]["checkpoint_sha256"] for x in performance}) == 1
    assert len({x["deployment"]["manifest_sha256"] for x in performance}) == 1
    for kind in ("single", "multi"):
        assert len({x["deployment"]["packages"][kind]["sha256"] for x in performance}) == 1
    (args.output_dir/"hydrogen_pipeline_analysis_271026.json").write_text(json.dumps(data, indent=2)+"\n")
    columns = ["study", "count", "route", "condition", "workers", "schedule", "median_seconds", "structures_per_minute", "peak_additional_ram_gib", "drift_flags"]
    with (args.output_dir/"hydrogen_pipeline_summary_271026.csv").open("w") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore", lineterminator="\n"); writer.writeheader(); writer.writerows(summaries)
    print("Analysed", len(performance), "complete jobs; raw predictions match in every case.")
    for row in summaries:
        if row["study"] == "confirmation128":
            print(row["route"], row["condition"], round(row["median_seconds"], 3), "s", round(row["peak_additional_ram_gib"], 3), "GiB")


if __name__ == "__main__":
    main()
