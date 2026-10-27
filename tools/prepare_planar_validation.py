"""Freeze matched planar qualification cohorts, retaining every rejection.

Uses existing native heavy/light pairs. No sequences are generated or re-paired.
The 509 compatible cases are conditional validation, not a 100% success claim.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


OLD = "sha256:3b13290a647b29766a8d0c65689fbbe0dea40bdf600318a9b26dcc0bd6ce2ce4"
NEW = "sha256:ef13b76eb63cedfd4cb089b0f03511e81d4968070dd52e3207e0c968c1e39b23"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    preflight = json.loads((root / "planar_preflight_271026.json").read_text())
    assert preflight["completed"] and len(preflight["rows"]) == 512
    passed = {(r["stage"], r["target"]): r["passed"] for r in preflight["rows"]}
    cohorts = {"planar127": [], "planar382": [], "planar_rejections": []}
    provenance = []
    seen = set()
    for stage, filename in (("tuned128", "antibodies_128.csv"), ("validation256", "validation_256.csv"), ("fresh128", "fresh_128.csv")):
        source = root / stage / "inputs" / filename
        for row in csv.DictReader(source.open()):
            original_name = row["name"]
            key = tuple(row[k] for k in row if k != "name")
            assert key not in seen
            seen.add(key)
            compatible = passed[(stage, original_name)]
            cohort = ("planar127" if stage == "tuned128" else "planar382") if compatible else "planar_rejections"
            row["name"] = stage + "_" + original_name
            cohorts[cohort].append(row)
            provenance.append({"stage": stage, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                               "original_target": original_name, "target": row["name"], "cohort": cohort,
                               "passed": compatible})
    assert [len(cohorts[k]) for k in cohorts] == [127, 382, 3]
    catalog = {}
    for cohort, rows in cohorts.items():
        target = root / cohort
        (target / "inputs").mkdir(parents=True)
        (target / "cli").mkdir()
        filename = cohort + ".csv"
        path = target / "inputs" / filename
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        catalog[cohort] = {"count": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if cohort == "planar_rejections":
            continue
        configs = [("cpu_planar20", "cpu1", 20, NEW, "robust_planar"), ("gpu_planar12", "gpu_mixed", 12, NEW, "robust_planar")]
        if cohort == "planar127":
            configs = [("cpu_original20", "cpu1", 20, OLD, "off"), ("gpu_original20", "gpu_mixed", 20, OLD, "off"),
                       configs[0], ("gpu_planar8", "gpu_mixed", 8, NEW, "robust_planar"), configs[1]]
        cases = []
        for replicate in range(1, 3 if cohort == "planar127" else 2):
            combinations = [(route, config) for route in ("openmm", "reduce") for config in configs]
            if replicate == 2:
                combinations.reverse()
            for route, (condition, hydrogen, workers, image, geometry) in combinations:
                cases.append({"label": f"{route}_{condition}_r{replicate}", "condition": condition,
                              "input": filename, "route": route, "policy": "cpu1", "hydrogen": hydrogen,
                              "workers": workers, "schedule": "phased", "geometry": geometry,
                              "gradient_tolerance": 5.0, "replicate": replicate, "image_id": image})
        (target / "cli_protocol_271026.json").write_text(json.dumps({
            "protocol": "Matched compatible subset after explicit preflight. Rejections retained separately; report overall coverage 509/512. No model, raw-prediction, main or dev changes. Paired full-container runtime and system MemAvailable measurements.",
            "inputs": {filename: catalog[cohort]}, "cases": cases}, indent=2) + "\n")
    (root / "planar_cohort_manifest_271026.json").write_text(json.dumps({"completed": True, "cohorts": catalog,
        "scope": "All 512 original pairs retained: 509 compatible and three explicit heavy-geometry rejections. Performance estimates condition on compatibility.",
        "rows": provenance}, indent=2) + "\n")


if __name__ == "__main__":
    main()
