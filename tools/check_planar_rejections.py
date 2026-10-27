"""Exercise real worker failures without losing raw inputs or independent jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, "/opt/abodybuilder3/scripts")
from preparation_pool import prepare_paths
from structure_preparation import Options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--route", choices=("openmm", "reduce"), required=True)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    engine = os.environ["AB3_EXPERIMENTAL_HYDROGEN"]
    out = args.root / "planar_rejections" / (f"{args.route}_{engine}" + ("_" + args.output_tag if args.output_tag else ""))
    out.mkdir()
    manifest = json.loads((args.root / "planar_cohort_manifest_271026.json").read_text())
    records = [r for r in manifest["rows"] if not r["passed"]]
    cases = {"tuned128": "openmm_gpu_fixed12_gt5_r1", "validation256": "openmm_gpu_fixed8_r1", "fresh128": "openmm_gpu_fixed12_gt5_r1"}
    paths = []
    for row in records:
        source = args.root / row["stage"] / "cli" / cases[row["stage"]] / "pdbs" / (row["original_target"] + ".pdb")
        target = out / (row["target"] + ".pdb")
        target.write_bytes(source.read_bytes())
        paths.append(target)
    good = out / "independent_good.pdb"
    good.write_bytes((args.root / "planar_smoke6/cli/openmm_gpu_planar_r1/pdbs/antibody_000000.pdb").read_bytes())
    paths.append(good)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    try:
        prepare_paths(paths, Options(hydrogen_method=args.route, sidechain_flips=args.route == "reduce", ph=7, seed=42), 4)
    except RuntimeError as exc:
        assert "failed for 3" in str(exc), str(exc)
    else:
        raise AssertionError("Incompatible frozen-heavy structures were accepted")
    rows = []
    for path in paths:
        audit = json.loads(path.with_suffix(".preparation.json").read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path.name]
        expected_failure = path != good
        assert (audit["status"] == "failed") == expected_failure
        assert path.with_suffix(".prepared.pdb").exists() != expected_failure
        if expected_failure:
            assert "heavy-atom repair" in json.dumps(audit)
        rows.append({"target": path.stem, "expected_failure": expected_failure, "status": audit["status"], "raw_sha256": hashes[path.name]})
    (out / "rejection_checks_271026.json").write_text(json.dumps({"completed": True, "route": args.route, "engine": engine,
        "all_raw_files_unchanged": True, "independent_good_completed": True, "rows": rows}, indent=2) + "\n")
    print("REJECTIONS PASSED", args.route, engine, flush=True)


if __name__ == "__main__":
    main()
