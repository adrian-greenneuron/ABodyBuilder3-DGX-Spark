"""Check installed rescue against the retained six-case qualification panel.

Inputs and earlier outputs stay outside Git. This runs actual isolated workers;
it does not run neural inference. Use the saved panel from 27 October 2026.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, "/opt/abodybuilder3/scripts")
from preparation_pool import prepare_paths
from structure_preparation import Options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rescue", choices=["local", "restrained"], default="local")
    parser.add_argument("--route", choices=["openmm", "reduce"], default="openmm")
    args = parser.parse_args()
    cases = json.loads((args.source / "protocol_271026.json").read_text())["inputs"]
    args.output.mkdir()
    paths = []
    for case in cases:
        path = args.output / (case["target"] + ".pdb")
        shutil.copy2(args.source / "inputs" / path.name, path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["raw_sha256"]
        paths.append(path)
    # Exercise the actual default (not an explicit "local" override).
    options = Options(seed=42, ph=7., hydrogen_method=args.route,
                      sidechain_flips=args.route == "reduce",
                      rescue=None if args.rescue == "local" else "restrained")
    records = prepare_paths(paths, options, 3)
    assert len(records) == len(cases) == 6
    rows = []
    for case, path in zip(cases, paths):
        report = json.loads(path.with_suffix(".preparation.json").read_text())
        assert report["rescue"]["policy"] == args.rescue
        assert report["rescue"]["applied"] == (case["group"] == "rejected")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["raw_sha256"]
        prepared = path.with_suffix(".prepared.pdb")
        previous = (args.source / "qualification" / f"{args.route}_{args.rescue}_run1" /
                    "preparation" / case["target"] / "rescued.prepared.pdb")
        identical = prepared.read_bytes() == previous.read_bytes()
        if case["group"] == "compatible_control" or args.rescue == "local":
            assert identical, ("Qualified prepared output changed", case["target"])
        if report["rescue"]["applied"]:
            assert report["rescue"]["geometry_gate"]["passed"]
        rows.append({"group": case["group"], "rescue_applied": report["rescue"]["applied"],
                     "raw_sha256": case["raw_sha256"],
                     "prepared_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
                     "identical_to_qualified_output": identical, "status": report["status"]})
    summary = {"completed": True, "route": args.route, "policy": args.rescue, "rows": rows}
    (args.output / "integration_271026.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASSED {args.route}/{args.rescue}: six prepared, exactly three rescues; controls unchanged.", flush=True)


if __name__ == "__main__":
    main()
