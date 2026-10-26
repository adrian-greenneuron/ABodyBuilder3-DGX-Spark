"""A real worker failure must preserve its audit, raw files and other jobs."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, "/opt/abodybuilder3/scripts")
from preparation_pool import prepare_paths
from structure_preparation import Options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output; root.mkdir(parents=True)
    raw = args.raw.read_text()
    (root / "good.pdb").write_text(raw)
    # Duplicate PDB atom lines may be merged by the parser. An already
    # hydrogenated structure is explicitly rejected by the preparation contract.
    bad = args.raw.with_suffix(".prepared.pdb").read_text()
    (root / "bad.pdb").write_text(bad)
    try:
        prepare_paths([root / "bad.pdb", root / "good.pdb"], Options(seed=42), 2)
    except RuntimeError as exc:
        assert "failed for 1" in str(exc), str(exc)
    else:
        raise AssertionError("Failed chemistry was reported as success")
    assert (root / "good.prepared.pdb").exists()
    assert not (root / "bad.prepared.pdb").exists()
    assert json.loads((root / "bad.preparation.json").read_text())["status"] == "failed"
    assert (root / "good.pdb").read_text() == raw
    assert (root / "bad.pdb").read_text() == bad
    (root / "validation_261026.json").write_text(json.dumps({"completed": True,
        "failed_record_retained": True, "independent_job_completed": True,
        "raw_files_unchanged": True}, indent=2) + "\n")
    print("Real worker failure retention passed", flush=True)


if __name__ == "__main__":
    main()
