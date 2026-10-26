"""Archive compact protocols, telemetry and checks without biological payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--component",type=Path,required=True)
    args=parser.parse_args();root=args.study;component=args.component
    metadata = ["protocol_261026.json","source_manifest_261026.json",
                "cli_protocol_261026.json","cli_protocol_v2_261026.json",
                "validation_correction_261026.json","cli_controller_261026.log",
                "cli_controller_v2_261026.log","screen_261026.log","screen2_261026.log",
                "runtime_identity_261026.log","policy_tests_261026.log"]
    files=[root/name for name in metadata]
    for directory in ["screen","screen2"]:
        files.append(root/directory/"screen_measurements_261026.json")
    for path in sorted((root/"cli").iterdir()):
        files.extend(path/name for name in ["command_261026.json","system_memory_trace_261026.csv","inference.log"])
    for directory in ["frozen_tools","frozen_tools2","frozen_tools3"]:
        files.extend(sorted((root/directory).glob("*.py")))
    archive=component/"docs/data/deterministic_repair_evidence_261026.zip"
    hashes={}
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
        for path in files:
            data=path.read_bytes();name=str(path.relative_to(root));hashes[name]=hashlib.sha256(data).hexdigest()
            stream.writestr(name,data)
        stream.writestr("README.txt", "Compact deterministic-repair evidence. Includes the retained failed validation attempt.\n"
                        "Original inputs, PDB predictions, full coordinate arrays and per-residue identity lists remain local.\n"
                        "All current main/dev defaults remain unchanged. Full report/source on the component branch.\n")
        stream.writestr("SHA256.json",json.dumps(hashes,indent=2)+"\n")
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
        for name,digest in hashes.items():assert hashlib.sha256(stream.read(name)).hexdigest()==digest
        assert not any(Path(name).suffix in [".pdb",".npz"] for name in stream.namelist())
    report_files=[archive,*sorted((component/"docs/data").glob("deterministic_repair_*.json")),
                  component/"docs/data/deterministic_repair_summary_261026.csv"]
    (component/"docs/data/deterministic_repair_checksums_261026.txt").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in report_files))
    print("Evidence files:",len(files),"archive bytes:",archive.stat().st_size)


if __name__=="__main__":main()
