"""Package compact, sequence-free protocols and telemetry for the experiment."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

STUDIES = ["engine", "smoke", "pipeline64", "pipeline64_followup", "confirmation128"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--component", type=Path, required=True)
    args = parser.parse_args(); root, component = args.study, args.component
    parent = subprocess.check_output(["git", "show", "619d637:candidate/structure_preparation.py"], cwd=component)
    (component/"reference/deterministic_structure_preparation.py").write_bytes(parent)
    for name, before, after in [
        ("gpu_hydrogen", "reference/openmm_modeller.py", "candidate/openmm_modeller.py"),
        ("hydrogen_wrapper", "reference/deterministic_structure_preparation.py", "candidate/structure_preparation.py"),
        ("streaming_cli", "reference/predict.py", "candidate/predict.py"),
        ("preparation_pool", "reference/preparation_pool.py", "candidate/preparation_pool.py")]:
        a, b = component/before, component/after
        text = "".join(difflib.unified_diff(a.read_text().splitlines(True), b.read_text().splitlines(True), fromfile=before, tofile=after))
        (component/"candidate"/(name+"_271026.patch")).write_text(text)
    source = {}
    for directory in ["reference", "candidate", "tests", "tools"]:
        for path in sorted((component/directory).glob("*.py")):
            source[str(path.relative_to(component))] = digest(path)
    for path in component.glob("Dockerfile*"): source[path.name] = digest(path)
    manifest = {"generation_date": "2026-10-27", "sha256": source,
                "source_note": "Original OpenMM Modeller preserves its MIT notice. PDBFixer extension inherited from the prior deterministic component. Only addHydrogens accepts optional Context properties in the Modeller patch."}
    manifest_path = component/"docs/data/hydrogen_pipeline_sources_271026.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    files = [p for p in root.glob("*.log")]
    files += [root/"protocol_261026.json"]
    for study in STUDIES:
        base = root/study
        files += [base/"cli_protocol_261026.json"]
        files += list((base/"tools").glob("*.py"))
        selection = base/"inputs/selection_manifest_261026.json"
        if selection.exists(): files.append(selection)
        for case in (base/"cli").iterdir():
            for name in ["command_261026.json", "execution.json", "system_memory_trace_261026.csv", "gpu_utilisation.csv", "batching.json"]:
                p = case/name
                if p.exists(): files.append(p)
    archive = component/"docs/data/hydrogen_pipeline_evidence_271026.zip"
    hashes = {}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for path in sorted(set(files)):
            data = path.read_bytes(); name = str(path.relative_to(root))
            hashes[name] = hashlib.sha256(data).hexdigest(); stream.writestr(name, data)
        stream.writestr("source_manifest.json", manifest_path.read_bytes())
        stream.writestr("SHA256.json", json.dumps(hashes, indent=2)+"\n")
        stream.writestr("README.txt", "65 complete pipeline jobs; protocols, commands, telemetry, batch reports and test/controller logs.\n"
                        "No biological input CSVs, PDBs, complete atom arrays, per-residue geometry lists or checkpoints.\n"
                        "Input identities are recorded by SHA-256 and aggregate counts. Main and dev defaults remain unchanged.\n")
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
        assert not any(Path(n).suffix in (".pdb", ".npz", ".ckpt") for n in stream.namelist())
        assert not any(n.endswith("antibodies_64.csv") or n.endswith("antibodies_128.csv") or n.endswith("antibodies_8.csv") for n in stream.namelist())
        for name, value in hashes.items(): assert hashlib.sha256(stream.read(name)).hexdigest() == value
    report_files = [p for p in (component/"docs/data").glob("hydrogen_pipeline_*") if "checksums" not in p.name]
    report_files += list((component/"docs/assets").glob("hydrogen_pipeline_*"))
    report_files += [component/"docs/hydrogen_pipeline_271026.md"]
    (component/"docs/data/hydrogen_pipeline_checksums_271026.txt").write_text("".join(
        f"{digest(p)}  {p.relative_to(component/'docs')}\n" for p in sorted(report_files)))
    print("Packaged", len(hashes), "evidence files; ZIP bytes", archive.stat().st_size)


if __name__ == "__main__": main()
