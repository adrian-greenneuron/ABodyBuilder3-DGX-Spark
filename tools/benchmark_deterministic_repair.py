"""Host-side fresh-container comparison with shared CPU/GPU RAM accounting.

Use a frozen protocol and immutable images. No process-RSS/CUDA sums, cache
clearing, model setting changes or automatic reruns on baseline drift.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

FIELDS = ["MemTotal", "MemAvailable", "MemFree", "Cached", "SReclaimable",
          "SUnreclaim", "AnonPages", "Shmem", "SwapFree", "Dirty", "PageTables"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--protocol", default="cli_protocol_261026.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.study.resolve()
    protocol = json.loads((root / args.protocol).read_text())
    inputs = root / "inputs/antibodies_8.csv"
    assert hashlib.sha256(inputs.read_bytes()).hexdigest() == protocol["input_sha256"]
    expected = list(csv.DictReader(inputs.open()))
    assert len(expected) == 8
    measurements = root / "cli_measurements_261026.json"
    records = json.loads(measurements.read_text())["rows"] if args.resume else []
    completed = {row["label"] for row in records}
    for case in protocol["cases"]:
        if case["label"] in completed:
            continue
        output = root / "cli" / case["label"]
        output.mkdir()
        (output / "pdbs").mkdir()
        samples = []
        stop = threading.Event()
        phase = "pre"
        errors = []

        def observe():
            try:
                with (output / "system_memory_trace_261026.csv").open("w") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["monotonic", "phase", *FIELDS])
                    while not stop.is_set():
                        mem = {key: int(value.split()[0]) * 1024
                               for line in Path("/proc/meminfo").read_text().splitlines()
                               for key, value in [line.split(":", 1)]}
                        row = {"monotonic": time.monotonic(), "phase": phase,
                               **{key: mem[key] for key in FIELDS}}
                        samples.append(row)
                        writer.writerow([row["monotonic"], phase, *[mem[key] for key in FIELDS]])
                        stop.wait(.02)
            except Exception as exc:
                errors.append(repr(exc))

        thread = threading.Thread(target=observe, daemon=True)
        thread.start()
        time.sleep(5)
        command = ["docker", "run", "--rm", "--network", "none", "--gpus", "all", "--ipc=host",
                   "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                   "-e", "PYTHONHASHSEED=0", "-e", "OPENMM_CPU_THREADS=1", "-e", "OMP_NUM_THREADS=1",
                   "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                   "-e", f"AB3_EXPERIMENTAL_REPAIR={case['policy']}",
                   "-v", f"{root}/inputs:/inputs:ro", "-v", f"{output}:/results",
                   case["image_id"], "python", "scripts/predict.py", "predict",
                   "--csv", "/inputs/antibodies_8.csv", "--output", "/results/pdbs",
                   "--model", "plddt", "--seed", "42", "--quiet", "--precision", "strict",
                   "--add-hydrogens", "--ph", "7", "--batch-report", "/results/batching.json"]
        if case["condition"] != "reference_serial":
            command += ["--preparation-workers", str(case["workers"])]
        if case["route"] == "reduce":
            command += ["--hydrogen-method", "reduce", "--sidechain-flips"]
        (output / "command_261026.json").write_text(json.dumps(command, indent=2) + "\n")
        print("START", case["label"], flush=True)
        phase = "job"
        started = time.perf_counter()
        try:
            with (output / "inference.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - started
            phase = "post"
            time.sleep(8)
        finally:
            stop.set()
            thread.join()
        assert not errors, errors
        if result.returncode:
            raise RuntimeError(f"{case['label']} failed: {result.returncode}; inspect its retained log")
        before = [row for row in samples if row["phase"] == "pre"]
        during = [row for row in samples if row["phase"] == "job"]
        after = [row for row in samples if row["phase"] == "post"]
        baseline = statistics.median(row["MemAvailable"] for row in before)
        post = statistics.median(row["MemAvailable"] for row in after[-100:])
        minimum = min(row["MemAvailable"] for row in during)
        paths = output / "pdbs"
        assert {p.name for p in paths.glob("*.pdb")} == {
            row["name"] + suffix for row in expected for suffix in (".pdb", ".prepared.pdb")}
        audits = []
        hashes = {}
        for row in expected:
            for suffix in (".pdb", ".prepared.pdb"):
                path = paths / (row["name"] + suffix)
                content = path.read_bytes()
                assert content.endswith(b"END\n")
                hashes[path.name] = hashlib.sha256(content).hexdigest()
                atoms = [line for line in content.decode().splitlines() if line.startswith("ATOM  ")]
                assert atoms and all(math.isfinite(float(line[k:k+8])) for line in atoms for k in (30, 38, 46))
            audit = json.loads((paths / (row["name"] + ".preparation.json")).read_text())
            assert audit["status"] in ("completed", "completed_with_warnings")
            changes = audit["preparation_changes"]
            if case["route"] == "reduce":
                assert audit["options"]["sidechain_flips"]
                assert all(item["atom"][-1] not in ("N", "CA", "C", "O")
                           for item in changes["moved_heavy_atoms"])
            else:
                assert changes["max_original_heavy_displacement_angstrom"] <= 1e-5
            audits.append({"name": row["name"], "seconds": audit["seconds"],
                           "repair": audit["repair"], "repair_runtime": audit.get("repair_runtime"),
                           "geometry": audit["prepared_geometry"], "warnings": audit["warnings"]})
        batching = json.loads((output / "batching.json").read_text())
        assert batching["completed"] and batching["saved"] == 8 and not batching["oom_retries"]
        record = {**case, "complete_container_seconds": elapsed,
                  "baseline_available_bytes": baseline, "min_job_available_bytes": minimum,
                  "peak_additional_unavailable_bytes": baseline - minimum,
                  "post_idle_drift_bytes": post - baseline,
                  "baseline_drift_flag": abs(post-baseline) > .25 * 2**30,
                  "ram_sensitivity_bytes": sorted([baseline-minimum, post-minimum]),
                  "pre_idle_range_bytes": max(row["MemAvailable"] for row in before) - min(row["MemAvailable"] for row in before),
                  "swap_change_bytes": before[0]["SwapFree"] - min(row["SwapFree"] for row in during),
                  "outputs": hashes, "audits": audits, "exit_code": result.returncode}
        records.append(record)
        (root / "cli_measurements_261026.json").write_text(json.dumps(
            {"completed": len(records) == len(protocol["cases"]), "protocol": protocol, "rows": records},
            indent=2) + "\n")
        print("DONE", case["label"], "seconds", round(elapsed, 3), "RAM GiB",
              round((baseline-minimum)/2**30, 3), "idle drift GiB", round((post-baseline)/2**30, 3), flush=True)


if __name__ == "__main__":
    main()
