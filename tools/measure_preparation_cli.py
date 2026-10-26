"""Installed-CLI comparison including prediction and optional preparation."""
import time
START = time.perf_counter()
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch
from benchmark_runtime import Memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["openmm", "reduce"], required=True)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--preparation-workers", default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    (args.output / "pdbs").mkdir()
    sys.path.insert(0, "/opt/abodybuilder3/scripts")
    spec = importlib.util.spec_from_file_location("preparation_cli", "/opt/abodybuilder3/scripts/predict.py")
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    sys.argv = ["predict.py", "predict", "--csv", str(args.input), "--output", str(args.output / "pdbs"),
                "--model", "plddt", "--workers", "1", "--seed", "42", "--precision", "strict",
                "--add-hydrogens", "--ph", "7", "--quiet"]
    if args.mode == "reduce":
        sys.argv += ["--hydrogen-method", "reduce", "--sidechain-flips"]
    if args.eager:
        sys.argv += ["--execution", "eager", "--batch-size", "1"]
    else:
        sys.argv += ["--batch-report", str(args.output / "batching.json")]
    if args.preparation_workers is not None:
        sys.argv += ["--preparation-workers", args.preparation_workers]
    with Memory() as memory:
        try:
            cli.app()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        end = time.perf_counter()
    records = list((args.output / "pdbs").glob("*.preparation.json"))
    result = {"completed": True, "candidate": args.candidate, "eager": args.eager,
              "mode": args.mode, "script_to_cli_complete_seconds": end - START,
              "preparation_records": len(records),
              "peak_pytorch_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "cuda_scope": "PyTorch inference allocation only; OpenMM allocations are not included.",
              **memory.result()}
    (args.output / "metrics_261026.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
