"""Measure preparation separately from inference on retained public predictions.

Run in the verified runtime with /scripts, /inputs and /output mounted. Each
case has one warm-up and two measurements; failures retain their partial audit.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, "/scripts")
from structure_preparation import Options, PreparationError, prepare_structure
from benchmark_runtime import Memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["base", "plddt", "language"])
    parser.add_argument("--modes", nargs="+", default=["openmm", "reduce", "reduce_flips", "restrained"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--platform", choices=["CPU", "CUDA"], default="CPU")
    args = parser.parse_args()
    out = Path("/output")
    records = []
    for model in args.models:
        sources = sorted((Path("/inputs") / model).glob("*.pdb"))[:args.limit]
        if len(sources) != args.limit:
            raise ValueError(f"Expected {args.limit} public inputs for {model}")
        for mode in args.modes:
            options = Options(hydrogen_method="reduce" if mode.startswith("reduce") else "openmm",
                              sidechain_flips=mode == "reduce_flips",
                              relax="restrained" if mode == "restrained" else "none", seed=42,
                              relax_platform=args.platform if mode == "restrained" else "CPU")
            for rep in range(3):
                dest = out / model / mode / f"run{rep}"
                dest.mkdir(parents=True, exist_ok=False)
                for source in sources:
                    start = time.perf_counter()
                    with Memory() as memory:
                        try:
                            prepared, relaxed, report = prepare_structure(source.read_text(), options)
                        except PreparationError as exc:
                            (dest / (source.stem + ".failed.json")).write_text(json.dumps(exc.report, indent=2) + "\n")
                            raise
                    row = {"model": model, "mode": mode, "input": source.stem, "replicate": rep,
                           "warmup": rep == 0, "wall_seconds": time.perf_counter()-start,
                           **memory.result(), "report": report}
                    (dest / (source.stem + ".prepared.pdb")).write_text(prepared)
                    if relaxed is not None:
                        (dest / (source.stem + ".relaxed.pdb")).write_text(relaxed)
                    (dest / (source.stem + ".json")).write_text(json.dumps(row, indent=2) + "\n")
                    records.append(row)
                    (out / "measurements.json").write_text(json.dumps(records, indent=2) + "\n")
                print(f"{model} {mode} repetition {rep} complete", flush=True)


if __name__ == "__main__":
    main()
