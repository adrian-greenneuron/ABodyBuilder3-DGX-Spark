"""Finite, interleaved CLI comparison and chemical-preparation study.

Uses public retained raw predictions for chemistry, and regenerates predictions
for timing/parity. Preserves images, containers, inputs, outputs and failed logs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--prott5", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="3e64fabfc3f61d4816d5c62eaf3675e00c98ce1d")
    parser.add_argument("--baseline-image", default="abodybuilder3-official-spark:modernised-pinned-251026")
    parser.add_argument("--candidate-image", default="abodybuilder3-official-spark:preparation-251026")
    args = parser.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    tools = root / "tools"; shutil.copytree(repo / "tools", tools, ignore=shutil.ignore_patterns("__pycache__"))
    candidate = root / "candidate"
    run([sys.executable, str(repo / "tools/prepare_benchmark_inputs.py"), "--output", str(candidate)], root / "freeze.log")
    baseline = root / "baseline"; shutil.copytree(candidate, baseline)
    (baseline / "predict_main.py").write_bytes(subprocess.check_output(["git", "show", f"{args.baseline_ref}:scripts/predict.py"], cwd=repo))
    images = {label: subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
              for label,image in [("baseline",args.baseline_image),("candidate",args.candidate_image)]}
    manifest = {"baseline_ref": args.baseline_ref, "images": images,
                "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for folder in (candidate,baseline,tools) for p in folder.rglob("*.py")}}
    (root / "frozen_identity_251026.json").write_text(json.dumps(manifest, indent=2)+"\n")
    # Exercise the installed CLI, not just the module: raw + prepared + relaxed
    # files, including parameterisation and minimisation on all five pairs.
    e2e = root / "e2e"; e2e.mkdir()
    run(["docker", "run", "--name", "abb3-preparation-e2e-251026", "--network", "none", "--gpus", "all", "--ipc=host",
         "-e", "OPENMM_CPU_THREADS=1", "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
         "-v", f"{candidate}:/inputs:ro", "-v", f"{e2e}:/output", images["candidate"],
         "python", "/opt/abodybuilder3/scripts/predict.py", "predict", "--csv", "/inputs/antibodies_5.csv",
         "--output", "/output/", "--workers", "1", "--seed", "42", "--relax", "restrained", "--relax-platform", "cuda"], root / "e2e.log")
    if len(list(e2e.glob("*.prepared.pdb"))) != 5 or len(list(e2e.glob("*.relaxed.pdb"))) != 5:
        raise ValueError("Installed CLI did not produce all requested preparation files")
    print("Installed preparation CLI passed", flush=True)
    for model,count,workers in [("base",1000,8),("plddt",1000,8),("language",20,1)]:
        for rep in range(3):
            order = ["baseline","candidate"] if rep != 2 else ["candidate","baseline"]
            for label in order:
                command = [sys.executable,str(tools / "benchmark_cli.py"),"--condition","modernised",
                           "--model",model,"--count",str(count),"--workers",str(workers),
                           "--replicate",str(rep),"--label",f"prep-{label}","--image",images[label],
                           "--frozen",str(root / label),"--output",str(root / "cli" / model / label)]
                if model == "language":
                    command += ["--prott5",str(args.prott5.resolve())]
                run(command,root / "logs" / f"{model}_{label}_r{rep}.log")
                print(f"Fast CLI {model} {label} replicate {rep} passed",flush=True)
    chemistry = root / "chemistry"; chemistry.mkdir()
    run(["docker","run","--name","abb3-preparation-study-251026","--network","none","--gpus","all","--ipc=host",
         "-e","OPENMM_CPU_THREADS=1","-e","PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
         "-e","PYTHONPATH=/measurement","-v",f"{candidate}:/scripts:ro","-v",f"{tools}:/measurement:ro",
         "-v",f"{args.inputs.resolve()}:/inputs:ro","-v",f"{chemistry}:/output",images["candidate"],
         "python","/measurement/benchmark_preparation.py","--modes","openmm","reduce_flips","restrained",
         "--platform","CUDA"],root / "chemistry.log")
    # Check every generated fast-path prediction against its retained original.
    fixtures = json.loads((candidate / "antibodies.json").read_text())
    parity = []
    for model,count in [("base",1000),("plddt",1000),("language",20)]:
        for label in ("baseline","candidate"):
            for rep in range(3):
                folder = root / "cli" / model / label / f"run{rep}" / "pdbs"
                for i in range(count):
                    name = fixtures[i % len(fixtures)]["name"]
                    reference = (args.inputs / model / (name+".pdb")).read_bytes()
                    actual = (folder / f"{i:04d}_{name}.pdb").read_bytes()
                    if actual != reference:
                        raise ValueError(f"Raw prediction changed: {model}/{label}/run{rep}/{i}")
                parity.append({"model":model,"condition":label,"replicate":rep,"pdbs_checked":count,"byte_identical":True})
    (root / "raw_parity_251026.json").write_text(json.dumps(parity,indent=2)+"\n")
    (root / "completed_251026.json").write_text(json.dumps({"completed":True,"images":images,"raw_pdbs_checked":sum(x["pdbs_checked"] for x in parity)},indent=2)+"\n")
    print("Study completed; all raw predictions byte-identical",flush=True)


if __name__ == "__main__":
    main()
