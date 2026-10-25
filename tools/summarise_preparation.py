"""Publish compact, reproducible tables from the completed preparation study."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    args.docs.mkdir(parents=True, exist_ok=True)
    data_dir = args.docs / "data"; data_dir.mkdir(exist_ok=True)
    done = json.loads((args.study / "completed_251026.json").read_text())
    assert done["completed"]
    fast = []
    for path in sorted((args.study / "cli").glob("*/*/measurements.json")):
        for row in json.loads(path.read_text()):
            row["condition"] = path.parent.name
            fast.append(row)
    chemistry = json.loads((args.study / "chemistry/measurements.json").read_text())
    compact = []
    for row in chemistry:
        record = {k:v for k,v in row.items() if k != "report"}
        report = row["report"]
        keep = ["status", "options", "raw_sha256", "versions", "forcefield", "effective_ph",
                "hydrogen_backend", "warnings", "raw_geometry", "prepared_geometry", "relaxed_geometry",
                "hydrogen_minimisation", "restrained_minimisation"]
        record["report"] = {k:report[k] for k in keep if k in report}
        for name in ("preparation_changes", "relaxation_changes"):
            if name in report:
                changes = report[name]
                record["report"][name] = {k:v for k,v in changes.items() if k not in ("added_atoms", "moved_heavy_atoms")}
                record["report"][name].update(added_atom_count=len(changes["added_atoms"]), moved_heavy_atom_count=len(changes["moved_heavy_atoms"]))
        if "reduce" in report:
            record["report"]["reduce"] = {k:v for k,v in report["reduce"].items() if k != "residue_mapping"}
        compact.append(record)
    manifest = json.loads((args.study / "frozen_identity_251026.json").read_text())
    parity = json.loads((args.study / "raw_parity_251026.json").read_text())
    evidence = {"completed":done,"frozen_identity":manifest,"raw_parity":parity,"fast_cli":fast,"preparation":compact}
    (data_dir / "abodybuilder3_preparation_measurements_251026.json").write_text(json.dumps(evidence,indent=2)+"\n")
    summary = []
    lines = ["# Preparation and fast-prediction benchmarks — 25 October 2026", "",
        "The updated CLI was compared with the previous modernised deployment (`3e64fab`). Both use the same ABodyBuilder3 model source and weights. The updated image also pins OpenMM's CUDA dependencies to 13.0 for driver 580.178.04. Measurements preceded promotion of this implementation to main; the retained original image was unchanged.", "",
        "## Fast prediction", "",
        "Complete fresh-container jobs include imports, loading, inference and writing PDBs. Each condition has one warm-up and two measured runs, interleaved between conditions; the final pair reverses order. Base/pLDDT use 1,000 inputs and eight workers; language uses 20 inputs and one worker, with live embeddings from the same pinned offline ProtT5 assets. Workloads cycle five public sequence pairs, not 1,000 distinct antibodies.", "",
        "Times are medians of the two measured runs, with their ranges. RSS and CUDA columns are the higher measured peak. RSS is summed process-tree RSS (shared pages can be counted more than once); CUDA is the sampled concurrent PyTorch allocation. These are different quantities and must not be added on the Spark's unified-memory system.", "",
        "| Model / inputs | Previous time, s | Updated time, s | Time change | Previous / updated RSS peak, GiB | Previous / updated CUDA allocated peak, GiB |",
        "|---|---:|---:|---:|---:|---:|"]
    for model,count in [("base",1000),("plddt",1000),("language",20)]:
        values = {}
        for condition in ("baseline","candidate"):
            rows = [r for r in fast if r["model"] == model and r["condition"] == condition and not r["warmup"]]
            times = [r["complete_process_seconds"] for r in rows]
            values[condition] = {"median":statistics.median(times), "min":min(times), "max":max(times),
                "rss":max(r["sampled_process_tree_rss_peak_bytes"] for r in rows)/2**30,
                "cuda":max(r["sampled_concurrent_cuda_allocated_peak_bytes"] for r in rows)/2**30}
            summary.append({"kind":"fast", "model":model,"mode":condition,"count":count,"median_seconds":statistics.median(times),
                            "min_seconds":min(times),"max_seconds":max(times),"rss_peak_gib":values[condition]["rss"]})
        a,b = values["baseline"],values["candidate"]
        change = 100*(b["median"]/a["median"]-1)
        lines.append(f"| {model} / {count:,} | {a['median']:.2f} ({a['min']:.2f}–{a['max']:.2f}) | {b['median']:.2f} ({b['min']:.2f}–{b['max']:.2f}) | {change:+.1f}% | {a['rss']:.3f} / {b['rss']:.3f} | {a['cuda']:.3f} / {b['cuda']:.3f} |")
    lines += ["", "All **12,120** fast-path PDB files, including warm-ups, matched the retained original reference bytes. Plain prediction never imports the new preparation module or runs its chemistry stages. Base/pLDDT medians remained within 1.3% of the previous deployment. Language timings vary widely; the observed median change is not evidence of a reliable speed improvement. With two repetitions these measurements cannot establish a precise small performance difference.", "",
        "## Optional preparation", "",
        "The same five distinct saved public predictions from each model were prepared once for warm-up and twice for measurement in one container. Values below are median per-antibody times across five pairs × two measurements, with the full measured range. They include repair, hydrogen placement, validation and optional minimisation; they exclude model prediction, container startup and output-file writing. Process RSS is sampled every 50 ms and is the highest measured value for that row. OpenMM allocations are not tracked by PyTorch's allocator, so no OpenMM CUDA peak is claimed.", "",
        "| Model | Preparation | Time per antibody, median (range), s | Process RSS peak, GiB |",
        "|---|---|---:|---:|"]
    labels = {"openmm":"OpenMM hydrogens", "reduce_flips":"Reduce side-chain optimisation + OpenMM hydrogens", "restrained":"OpenMM hydrogens + restrained CUDA relaxation"}
    for model in ("base","plddt","language"):
        for mode in labels:
            rows = [r for r in chemistry if r["model"]==model and r["mode"]==mode and not r["warmup"]]
            times = [r["wall_seconds"] for r in rows]
            peak = max(r["sampled_process_tree_rss_peak_bytes"] for r in rows)/2**30
            lines.append(f"| {model} | {labels[mode]} | {statistics.median(times):.2f} ({min(times):.2f}–{max(times):.2f}) | {peak:.3f} |")
            summary.append({"kind":"preparation","model":model,"mode":mode,"count":len(rows),"median_seconds":statistics.median(times),"min_seconds":min(times),"max_seconds":max(times),"rss_peak_gib":peak})
    restrained = [r["report"] for r in chemistry if r["mode"]=="restrained" and not r["warmup"]]
    assert all(r["restrained_minimisation"]["converged"] for r in restrained)
    assert all(r["relaxed_geometry"]["severe_heavy_atom_overlaps"]==0 for r in restrained)
    assert all(r["prepared_geometry"]["residues"]==r["raw_geometry"]["residues"] for r in [x["report"] for x in chemistry])
    ca = [r["relaxation_changes"]["ca_rms_displacement_angstrom"] for r in restrained]
    lines += ["", f"All **135** preparation cases (including warm-ups) completed. All **30** measured restrained relaxations met the force criterion, reduced the restrained energy, and had zero severe heavy-atom overlaps under the recorded metric. Cα RMS displacement from the raw predictions ranged from **{min(ca):.3f} to {max(ca):.3f} Å**. No residues were added or removed. Preparation without requested side-chain flips preserved all original heavy-atom coordinates.", "",
        "The overlap metric counts nonbonded heavy-atom pairs closer than 70% of their van der Waals radius sum, excluding atoms separated by up to three covalent bonds. It is a local diagnostic, not the MolProbity clashscore or a complete stereochemical assessment. No accuracy improvement against experimental structures is claimed.", "",
        "The fixed protocol used Amber ff14SB + GBn2 implicit solvent, pH 7, seed 42, a heavy-atom restraint of 1,000 kJ/mol/nm², a 10 kJ/mol/nm RMS-force target and a 1,000-iteration budget for each minimisation stage. Hydrogens were first optimised with heavy atoms fixed, then all atoms minimised with heavy-atom restraints. CUDA used mixed precision and deterministic force accumulation. This reduces one source of variation; prepared/relaxed coordinates are not promised to be byte-identical across runs/platforms.", "",
        "A separate CPU qualification on the pLDDT 6yio example took 252.81 and 252.71 seconds for the two measured preparation-plus-relaxation calls. It used the earlier CUDA-13.4-wheel image's CPU path. The corresponding preliminary CUDA qualification after the dependency correction took 10.67 and 10.58 seconds. Those limited qualification timings are not a controlled CPU/GPU speedup experiment; the table above reports the final protocol.", "",
        "## Checks and reproducibility", "",
        "The installed Docker CLI produced raw, prepared, relaxed and JSON audit files for all five public pLDDT examples. Four interface/policy tests and four real-chemistry tests passed, alongside the existing five measurement tests. Chemistry tests cover heavy-atom preservation, terminal repair, disulfides, pH-dependent protonation, missing-heavy-atom repair and explicit failure without backend substitution. The removed `--relaxation`/`-r` flag is rejected before importing the model runtime.", "",
        "The earlier CUDA 13.4 NVRTC dependency produced `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` with the installed driver. The final recipe pins published CUDA 13.0 wheel dependencies; its CUDA relaxation path passed the tests above. No driver changes or custom OpenMM compilation were needed.", "",
        "Full compact measurements, warm-up flags, image/source identities and parity counts are in [the measurement JSON](data/abodybuilder3_preparation_measurements_251026.json); [CSV summaries](abodybuilder3_preparation_summary_251026.csv) and the [preparation guide](abodybuilder3_preparation_251026.md) accompany them. The local study retains the complete per-atom changes, source snapshots, commands and logs. `tools/run_preparation_study.py` queues the finite comparison; `tools/summarise_preparation.py` regenerates these tables.", ""]
    (args.docs / "abodybuilder3_preparation_benchmarks_251026.md").write_text("\n".join(lines))
    with (args.docs / "abodybuilder3_preparation_summary_251026.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summary)


if __name__ == "__main__":
    main()
