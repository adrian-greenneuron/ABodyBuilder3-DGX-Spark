# Preparation and fast-prediction benchmarks — 25 October 2026

The updated CLI was compared with the previous modernised deployment (`3e64fab`). Both use the same ABodyBuilder3 model source and weights. The updated image also pins OpenMM's CUDA dependencies to 13.0 for driver 580.178.04. Main and the retained original image were unchanged.

## Fast prediction

Complete fresh-container jobs include imports, loading, inference and writing PDBs. Each condition has one warm-up and two measured runs, interleaved between conditions; the final pair reverses order. Base/pLDDT use 1,000 inputs and eight workers; language uses 20 inputs and one worker, with live embeddings from the same pinned offline ProtT5 assets. Workloads cycle five public sequence pairs, not 1,000 distinct antibodies.

Times are medians of the two measured runs, with their ranges. RSS and CUDA columns are the higher measured peak. RSS is summed process-tree RSS (shared pages can be counted more than once); CUDA is the sampled concurrent PyTorch allocation. These are different quantities and must not be added on the Spark's unified-memory system.

| Model / inputs | Previous time, s | Updated time, s | Time change | Previous / updated RSS peak, GiB | Previous / updated CUDA allocated peak, GiB |
|---|---:|---:|---:|---:|---:|
| base / 1,000 | 82.47 (81.06–83.88) | 82.18 (81.96–82.41) | -0.3% | 14.678 / 14.577 | 1.041 / 1.018 |
| plddt / 1,000 | 82.28 (82.00–82.55) | 83.32 (83.18–83.45) | +1.3% | 14.676 / 14.620 | 0.986 / 0.978 |
| language / 20 | 339.14 (289.30–388.99) | 328.83 (314.65–343.00) | -3.0% | 8.790 / 8.894 | 4.552 / 4.552 |

All **12,120** fast-path PDB files, including warm-ups, matched the retained original reference bytes. Plain prediction never imports the new preparation module or runs its chemistry stages. Base/pLDDT medians remained within 1.3% of the previous deployment. Language timings vary widely; the observed median change is not evidence of a reliable speed improvement. With two repetitions these measurements cannot establish a precise small performance difference.

## Optional preparation

The same five distinct saved public predictions from each model were prepared once for warm-up and twice for measurement in one container. Values below are median per-antibody times across five pairs × two measurements, with the full measured range. They include repair, hydrogen placement, validation and optional minimisation; they exclude model prediction, container startup and output-file writing. Process RSS is sampled every 50 ms and is the highest measured value for that row. OpenMM allocations are not tracked by PyTorch's allocator, so no OpenMM CUDA peak is claimed.

| Model | Preparation | Time per antibody, median (range), s | Process RSS peak, GiB |
|---|---|---:|---:|
| base | OpenMM hydrogens | 9.58 (7.79–12.24) | 0.722 |
| base | Reduce side-chain optimisation + OpenMM hydrogens | 11.50 (8.56–12.54) | 0.816 |
| base | OpenMM hydrogens + restrained CUDA relaxation | 12.82 (8.41–15.85) | 0.768 |
| plddt | OpenMM hydrogens | 8.46 (7.79–10.48) | 0.777 |
| plddt | Reduce side-chain optimisation + OpenMM hydrogens | 9.52 (9.03–11.00) | 0.872 |
| plddt | OpenMM hydrogens + restrained CUDA relaxation | 10.66 (8.83–11.37) | 0.803 |
| language | OpenMM hydrogens | 8.80 (8.51–9.80) | 0.808 |
| language | Reduce side-chain optimisation + OpenMM hydrogens | 9.81 (9.18–10.52) | 0.901 |
| language | OpenMM hydrogens + restrained CUDA relaxation | 11.27 (9.71–13.45) | 0.843 |

All **135** preparation cases (including warm-ups) completed. All **30** measured restrained relaxations met the force criterion, reduced the restrained energy, and had zero severe heavy-atom overlaps under the recorded metric. Cα RMS displacement from the raw predictions ranged from **0.120 to 0.162 Å**. No residues were added or removed. Preparation without requested side-chain flips preserved all original heavy-atom coordinates.

The overlap metric counts nonbonded heavy-atom pairs closer than 70% of their van der Waals radius sum, excluding atoms separated by up to three covalent bonds. It is a local diagnostic, not the MolProbity clashscore or a complete stereochemical assessment. No accuracy improvement against experimental structures is claimed.

The fixed protocol used Amber ff14SB + GBn2 implicit solvent, pH 7, seed 42, a heavy-atom restraint of 1,000 kJ/mol/nm², a 10 kJ/mol/nm RMS-force target and a 1,000-iteration budget for each minimisation stage. Hydrogens were first optimised with heavy atoms fixed, then all atoms minimised with heavy-atom restraints. CUDA used mixed precision and deterministic force accumulation. This reduces one source of variation; prepared/relaxed coordinates are not promised to be byte-identical across runs/platforms.

A separate CPU qualification on the pLDDT 6yio example took 252.81 and 252.71 seconds for the two measured preparation-plus-relaxation calls. It used the earlier CUDA-13.4-wheel image's CPU path. The corresponding preliminary CUDA qualification after the dependency correction took 10.67 and 10.58 seconds. Those limited qualification timings are not a controlled CPU/GPU speedup experiment; the table above reports the final protocol.

## Checks and reproducibility

The installed Docker CLI produced raw, prepared, relaxed and JSON audit files for all five public pLDDT examples. Four interface/policy tests and four real-chemistry tests passed, alongside the existing five measurement tests. Chemistry tests cover heavy-atom preservation, terminal repair, disulfides, pH-dependent protonation, missing-heavy-atom repair and explicit failure without backend substitution. The removed `--relaxation`/`-r` flag is rejected before importing the model runtime.

The earlier CUDA 13.4 NVRTC dependency produced `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` with the installed driver. The final recipe pins published CUDA 13.0 wheel dependencies; its CUDA relaxation path passed the tests above. No driver changes or custom OpenMM compilation were needed.

Full compact measurements, warm-up flags, image/source identities and parity counts are in [the measurement JSON](data/abodybuilder3_preparation_measurements_251026.json); [CSV summaries](abodybuilder3_preparation_summary_251026.csv) and the [preparation guide](abodybuilder3_preparation_251026.md) accompany them. The local study retains the complete per-atom changes, source snapshots, commands and logs. `tools/run_preparation_study.py` queues the finite comparison; `tools/summarise_preparation.py` regenerates these tables.
