# Prediction, preparation and relaxation — 25 October 2026

Plain prediction remains the fast default. Optional chemical preparation runs after inference and writing is kept separate from the model. This preserves an unmodified raw prediction for comparison. The model source, weights, precision and default pLDDT model are unchanged.

Preparation performs more validation and force-field work than the legacy hydrogen-only path, and its cost is opt-in. In this study OpenMM preparation took roughly 8–10 seconds per antibody; CUDA preparation plus restrained relaxation took roughly 11–13 seconds. CPU relaxation is substantially slower. See the benchmark protocol before comparing these timings with inference or the old hydrogen-addition measurements.

## Commands and outputs

```bash
# Raw heavy-atom predictions
python3 scripts/predict.py predict --csv antibodies.csv -o predictions/

# Missing-atom repair and hydrogens, default pH 7
python3 scripts/predict.py predict --csv antibodies.csv -o predictions/ \
    --add-hydrogens --ph 7 --seed 42

# Explicit optional side-chain optimisation, then canonical protonation
python3 scripts/predict.py predict --csv antibodies.csv -o predictions/ \
    --add-hydrogens --hydrogen-method reduce --sidechain-flips --ph 7 --seed 42

# Preparation followed by genuine restrained minimisation on the GPU
python3 scripts/predict.py predict --csv antibodies.csv -o predictions/ \
    --relax restrained --relax-platform cuda --seed 42
```

For each antibody, `.pdb` remains the raw prediction. Requested preparation adds `.prepared.pdb` and `.preparation.json`; relaxation also adds `.relaxed.pdb`. The JSON records repairs, added/moved atoms, inferred disulfides, protonation variants, tool versions, options, warnings, timings and minimisation outcomes. Raw coordinates are never replaced by prepared coordinates.

`--relaxation` and `-r` have been removed. They fail with a message explaining `--add-hydrogens` and `--relax restrained`; they no longer act as aliases. `-H` is reserved for the heavy-chain sequence. The old `--hydrogen-method pdbfixer` value is replaced by `openmm`, which includes the PDBFixer repair step. Preparation currently requires PDB output; CIF/JSON requests with preparation fail rather than silently omit it.

## Chemical preparation

1. PDBFixer finds and adds missing heavy and terminal atoms. It preserves existing atoms and chain/residue identities. It does not rebuild missing sequence stretches, replace nonstandard residues or delete ligands/water. This pipeline is qualified for the standard protein residues emitted by ABodyBuilder3; arbitrary ligand-containing structures are outside that qualification.
2. If explicitly selected, Reduce runs with side-chain flips disabled unless `--sidechain-flips` is present. Chain-local numbering is used internally to avoid residue-number-dependent N-terminal behaviour, then original identifiers are restored. Reduce output and warnings are recorded. A failed Reduce command fails preparation; no alternative is selected automatically.
3. OpenMM adds the final hydrogens against the repaired topology and requested pH. After optional Reduce processing, its hydrogens are discarded while requested heavy-atom orientations are retained. This declared combination avoids inconsistent disulfide/peptide assignments from Reduce's distance-based bond inference on strained predictions. No Reduce-only hydrogen result is presented as the canonical prepared structure.
4. The force field must parameterise the prepared topology. Finite coordinates, preserved residues and the expected heavy atoms are checked. Existing heavy atoms must stay fixed unless side-chain flips were requested; those flips still cannot move backbone atoms. Long peptide bonds are reported, not silently turned into internal termini.

The default pH is 7. pH-based protonation follows OpenMM's standard residue rules and local choices; it is not a full pKa calculation. Not every terminal/protonation combination has an Amber template. For example, the checked antibody's pH-2 protonated N-terminal Asp could not be parameterised; the command reports that failure rather than replacing it with a different charge state. Histidine protonation at pH 5 versus 7 was tested successfully.

Disulfides are inferred by the PDB reader from geometry/connectivity and recorded in the audit. Recording a bond does not independently establish the biological disulfide assignment. Review unexpected assignments, missing atoms and long-bond warnings before downstream simulation or docking.

## Restrained minimisation

`--relax restrained` includes preparation. It uses the pinned OpenMM 8.6.1 runtime, Amber ff14SB protein parameters and GBn2 implicit solvent. First, hydrogens are optimised with heavy atoms immobile. A second minimisation allows atom movement while restraining all heavy atoms to the prepared coordinates with the harmonic potential `0.5*k*displacement²`.

| Setting | Default | Meaning |
|---|---:|---|
| `--relax-platform` | `cpu` | Portable CPU implementation, one thread; explicitly select `cuda` for the tested Spark GPU path |
| `--relax-restraint` | 1000 | Heavy-atom harmonic coefficient, kJ/mol/nm² |
| `--relax-tolerance` | 10 | RMS force target, kJ/mol/nm |
| `--relax-max-iterations` | 1000 | Positive iteration budget for each stage; never an unbounded run |
| `--seed` | preparation uses 0 if omitted | Seed recorded for atom/hydrogen preparation; inference behaviour is otherwise unchanged |

CUDA uses mixed precision and deterministic force accumulation. The CPU and CUDA paths can give different minimised coordinates. Numerical variation can remain between runs; these settings do not promise byte-identical prepared or relaxed PDBs. Energy and force criteria are evaluated before PDB coordinate rounding.

The audit records initial/final restrained energies, final physical energy, force norm, convergence, geometry and displacement from the raw model. A finite-budget run that does not meet tolerance is explicitly marked with a warning; inspect `restrained_minimisation.converged` rather than treating completion as convergence. A backend error, invalid coordinates or increased restrained energy fails preparation. Raw output and a failure audit are retained, and the CLI exits nonzero if any prediction/preparation fails. Logs and exit codes distinguish failures from confidence-filtered predictions.

No molecular dynamics trajectory is run. Lower energy, fewer overlaps and better local geometry do not establish improved structural accuracy. Relaxation may change a correct local conformation; the raw prediction remains available.

## Rebuild, test and measure

The Dockerfile obtains pinned model source and weights from public sources, then copies this checkout's CLI and preparation module into the image. `runtime-constraints.txt` pins CUDA 13.0 dependencies compatible with tested driver 580.178.04. Newer CUDA-13.4 NVRTC wheels emitted PTX that this driver could not load; the explicit pins resolve that failure without a driver change or custom OpenMM build.

```bash
IMAGE_TAG=preparation-local ./docker_build.sh build
IMAGE_TAG=preparation-local ./docker_build.sh test
python3 -m unittest discover -s tests -p 'test_structure_preparation.py' -v
```

`tests/test_preparation_integration.py` runs in the Docker environment with `ABB3_TEST_PDB` pointing to a saved public heavy-atom prediction and `trastuzumab_like.pdb` in the same directory. Those inputs are generated from `tests/benchmark_antibodies.json`. `tools/run_preparation_study.py --help` describes the finite matched benchmark queue. It requires the previous runtime/CLI identity and the pinned ProtT5 snapshot, plus the five saved public predictions from each model. Detailed protocol and measured results are in [the preparation benchmarks](abodybuilder3_preparation_benchmarks_251026.md).
