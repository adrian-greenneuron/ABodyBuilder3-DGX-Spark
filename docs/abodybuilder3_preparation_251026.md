# Prediction, preparation and relaxation

Updated for dev on 27 October 2026. Main retains its previous preparation code.

Plain prediction remains the fast default. Optional chemical preparation runs after inference and writing is kept separate from the model. This preserves an unmodified raw prediction for comparison. The model source, weights, precision and default pLDDT model are unchanged.

At pH 7, dev uses GPU hydrogen placement, deterministic single-threaded CPU atom
repair and a phased pool of up to 12 preparation workers. Local rescue is the
default for rejected fixed-heavy geometry. The qualified 512-antibody pipeline
achieved 146.1 structures/minute, or 129.3/minute with Reduce, including prediction,
startup and writing. See the [rescue results](abodybuilder3_rescue_integration_271026.md)
for the protocol and limits. The linked 25 October benchmark report retains the
earlier serial preparation measurements; they are not timings for this new path.

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

# More extensive rescue, only on rejected cases
python3 scripts/predict.py predict --csv antibodies.csv -o predictions/ \
    --add-hydrogens --rescue restrained --seed 42

# Keep heavy atoms fixed even if preparation must fail on incompatible geometry
python3 scripts/predict.py predict --csv antibodies.csv -o predictions/ \
    --add-hydrogens --rescue off --seed 42

# Preparation followed by genuine restrained minimisation on the GPU
python3 scripts/predict.py predict --csv antibodies.csv -o predictions/ \
    --relax restrained --relax-platform cuda --seed 42
```

For each antibody, `.pdb` remains the raw prediction. Requested preparation adds `.prepared.pdb` and `.preparation.json`; relaxation also adds `.relaxed.pdb`. The JSON records repairs, added/moved atoms, inferred disulfides, protonation variants, tool versions, options, warnings, timings and minimisation outcomes. Raw coordinates are never replaced by prepared coordinates.

`--relaxation` and `-r` have been removed. They fail with a message explaining `--add-hydrogens` and `--relax restrained`; they no longer act as aliases. `-H` is reserved for the heavy-chain sequence. The old `--hydrogen-method pdbfixer` value is replaced by `openmm`, which includes the PDBFixer repair step. Preparation currently requires PDB output; CIF/JSON requests with preparation fail rather than silently omit it.

## Chemical preparation

1. PDBFixer finds and adds missing heavy and terminal atoms. It preserves existing atoms and chain/residue identities. It does not rebuild missing sequence stretches, replace nonstandard residues or delete ligands/water. This pipeline is qualified for the standard protein residues emitted by ABodyBuilder3; arbitrary ligand-containing structures are outside that qualification.
2. If explicitly selected, Reduce runs with side-chain flips disabled unless `--sidechain-flips` is present. Chain-local numbering is used internally to avoid residue-number-dependent N-terminal behaviour, then original identifiers are restored. Reduce output and warnings are recorded. A failed Reduce command fails preparation; no alternative is selected automatically.
3. OpenMM assigns the final hydrogen topology and requested pH. After optional Reduce processing, its hydrogens are discarded while requested heavy-atom orientations are retained. This combination avoids inconsistent disulfide/peptide assignments from Reduce's distance-based bond inference on strained predictions. No Reduce-only hydrogen result is presented as the canonical prepared structure.
4. At pH 7, hydrogen positions use the qualified geometry objective on CUDA with mixed precision and deterministic force accumulation. It enforces fixed X–H lengths, planar/outward hydrogen placement and chirality checks, using Amber angle/torsion targets and bounded sterics. Coulomb and GB energies are omitted from this objective; it is not full Amber minimisation. At most three declared steric weights are tried against unchanged geometry thresholds.
5. If a fixed heavy-atom motif cannot support those hydrogen constraints, the selected rescue policy may repair it. Other failures propagate. The force field must parameterise the result, and finite coordinates, residue/atom identities, disulfides, geometry and permitted coordinate movement are checked. Long peptide bonds are reported, not silently turned into internal termini.

## Rescue only rejected structures

| `--rescue` | Behaviour at pH 7 without `--relax` |
|---|---|
| `local` (default) | Bounded repair around the rejected motif; successful preparations bypass rescue |
| `restrained` | Guarded whole-structure restrained rescue, applied only to rejected cases |
| `off` | No heavy-atom rescue; normal fixed-heavy geometry placement still runs |

Local rescue tries residue radii 1 then 2, each from the original repaired
coordinates. It applies strong positional restraints and requires the original
motif's bond errors to be at most 0.05 Å and angle error at most 10°. It must not
introduce new heavy-bond, heavy-angle or severe-contact flags, or invert a tested
biological stereocentre. Unrelated pre-existing strain can remain. Exhausting
the local search is a failure, never permission to switch to restrained rescue.

Restrained rescue first repairs heavy geometry, then uses fixed-heavy hydrogen
minimisation and full Amber ff14SB/GBn2 minimisation with heavy-position restraints
and a chirality guard. Both physical stages must converge and the paired geometry
gate must pass. Final hydrogens are rebuilt with the common geometry objective
and frozen protonation choices. The final prepared PDB is therefore not the
physical minimisation endpoint. Larger heavy-coordinate changes are expected.

Both policies retain atom identities and disulfide topology. Original raw PDBs
remain separate. The preparation audit records the policy, whether rescue ran,
the triggering failure, intermediate structures, displacement, geometry gates
and timings. The default seed for preparation is 0; use `--seed 42` to reproduce
the qualification protocol. Numerical variation remains possible, particularly
in restrained rescue.

`--preparation-workers auto` uses at most 12 isolated processes after inference,
with one CPU thread per repair engine. Explicit counts 1–20 are available.
Whole-structure relaxation stays serial. CUDA errors fail without switching to
CPU or another chemical method.

The default pH is 7. Rescue is qualified only at this pH. Other pH values require
`--rescue off` and use the earlier OpenMM CPU hydrogen-placement protocol.
Whole-structure `--relax restrained` also retains that initial preparation
protocol, with rescue automatically off. Explicit `--rescue local` or
`--rescue restrained` with whole-structure relaxation fails before model loading. pH-based protonation follows OpenMM's
standard residue rules and local choices; it is not a full pKa calculation. Not
every terminal/protonation combination has an Amber template. The checked
antibody's pH-2 protonated N-terminal Asp failed parameterisation rather than
silently changing its charge state. Histidine protonation at pH 5 versus 7 was
tested successfully on the earlier protocol.

Disulfides are inferred by the PDB reader from geometry/connectivity and recorded in the audit. Recording a bond does not independently establish the biological disulfide assignment. Review unexpected assignments, missing atoms and long-bond warnings before downstream simulation or docking.

## Restrained minimisation

`--relax restrained` includes preparation and minimises every requested structure;
it is separate from rejection-only `--rescue restrained`. It uses the pinned
OpenMM 8.6.1 runtime, Amber ff14SB protein parameters and GBn2 implicit solvent.
First, hydrogens are optimised with heavy atoms immobile. A second minimisation
allows atom movement while restraining all heavy atoms to the prepared
coordinates with the harmonic potential `0.5*k*displacement²`. This established
whole-structure option does not add the rejection-only rescue chirality guard.

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

The Dockerfile obtains pinned model source and weights from public sources, then
copies this checkout's CLI and preparation modules into the image. The
version-guarded patches in `patches/preparation` expose explicit PDBFixer context
properties and an OpenMM hydrogen-optimiser callback. They are the exact changes
qualified on the [component branch](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/tree/da2672e33dd1afdc4f1b3fadf5d997ef9f8fdd99);
original/candidate sources and comparisons remain there. Installation fails if
the upstream source hashes differ. No further third-party code was changed for
the dev integration. `runtime-constraints.txt` pins CUDA 13.0 dependencies
compatible with tested driver 580.178.04. These avoid unsupported newer NVRTC
PTX without a driver change or custom OpenMM binary build.

```bash
IMAGE_TAG=preparation-local ./docker_build.sh build
IMAGE_TAG=preparation-local ./docker_build.sh test
python3 -m unittest discover -s tests -p 'test_structure_preparation.py' -v
```

`tests/test_preparation_integration.py` runs in the Docker environment with `ABB3_TEST_PDB` pointing to a saved public heavy-atom prediction and `trastuzumab_like.pdb` in the same directory. Those inputs are generated from `tests/benchmark_antibodies.json`. `tools/run_preparation_study.py --help` describes the finite matched benchmark queue. It requires the previous runtime/CLI identity and the pinned ProtT5 snapshot, plus the five saved public predictions from each model. Detailed protocol and measured results are in [the preparation benchmarks](abodybuilder3_preparation_benchmarks_251026.md).
