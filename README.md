# Heavy-atom rescue experiments

**Both local repair and guarded restrained relaxation rescue all three antibodies
that the fixed-heavy hydrogen protocol rejected.** Results repeat with and
without Reduce orientation. Main, dev and their default images are unchanged;
these tools are available only for experimental review on this component branch.

Local repair takes **3.8–4.7 seconds per rejected antibody**, versus **8.9–12.1
seconds** for restrained relaxation on the OpenMM route (two-call medians, three
workers). Local repair moves 28–33 original heavy atoms, with a maximum movement
of 1.29–3.02 Å. Relaxation moves almost the whole structure and removes much more
pre-existing heavy strain. Passing the rescue checks does not establish better
experimental structural accuracy or a clash-free structure.

![Speed, atom movement and hydrogen contacts](docs/assets/heavy_atom_rescue_tradeoffs_271026.png)

The [report](docs/heavy_atom_rescue_271026.md) includes both routes, individual
fresh-container times, unified RAM, stored-file geometry checks, residual
clashes and failed approaches. The three compatible controls remain
byte-identical across all rescue policies and repeats within each route.
All **40 preparation and geometry regression tests pass**.

- [Paired geometry figure](docs/assets/heavy_atom_rescue_geometry_271026.png)
- [Compact measurements and screening history](docs/data/heavy_atom_rescue_results_271026.json)
- [Source/runtime provenance](provenance_heavy_atom_rescue_271026.json)
- [Explicit rescue runner](tools/run_heavy_rescue.py) and [experimental image layer](Dockerfile.heavy-rescue)

The runner supports explicit `fixed`, `local` and `restrained` policies on saved
predictions. Compatible structures bypass heavy repair. Failed local attempts
are recorded; they never trigger automatic whole-protein relaxation. No
installed predictor or preparation CLI is modified. This is an incremental
layer over the retained R8 runtime, whose exact identity must be checked before
building. See the report for methods and reproduction details.

The original and modified OpenMM/PDBFixer component sources remain in
`reference/` and `candidate/`. This rescue study adds application code using
OpenMM's public force/minimisation API; it makes no further upstream-library
edits. Sequences, raw/prepared structures, per-residue diagnostics and traces
remain local. Nothing was deleted.

Earlier studies remain unchanged: [fixed-heavy H geometry](docs/hydrogen_geometry_271026.md),
[GPU/streaming preparation](docs/hydrogen_pipeline_271026.md),
[deterministic atom repair](docs/deterministic_repair_261026.md), and
[preparation concurrency](docs/preparation_optimisation_261026.md).
