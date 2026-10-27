# Rejection-only rescue: full 512-antibody pipeline

**Both rescue options prepare all 512 antibodies, repairing only the three
rejected cases. The other 509 prepared outputs remain byte-identical to their
matched controls.** Main, dev and their default images are unchanged. These
experiments are available for review on this component branch.

| Full pipeline | Local rescue | Restrained rescue |
| --- | ---: | ---: |
| OpenMM preparation | **146.1 structures/min** | 138.4 structures/min |
| Reduce + OpenMM preparation | **129.3 structures/min** | 124.9 structures/min |

Two completed technical repeats per condition, strict FP32 saved AOT and 12
preparation workers. These rates include model loading, prediction, chemical
preparation, rejection-trigger attempts and writing; they are **not the
prediction-only repertoire rates**. Local rescue adds less than 1% median wall
time relative to the matched no-rescue controls; restrained rescue adds 4–6%.
Those controls prepare only 509/512.

![Full-pipeline throughput and unified RAM](docs/assets/rescue_pipeline512_performance_271026.png)

Completed rescue jobs show about **8–11 GiB additional unavailable unified
RAM**. Host-memory drift prevents a reliable ranking between the two policies.
The first Reduce/local attempt had four CUDA-initialisation failures; it is
retained separately, and one unchanged replacement completed all 512. See the
[full report](docs/rescue_pipeline512_271026.md) for every timing, memory
sensitivity and the failed attempt.

All **922 unique prepared PDBs pass the hydrogen geometry screens**. Restrained
rescue clears the tested heavy bond, angle and severe-contact flags on the
three repaired cases, with larger coordinate changes; local rescue leaves
unrelated pre-existing strain. The other 509 are not relaxed. This does not
establish improved experimental structural accuracy or clash-free structures.
All **45 preparation and dispatch regression tests pass**.

- [Paired geometry figure](docs/assets/rescue_pipeline512_geometry_271026.png)
- [Anonymised measurements and geometry summaries](docs/data/rescue_pipeline512_results_271026.json)
- [Full-pipeline source/runtime provenance](provenance_rescue_pipeline512_271026.json)
- [Experimental pipeline image](Dockerfile.rescue-pipeline) and [explicit dispatch](candidate/rescue_policy.py)

The experimental image accepts `AB3_EXPERIMENTAL_RESCUE=local|restrained`; `off`
retains R8 behaviour. It tries normal preparation first and repairs only an
explicit frozen-heavy planar rejection, confirmed by preflight. Other failures
propagate. Failed local repair never silently invokes whole-protein relaxation.
The original R8 function is retained verbatim, with its source also saved in
`reference/structure_preparation_r8.py`. No further upstream-library code is
changed. Input sequences, structures, detailed diagnostics and traces stay
local; all prior images, failed outputs and records are retained.

Earlier reports remain available: [isolated rescue qualification](docs/heavy_atom_rescue_271026.md),
[fixed-heavy H geometry](docs/hydrogen_geometry_271026.md),
[GPU/streaming preparation](docs/hydrogen_pipeline_271026.md),
[deterministic atom repair](docs/deterministic_repair_261026.md), and
[preparation concurrency](docs/preparation_optimisation_261026.md).
