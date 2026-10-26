# Docker-packaged compilation caches — 26 October 2026

Follow-up: [saving a whole-model AOT executable](abodybuilder3_aot_deployment_261026.md)
substantially reduces the remaining startup cost. This page retains the earlier
disk-cached JIT measurements, which describe a different deployment path.

Packaging the prepared PyTorch/Triton caches cut complete compiled jobs from roughly **89–104 seconds to 28–39 seconds**, compared with the earlier empty-cache study. It still did not beat current dev on **100 or 1,000 inputs**. Large jobs subsequently became much faster when combined with the native writer; those separate measurements are in the [overnight results](abodybuilder3_overnight_optimisation_261026.md).

## What was packaged

Eight banks cover base/pLDDT × compilation-only/attention-plus-compilation × TF32-permitted/strict-FP32. Retained cold-study caches seeded the banks. Each bank was then prepared on the GPU for the 100- and 1,000-input workloads, hashed and copied into a Docker layer, together with frozen component and CLI code. Preparation and image building are deployment costs outside the measured fresh jobs. The preparation records and image/cache-manifest identities are retained in the evidence.

Measured compiled containers read their code, input CSVs and caches from the image, with **no host compiler-cache mount**. Only the output directory was mounted, at `/results`, leaving the baked `/output` cache paths visible. The initial eager control failed before inference because its user could not create the default cache directory; it was rerun with a writable `/results` cache. That failed attempt remains in the local audit and is excluded from successful timings.

These are disk/kernel caches, not a serialised whole-model executable. Fresh processes still import libraries, load weights, reconstruct Dynamo graphs/guards and cross existing Rigid/Rotation graph breaks. The first forward can therefore take around 16–20 seconds even when kernel artefacts are available. On-disk cache hits and cache-manifest identities were recorded; process-local `async_compile_cache_miss` counters must not be interpreted as proof that an on-disk kernel was regenerated. Other AOT cache misses can still occur. Caches are specific to the tested software, model, precision and GPU environment.

## Full CLI timings with the original writer

Medians of two fresh containers per cell, including startup and all PDB output. Bold marks the lowest displayed time, not statistical significance. Same 100 distinct experimental targets; 1,000 inputs repeat them ten times. No prediction caching, weight changes or optional chemistry. Normal is TF32-permitted FP32 tensors; strict disables TF32 matrix multiplication. Standard automatic batch caps and two CPU preparation/writing threads were preserved. All matched batch shapes agreed and there were no OOM retries.

| Inputs | Checkpoint | Precision | Current dev (s) | Cached compiler (s) | Cached attention + compiler (s) |
| --- | --- | --- | --- | --- | --- |
| 100 | base | TF32 permitted | **7.20** | 27.98 | 28.50 |
| 100 | base | Strict FP32 | **7.40** | 28.06 | 28.34 |
| 100 | plddt | TF32 permitted | **7.46** | 28.02 | 28.57 |
| 100 | plddt | Strict FP32 | **7.49** | 28.24 | 28.06 |
| 1,000 | base | TF32 permitted | **28.85** | 37.68 | 37.77 |
| 1,000 | base | Strict FP32 | **28.76** | 37.79 | 37.98 |
| 1,000 | plddt | TF32 permitted | **29.46** | 38.96 | 38.18 |
| 1,000 | plddt | Strict FP32 | **29.70** | 38.38 | 38.94 |

## Predictions and memory

Both fresh repetitions were scored on the first occurrence of all 100 distinct experimental targets. All 26,400 PDBs were written, and every raw fingerprint/PDB was identical between the two fresh repetitions of each condition in this cached study. This differs from the earlier cold-cache result; it is evidence for this pinned cache bank, not a universal determinism guarantee.

Compilation and attention still change floating-point arithmetic. Across these cached jobs, TF32-permitted candidates changed CDR-H3 experimental backbone RMSD by at most **0.003384 Å** relative to matched dev; a side-chain atom moved by up to **1.984 Å**. Strict-FP32 H3 changes were at most **0.00000839 Å**, with atom displacements up to **0.001382 Å**. Backbone changes were tiny, but atom displacement is not itself experimental error, and large side-chain movements should not be dismissed as uniformly harmless.

Peak CUDA allocation was lower for compiled paths, while process RSS was higher. The full resource tables, reserved CUDA peaks and container accounting are in the evidence and the bulk report. These measures must not be summed on the Spark's unified memory.

## Reproduction and evidence

`tools/run_pipeline_study.py --cache-source ...` prepares and packages banks from a retained cold comparison. `--resume` reuses frozen sources, finished jobs and bank identities after a controller failure. Build first from the pinned dev recipe, then perform the GPU preparation; a conventional GPU-less Docker build alone does not execute the model to discover all workload shapes. The measured runtime was NGC PyTorch 2.10.0a0/CUDA 13.0 on GB10, with the expandable-segments allocator enabled.

The [cache evidence archive](data/abodybuilder3_cached_pipeline_evidence_261026.zip) contains both structural analyses, per-target scores, compact individual measurements, image identities and preparation records. Original source/runtime identities and all local caches/outputs are retained. Main and dev are unchanged; neither result-changing candidate is enabled there.
