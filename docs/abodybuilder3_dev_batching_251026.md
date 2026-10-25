# Dev batching: design and measurements — 25 October 2026

These measurements describe dev at `07bfe1e`. The subsequent [inference optimisation report](abodybuilder3_dev_optimisation_251026.md) compares that frozen implementation with the newer compact-feature and buffer-reuse path; the values below remain the historical record.

Dev now includes main’s modernised runtime and explicit chemistry pipeline, with an integrated length-aware prediction pipeline. Main was left unchanged. Model source, released checkpoints and FP32 settings are unchanged. This is an implementation/throughput comparison, not a new model or an experimental-structure accuracy benchmark.

## Complete-job performance

Each measurement is a fresh offline container, including imports, model loading, feature preparation, prediction and writing every PDB. Each condition has one whole-job warm-up and two measured repetitions. The 1,000-input workloads cycle the same five public antibody pairs (227–231 residues); they are not 1,000 independent sequences. Main used eight GPU-owning workers; dev batching uses one model process. Runs were sequential; the first large comparison interleaved conditions and reversed the final order. The batch-cap-32 follow-up and final-image default checks were performed afterwards.

| Model | Condition | Complete time, median (range), s | Predictions/min | Peak summed RSS, GiB | Sampled concurrent CUDA allocation, GiB | Container accounted peak, GiB |
|---|---|---:|---:|---:|---:|---:|
| base | Main, 8 workers | 80.44 (80.19–80.68) | 746 | 14.287 | 1.003 | 10.895 |
| base | Dev, serial CPU formatting | 69.68 (69.23–70.13) | 861 | 1.763 | 0.178 | 1.521 |
| base | Dev, automatic batching | 33.18 (33.10–33.25) | 1809 | 4.158 | 4.591 | 4.017 |
| plddt | Main, 8 workers | 81.47 (81.43–81.51) | 737 | 14.208 | 1.141 | 10.915 |
| plddt | Dev, serial CPU formatting | 70.30 (70.15–70.45) | 854 | 1.767 | 0.178 | 1.529 |
| plddt | Dev, automatic batching | 33.66 (33.66–33.66) | 1782 | 4.169 | 4.530 | 4.027 |

Final default throughput improved **2.42× for base and 2.42× for pLDDT** versus the measured main CLI. The serial control shows that moving coordinate formatting to CPU contributes separately from GPU batching.

Memory columns are the higher measured peak across repetitions. Summed RSS double-counts some shared pages; sampled PyTorch allocations can miss short peaks. Container accounting includes warm-up/loading activity and differs from RSS. These quantities must not be added to estimate physical unified-memory usage. Dev has lower measured RSS/container accounting here, but higher CUDA allocation than main’s sampled concurrent total.

## Batch and CPU-worker selection

| Workload | Batch cap | CPU workers per stage | Complete time, median (range), s |
|---|---:|---:|---:|
| pLDDT, 100 inputs | 16 | 2 | 7.88 (7.87–7.88) |
| pLDDT, 100 inputs | 32 | 1 | 8.26 (8.25–8.27) |
| pLDDT, 100 inputs | 32 | 2 | 8.11 (8.09–8.13) |
| pLDDT, 100 inputs | 32 | 4 | 8.20 (8.14–8.26) |
| pLDDT, 100 inputs | 64 | 2 | 8.50 (8.45–8.56) |
| pLDDT, 1,000 inputs | 16 | 2 | 34.53 (34.29–34.78) |
| pLDDT, 1,000 inputs | 32 | 1 | 35.07 (34.98–35.15) |
| pLDDT, 1,000 inputs | 32 | 2 | 33.95 (33.92–33.98) |

The small screen favoured a cap of 16; the full-size follow-up slightly favoured 32 with two CPU workers per stage. Differences between nearby settings are small and two repetitions do not establish a universal optimum. `auto` therefore caps jobs of at most 128 inputs at 16 and larger jobs at 32. The memory estimate can reduce actual batches further: a requested cap of 64 in the screen did not guarantee 64 inputs per batch under the 6-GiB target. Explicit `--batch-size 16` remains a useful lower-allocation option.

## Language checkpoint

Language uses the existing single-model path, with fresh embeddings from the same pinned offline ProtT5 snapshot. The test has five inputs rather than 1,000. It has no new GPU embedding batching or prediction cache.

| Condition | Complete time, median (range), s | Peak summed RSS, GiB | Sampled CUDA allocation, GiB |
|---|---:|---:|---:|
| language-main | 211.94 (171.36–252.51) | 8.298 | 4.547 |
| language-dev | 198.47 (176.53–220.41) | 8.388 | 4.548 |

These language timings remain highly variable. They do not establish a reliable speed improvement; new language batching is explicitly rejected rather than silently applying base-model features.

## Architecture and limits

- One process owns the GPU model. CPU feature preparation and output formatting/writing overlap inference through bounded queues; `--cpu-workers 2` permits two workers per CPU stage. Native PyTorch CPU threads are limited to one in this mode to avoid nested oversubscription.
- Within windows of 512 inputs, sequences are grouped by combined heavy/light length into eight-residue bins. Batches also obey a count cap and an approximate pair-feature budget. Input identities and unique output names survive sorting; duplicate names cannot overwrite a naturally occurring suffixed name.
- Dense features are constructed directly in their final FP32 layout, with exact agreement against main’s CPU features. Padding is masked. Atom conversion is batched and coordinates are transferred to CPU before formatting; full GPU activations and all PDB strings are not retained for the whole workload.
- `--batch-memory-gib 6` is an estimated inference-allocation target, not a hard process/GPU/system-memory ceiling. Its coefficient was calibrated on the small public workload. A CUDA out-of-memory error splits the affected batch, logs retries and eventually fails clearly if a single item cannot fit. Other exceptions propagate. There is no fallback to lower precision or another model.
- Ordinary output uses bounded queues, but CSV/FASTA sequence metadata and batch-audit metadata are still retained in host memory. A single very long sequence can exceed the estimate. Broader loop lengths and diverse real sequences need additional evaluation.
- Chemistry remains optional and follows raw prediction, using main’s explicit PDBFixer/OpenMM/Reduce pipeline and separate raw/prepared/relaxed files. OpenMM GPU work does not overlap structure-model inference. There is no new claim of improved experimental-structure accuracy.
- No CUDA graphs, model compilation, mixed-precision change or output caching is enabled. High CPU/GPU percentages are not an acceptance criterion: completed predictions per minute is the objective.

## Validation

- Every saved benchmark PDB was audited: **34,840 files**, including warm-ups and screens. Counts, atom/residue identities and finite coordinates passed. All serial main/dev comparisons were byte-identical. Batched PDBs had a maximum Cα coordinate difference of 0.008 Å and maximum atom coordinate difference of 0.042 Å versus matched main predictions.
- Fourteen raw-array comparisons covered base/pLDDT, the five public pairs and artificial shorter/longer padding tests under the actual bucketed scheduler. Maximum Cα difference was 0.007956 Å, maximum all-atom coordinate RMS difference 0.001134 Å, and maximum derived pLDDT difference 0.0354 points. Raw Cα acceptance remained below 0.01 Å. The PDB audit allows rounding of the three-decimal coordinates.
- An early unbounded mixed-length stress test exceeded the Cα tolerance; this motivated the narrow length bins. The passing claim applies to the final scheduler and tested inputs, not arbitrary batch compositions.
- Twenty-two CPU/interface tests passed: scheduling/queue boundedness, failure propagation, output formats, duplicate naming, finite options, simulated CUDA OOM splitting, preparation policy and measurement correctness. The final installed CLI also completed real OpenMM preparation, Reduce orientation plus OpenMM preparation, and restrained CUDA minimisation on the public example. Geometry warnings were retained; minimisation met its convergence criterion.
- The 54 core model-source files and all three checkpoints match the retained main runtime. Numeric kernels/features are unchanged between the final default confirmation and published image; the final CLI also adds the small-job cap and collision-safe names. Frozen CLI hashes and image IDs identify each measurement stage.
- Confidence export remains main’s existing limitation: the CLI JSON/filter path contains flattened logits and PDB B-factors are zero. Derived pLDDT agreement above converts the model logits correctly and does not qualify the CLI confidence filter.

## Reproduce

Build this dev checkout with `./docker_build.sh build`; its default image name is `abodybuilder3-dev-spark:latest`. A fresh build uses the public pinned model/runtime recipe and copies the dev CLI, scheduler and batching module into the image. `NO_CACHE=1` disables the Docker build cache.

For strict identity verification, capture the built image with `tools/capture_runtime.py` and compare against `tests/reference_dev_batch_runtime_251026.json`. That manifest describes the final qualified image, including both new modules. The stable preparation reference describes main and is intentionally different. Commit `230ef920bf4879d1b398704b487cca322cd8d895` preserves the pipeline source used by the screen and serial controls; the final image differs in the adaptive small-job cap and collision-safe filename allocation.

The study uses `tools/benchmark_cli.py`, `tools/run_batching_study.py`, `tests/check_batch_gpu.py` and `tools/audit_batching_outputs.py`. The initial controller performs the small screen and large 16-cap comparison; the 32-cap follow-up and final-default rows are separate explicit invocations of the same benchmark tool. All arguments/configurations and warm-up flags are retained in the measurement JSON. `tools/publish_batching_results.py` regenerates this report from the local completed study.

[Measurements and validation records](data/abodybuilder3_dev_batching_measurements_251026.json) · [Main preparation guide](abodybuilder3_preparation_251026.md) · [Historical batching comparison](abodybuilder3_benchmarks_251026.md)
