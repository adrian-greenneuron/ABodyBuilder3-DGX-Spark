# Exact-output dev inference optimisations — 25 October 2026

**Historical measurements.** The [26 October saved-AOT deployment report](abodybuilder3_dev_aot_validation_261026.md) describes current dev; the results below retain their original runtime and precision.

Dev now avoids dense one-hot pair inputs, unused intermediate side-chain outputs and repeated copies of the static pair representation. On the tested Spark, complete 1,000-input jobs took about **14% less time**, with approximately **16–17% more throughput**, **57% lower sampled process RSS** and **19% lower peak CUDA allocated memory** than the preceding dev implementation.

**Every matched raw prediction and PDB was identical.** Main remains at `a5deeb1b73994491f94b5724d2f2baa9993d47a6`; no main files, checkpoints, chemistry dependencies or model settings were changed.

## Complete-job runtime

Each condition ran in a fresh offline container, including imports, model loading, features, inference and writing every PDB. One whole-job warm-up preceded two measured repetitions; conditions were interleaved and the final order reversed. Values are medians with ranges, in seconds. Every condition used the same raw-output fingerprint instrumentation, automatic batch caps and two CPU workers per stage.

| Workload | Model | Previous dev, s | Optimised dev, s | Speedup | Runtime reduction |
|---|---|---:|---:|---:|---:|
| 100 repeated inputs | base | 7.99 (7.98–8.01) | **7.54 (7.51–7.58)** | 1.059× | 5.6% |
| 100 repeated inputs | plddt | 8.15 (8.02–8.28) | **7.56 (7.38–7.74)** | 1.077× | 7.2% |
| 1,000 repeated inputs | base | 33.08 (33.04–33.13) | **28.43 (28.22–28.64)** | 1.164× | 14.1% |
| 1,000 repeated inputs | plddt | 33.79 (33.44–34.14) | **28.93 (28.92–28.95)** | 1.168× | 14.4% |
| 1,000 distinct synthetic inputs | base | 33.38 (33.32–33.45) | **28.56 (28.20–28.91)** | 1.169× | 14.5% |
| 1,000 distinct synthetic inputs | plddt | 33.96 (33.91–34.01) | **29.12 (28.90–29.35)** | 1.166× | 14.2% |

The repeated workload cycles the same five public antibody pairs, with 227–231 combined residues. The distinct workload contains 1,000 deterministic synthetic pairs made by two substitutions in each chain, retaining those lengths and cysteines. It tests implementation and throughput on distinct sequences; it is not a biological accuracy dataset or a broad real-antibody benchmark. There is no prediction cache. The projection table caches only sequence-independent chain/relative-position features, using the original learned weights.

The five-input installed-CLI smoke checks passed before scaling. Their timings are excluded from the performance table because they had no whole-job warm-up. Two timed repetitions do not establish a precise small performance difference or a universal speedup.

## Memory for 1,000-input jobs

Values are the higher peak across measured repetitions, in GiB. CUDA allocated peaks are PyTorch's exact per-process allocator peaks; these jobs have one GPU-owning process. RSS is sampled process-tree RSS. Container accounting is a separate lifetime measure.

| Workload | Model | Sampled process RSS, previous → new | CUDA allocated, previous → new | Container accounting, previous → new |
|---|---|---:|---:|---:|
| repeated | base | 4.157 → **1.789** | 4.685 → **3.782** | 3.994 → **1.638** |
| repeated | plddt | 4.161 → **1.789** | 4.686 → **3.783** | 4.019 → **1.631** |
| unique | base | 4.155 → **1.790** | 4.685 → **3.782** | 4.012 → **1.637** |
| unique | plddt | 4.171 → **1.800** | 4.686 → **3.783** | 4.030 → **1.634** |

**Do not add these columns** to estimate total physical unified-memory usage. RSS sampling can miss short peaks, and PyTorch CUDA accounting excludes allocations made outside that allocator. The compact measurements retain sampled CUDA concurrency, exact allocated/reserved peaks and main-process lifetime RSS as well.

## Changes retained

1. **Projection lookup.** Each pair input contains one chain-category indicator and one relative-position indicator. A small table evaluates all combinations with the unchanged learned linear projection, including bias and the all-zero padding case. Inference indexes that table instead of constructing and transferring a 132-channel dense tensor. The table is refreshed when the projection weights/bias change through ordinary PyTorch updates. It is restricted to inference.
2. **Final-only side-chain geometry.** All eight learned attention/backbone updates remain. Side-chain heads and coordinate/output reconstruction run only for the final update; their earlier outputs do not feed the next update. The ordinary model forward API still returns the original intermediate outputs by default.
3. **Pair-buffer reuse.** The static projected pair channels are copied once. Each update replaces only its distance channel, rather than concatenating/copying the entire representation again. The attention inputs retain the same values and layout. Reuse is forbidden in the unsupported offload path and with gradients/training enabled.

The matched pLDDT complete-job ablation supports retaining each step:

| Implementation, 1,000 repeated inputs | Complete time, median (range), s | Peak CUDA allocated, GiB |
|---|---:|---:|
| Previous dev | 33.79 (33.44–34.14) | 4.686 |
| Projection lookup | 31.78 (31.49–32.07) | 3.846 |
| Lookup + final-only geometry | 31.07 (30.87–31.26) | 3.783 |
| Lookup + final-only geometry + pair buffer | 28.93 (28.92–28.95) | 3.783 |

## Profiling evidence

The diagnostic serial profile labels CPU features, transfer, model operators, atom conversion and PDB formatting. It does not reproduce the production pipeline's CPU/GPU overlap. In its 64-input profiled pass, CPU feature construction fell from about 471 to 44 ms, while GPU time attributed to `aten::cat` fell from about 170 to 18 ms. Large elementwise point-attention intermediates remain a major GPU cost.

Profiler overhead affects these figures. Host wait time can move between labels when transfer/synchronisation changes; a longer host atom-transfer label is not by itself slower atom conversion. Inclusive ranges and CPU/device times must not be summed. The unprofiled fresh-container comparisons above determine acceptance. Full traces stay in the local study; compact operator summaries and source hashes are included in the measurement JSON.

## Prediction and runtime checks

- **18,610 matched candidate predictions**, including warm-ups: identical raw coordinate/mask/confidence fingerprints and PDB bytes. The complete study produced and checked 31,220 PDB outputs across controls, candidates and warm-ups.
- **960 strict deployed GPU comparisons**: base/pLDDT, caps 1/4/16/32, public pairs, artificial length changes and distinct substitutions; zero raw-coordinate differences. Projection-only, projection-plus-final-geometry and the combined path were checked separately.
- The isolated component comparison also checks exact final model states against the untouched reference. Compatibility controls pass with the original default API.
- Twenty-four focused CPU/interface/instrumentation checks passed across the applicable invocations, covering compact-feature padding, weight/bias cache invalidation, inference-only guards, retained updates, output formats/naming, bounded queues, error propagation and fingerprints sensitive to changes below PDB rounding precision.
- All three checkpoint hashes, all installed Python dependency versions, OpenMM identities, Reduce binary/dictionary and the other captured runtime sources match the previous dev runtime. Only the declared inference files differ.

The tests establish agreement with the previous implementation on these inputs, not accuracy against experimental structures or a promise of bitwise identity on every device/software version. The earlier batching differences relative to main remain separate; this optimisation adds no differences against the matched previous-dev batches.

The runtime retains FP32 tensors/weights and unchanged NGC settings, including `float32_matmul_precision='high'` and `allow_tf32=True`. FP32 therefore does not mean that every matrix multiplication uses strict IEEE FP32 arithmetic. Driver 580.178.04, kernel 7.0.0-1019-nvidia, PyTorch 2.10.0a0 NGC 25.11 and CUDA 13.0 were unchanged.

## Faster experiments held for discussion

- **Weighted-point matrix multiplication:** about 1.12× faster in the serial 64-input component screen, but Cα coordinates changed by up to 0.0443 Å. It is absent from the deployed dev module.
- **Compilation:** about 1.65× faster in the warmed 32-input pLDDT component screen, but Cα coordinates changed by up to 0.0093 Å. Compilation/startup costs were excluded, the public-only numerical screen was small, and no complete-job compiler speedup is claimed. Compilation is not enabled on dev.

Both fail the current exact-output acceptance gate and require discussion before adoption. No precision reduction, approximate model, new checkpoint or silent fallback was used. The unmodified reference, accepted subset, experimental candidates and comparison/regression tests are isolated on [components/openfold-structure-251026](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/tree/components/openfold-structure-251026).

## Reproduce and provenance

Baseline dev is commit `07bfe1eb7c01eae955835fee69e38a8ab3f23b7e`, image `sha256:0a3ffb4edcef8d608d65eca49787fcb4ea820098a63e6dfdca853baee0527afa`. The qualified optimised image is `sha256:18456b5147ea1ec35d8de4f664962baeea6af1bb473ad9107bb1cbfa37fd844d` (`abodybuilder3-dev-spark:optimised-251026`). The Dockerfile builds the reviewed inference subset from this checkout on the same pinned runtime/source foundation; no retained local image is required to rebuild it.

`tools/prepare_benchmark_inputs.py --synthetic-unique` prepares the distinct stress inputs; its manifest records their hash and purpose. `tools/run_optimisation_study.py` queues the finite interleaved comparisons and fails on any raw/PDB mismatch. Its `frozen_baseline`/`unique_baseline` directories must use the baseline CLI; the candidate directories use this checkout's CLI. `tools/benchmark_cli.py` records exact commands and image IDs. Measurement-only `--inference-mode` selects the ablations; it is not a prediction CLI option. `tools/profile_inference.py` produces the labelled profile, and `tests/check_optimised_gpu.py` performs the strict GPU check.

Use [the new runtime reference](../tests/reference_dev_optimised_runtime_251026.json) with `tools/capture_runtime.py` and `tools/verify_runtime.py`. Keep the older batching reference for its historical measurements. All existing images, benchmark inputs, outputs and logs were retained locally. The language/chemistry paths were not optimised in this study; their existing measurements and limitations remain applicable.

[Compact measurements and provenance](data/abodybuilder3_dev_optimisation_measurements_251026.json) · [Previous dev comparison](abodybuilder3_dev_batching_251026.md)
