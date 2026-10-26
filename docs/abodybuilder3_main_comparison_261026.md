# Previous and new main: matched deployment comparison — 26 October 2026

The new main is about **6.3× faster** for both 1,000-input base/pLDDT jobs and **21.5× faster** for the five-input language job. Summed process-tree RSS falls by 89% and 33% respectively. CUDA allocation rises; the deployment is faster with lower process RSS, not uniformly lower memory by every measure.

![Matched complete runtime and summed RSS](assets/abodybuilder3_main_comparison_261026.png)

## Protocol

Two technical repetitions per version and model ran consecutively on the DGX Spark, driver 580.178.04, in fresh offline GPU containers. Replicate 1 ran previous then new, replicate 2 reversed that order. All 12 runs completed. Timing surrounds `docker run` and includes container startup/exit, imports, checkpoint and executable loading, inference and all output writes. No build/export, download, chemical preparation or accuracy scoring is timed. No separate whole-job warm-up was run; these measure fresh processes on an already running host, not a cold operating-system cache.

Previous main is `a5deeb1b73994491f94b5724d2f2baa9993d47a6`, preserved at `pre-aot-261026`; the retained `preparation-251026` image contains its exact CLI, preparation module, structure module and utilities. New main is `78163003c6c18aa0557b23aae91848911d40553b`, tag `optimised-runtime-261026`, measured in the newly built `accepted-runtime-261026` image. Full image IDs and source references are in the measurement JSON. All three released checkpoints and the pinned offline ProtT5 snapshot are the previously qualified assets.

Both commands use CUDA, seed 42, PDB output, and their installed defaults for precision, workers, batching and writers. Previous main retains its legacy TF32-permitted policy and uses eight model workers for the 1,000-input jobs. New main uses strict FP32, a single batched saved executable and the native writer. Five language inputs use the single-model path in each version; new main keeps ProtT5 on GPU. This estimates the complete default-deployment change, not any isolated optimisation.

Base/pLDDT use identical 1,000-row CSVs across versions, cycling 100 distinct public antibodies ten times. Language uses the same five public examples across versions. Input SHA-256 hashes are recorded. Repeated inputs test throughput; they are not independent biological replicates. Predictions and embeddings are computed afresh, with no sequence-result cache. Every expected PDB filename, complete ending and finite coordinate was checked after timing: **8,020 files** in total. Existing numerical/experimental qualification is reused; this comparison makes no new structural-accuracy claim.

## Summary and uncertainty

| Model · inputs | Time: previous → new | Speedup | Peak summed RSS: previous → new | RSS reduction |
|---|---:|---:|---:|---:|
| Base · 1,000 | 81.9 → **13.0 s** | **6.3×** | 14.27 → **1.54 GiB** | **89%** |
| pLDDT · 1,000 | 82.8 → **13.2 s** | **6.3×** | 14.25 → **1.57 GiB** | **89%** |
| Language · 5 | 242.5 → **11.3 s** | **21.5×** | 8.35 → **5.59 GiB** | **33%** |


Times are medians; memory is the higher sampled peak across the two runs. The figure's time whiskers show min/max, not a confidence interval. Previous language runtimes were 262.26 and 222.73 seconds versus 11.53 and 11.03 seconds for new main; its speedup is descriptive and workload dependent. Two technical repetitions cannot establish precise uncertainty or general scaling rules.

Summed process-tree RSS was sampled every 50 ms with the same measurement wrapper. It counts shared pages separately for each process; eliminating seven model processes reduces duplicated RSS accounting as well as private allocations. These values do not quantify total physical unified-memory savings. Container cgroup peaks are also retained but are not used in the headline: page-cache charging and unified-memory accounting differ from process RSS.

CUDA allocations were sampled every 100 ms and exact per-process lifetime allocator peaks were captured at exit. Sampling misses short peaks: for new base/pLDDT it reports about 0.90 GiB even though the exact peak is **2.39 GiB**. Consequently sampled CUDA values are not used to claim a memory reduction. Previous workers' summed individual peaks are about **1.56 GiB**, an upper bound rather than an exact concurrent maximum; new main uses more allocator memory even relative to that bound. Language's exact single-process peak rises **4.55 → 4.69 GiB**. CUDA, RSS and cgroup accounting overlap and must not be added.

## Reproduction and records

- [Individual runs, protocol, input hashes, image IDs and summary](data/abodybuilder3_main_comparison_261026.json).
- [Plot generator](../tools/plot_main_comparison.py).
- [Host queue](../tools/main_comparison/run_comparison.py) and [installed-CLI measurement wrapper](../tools/main_comparison/measure_installed_cli.py).
- The queue accepts `--study` and `--prott5`. A fresh study directory needs `inputs/antibodies_5.csv`, `inputs/antibodies_1000.csv` matching the recorded hashes and `protocol_261026.json`; its `tools` directory contains the measurement wrapper, `benchmark_runtime.py`, and `instrumentation/sitecustomize.py`. Inputs, exact commands, per-process samples, logs and outputs are retained in the local study below. The script refuses existing run directories, preserving completed evidence.
- Local study: `/home/adrian/codex_wd/projects/abodybuilder3/runs/main_readme_benchmarks_261026`.

The production README includes only this accepted old/new comparison. Detailed experiments remain on dev and component branches. No source, model settings, images or existing benchmark records were changed to obtain these measurements.
