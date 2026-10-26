# Ordered AOT pipeline experiments — 26 October 2026

The integrated strict-FP32 AOT/native route remains the reference. These screens
found no compelling reason to replace its transfer path, package bank or batch
policy. A resident-process prototype substantially reduces the cost of repeated
small requests, but is separate from the installed CLI. The later language
lifecycle improvement is documented on [its own component branch](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/blob/components/prott5-runtime-261026/docs/prott5_lifecycle_261026.md).

## A: identify the critical path

![CPU/device timeline and bounded queues](assets/aot_pipeline_timeline_261026.png)

The uninstrumented 1,000-input pLDDT job took 13.183 seconds; the CUDA-event/NVTX
wrapper took 13.213 seconds. Collecting and exporting a Nsight trace increased
the complete command to 17.989 seconds. All three produced identical raw outputs
and PDBs. Profiled duration is not a new prediction benchmark.

The instrumented model-forward intervals covered 7.22 seconds on the CUDA
stream. Host atom conversion/download covered 7.04 seconds, but its own stream
intervals totalled only 0.138 seconds: much of that host time waits for the
preceding model. Treating it as seven seconds of conversion computation would
misidentify the bottleneck. Host intervals, device intervals and nested writer
calls overlap and must not be added together.

Weight loading took 0.84 seconds and package loading 0.24 seconds. Feature
construction and output work overlapped inference; the ready queue usually
contained one or two batches. Output-batch wall intervals totalled 1.80 seconds,
including 0.96 seconds in native formatting. The device trace is dominated by
matrix kernels and fused geometry/copy kernels. The raw Nsight report and SQLite
remain local; the archive contains timing aggregates and queue events, without
profiler environment tables.

## B–E: compare complete jobs

![Transfer, shape and resident-process screens](assets/aot_pipeline_screens_261026.png)

All comparisons used the same released weights, strict FP32 and native writer,
with two fresh repetitions per condition. The 1,000- and 10,000-input workloads
repeat the established 100-target cohort. They measure throughput, not additional
biological diversity. Source hashes, protocols and individual measurements are
in the [JSON](data/aot_pipeline_measurements_261026.json) and
[checksummed evidence archive](data/aot_pipeline_evidence_261026.zip).

### B: deferred copies and writer overlap

The candidate pins host features, enqueues nonblocking copies and delays CPU
access until a writer resolves an event-backed result. Queues and buffers stay
bounded. Both conditions collect fingerprints in output workers, so measurement
does not force the candidate to synchronise early.

| 10,000-input job | Current, two runs (s) | Deferred, two runs (s) | Median speedup |
|---|---:|---:|---:|
| base | 80.584, 80.917 | 80.141, 80.357 | 1.006× |
| pLDDT | 81.234, 81.687 | 80.645, 80.334 | 1.012× |

Four CPU tests and 14 GPU qualification cases passed. All paired raw arrays and
PDB bytes matched, including the larger jobs. Peak CUDA allocation remained
about 2.390 GiB. The 1,000-input screen showed similarly small, inconsistent
changes. A 0.6–1.2% difference over two repetitions does not justify the added
synchronisation machinery; this candidate remains isolated.

### C: batch caps and shape banks

The screen compared the accepted batch-32 package with a newly exported dynamic
batch domain supporting caps 16, 32, 64 and 128, plus fixed-length banks at 240
and 248 residues. The latter pad shorter inputs and preserve masks and original
lengths; inputs outside their domain use explicit dynamic routing.

No condition gave a convincing complete-job speed gain. Batch 16 used about
1.213 GiB, batch 32 2.390 GiB, batch 64 4.668 GiB and batch 128 9.300 GiB of
CUDA allocation. Fixed-length routing was slower, taking about 14.4 seconds
instead of 13.0–13.2 seconds for 1,000 inputs. Wider batches would need more
memory without a demonstrated benefit in this workload.

All numerical gates passed, but the newly compiled packages were **not
byte-identical** to the accepted packages. Mean aligned Cα RMSDs were roughly
1–2 × 10⁻⁶ Å; the largest Cα displacement was below 0.00014 Å. These are
implementation differences, not evidence of improved accuracy. No package or
default was replaced. The existing explicit `--batch-size 16` remains available
without requiring these experimental packages.

### D: metadata/output streaming

The timeline does not show a writer bottleneck. The earlier 20,000-input
qualification completed every output; sampled process RSS rose modestly from
about 1.571 to 1.599 GiB across its selected time windows, including retained
audit and measurement metadata. This does not establish constant-memory
streaming, but it provides no current reason for another output rewrite.

### E: repeated small submissions

The prototype retains the model and loaded executables, restoring the original
forward method and resetting request counters between calls. It caches no
predictions. Three state-isolation tests passed; all request outputs matched
fresh-process execution exactly.

| Requests | Model | Fresh processes, median total (s) | Resident process, median total (s) | Speedup |
|---|---|---:|---:|---:|
| 10 × 5 inputs | base | 52.27 | 5.75 | 9.09× |
| 10 × 5 inputs | pLDDT | 52.62 | 5.88 | 8.95× |
| 5 × 100 inputs | base | 30.37 | 9.41 | 3.23× |
| 5 × 100 inputs | pLDDT | 31.07 | 9.53 | 3.26× |

These totals include one startup per fresh request versus one initial startup
for the resident group. They exclude client waiting time. Combining a known
workload into one CSV already avoids repeated startup; the result is not a
ninefold acceleration of one combined inference job. Peak CUDA allocation was
unchanged, about 0.176 GiB for five-input jobs and 1.213 GiB for 100-input jobs.
This remains a local prototype, not a deployed service or production job queue.

## G: test a measured geometry kernel before rewriting it

The fused distance-feature update appeared prominently in the device profile.
A bounded microbenchmark compared the existing strided update of the last pair
channel with rebuilding the whole pair tensor by concatenation. Both use the
same compiled FP32 distance calculation.

| Batch × length | Existing update, median (ms) | Concatenation, median (ms) |
|---|---:|---:|
| 32 × 232 | 0.564 | 8.130 |
| 32 × 248 | 0.656 | 8.968 |
| 64 × 232 | 1.203 | 17.889 |

Outputs were exact, but concatenation was about 14–15 times slower. These are
standalone update timings, not whole-model speedups. The current implementation
was retained. A deeper pair-projection rewrite would change arithmetic and needs
a separate proposal and accuracy comparison; none is enabled here.

## Reproducibility and decisions

The reference is dev `d3c7444` and image
`sha256:2774ec553f5653e8ba97c761824f55b6d6c49b5df3573382079a23a1ae912354`.
Original adapters are in `reference/`; alternatives, controllers and regression
tests are retained on this branch. No datasets, predicted coordinates, weights
or AOT binaries are committed. Plots can be regenerated with
`tools/publish_pipeline_results.py --runs <local-study-parent>`.

No B/C/G candidate is promoted. E is a useful exact-output prototype for a
specific workload, but needs an explicit local job interface and operational
qualification before deployment. Optional Reduce/PDBFixer/OpenMM screens follow
these priorities and have [their own report](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/blob/components/structure-preparation-261026/docs/preparation_optimisation_261026.md).
Main remains unchanged. All images and local evidence are retained.
