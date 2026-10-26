# Dev default and next investigations — 26 October 2026

## Accepted direction and current state

The user approved **strict FP32 + saved AOT executable + native writer** as the
default direction for dev on 26 October 2026. This supersedes the earlier
proposal to use cached JIT only above a job-size threshold. Precision must remain
explicit: no automatic switch to TF32, BF16 or FP16.

The [AOT study](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/blob/components/openfold-structure-251026/docs/abodybuilder3_aot_deployment_261026.md), committed as `936cb6f`,
provides the current evidence. It tested both base/pLDDT checkpoints, 100 distinct
experimental antibodies of 219–242 residues, and complete jobs of 100, 1,000 and
10,000 inputs, with two fresh repetitions. Larger jobs repeat those targets;
they are not broader biological validation. AOT improved complete-job time and
CPU RSS, with about 0.132 GiB more peak CUDA allocation than cached JIT.

The approved route is now integrated into dev. Deployment priorities 1–5
passed the scoped checks in the [installed-CLI validation report](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/blob/dev/docs/abodybuilder3_dev_aot_validation_261026.md).
The initial integration is commit `159c8af`, followed by CLI failure checks and
compact evidence. The 150-target panel is independent of our implementation
tuning but is the upstream authors' validation split, not an unseen model test.
Length extremes remain engineering fixtures. Main remains
`a5deeb1b73994491f94b5724d2f2baa9993d47a6` and is outside this change.

The [language lifecycle improvement](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/blob/dev/docs/abodybuilder3_language_lifecycle_261026.md)
is integrated on dev in `49080c3`, retaining exact checked batch-1 embeddings
and predictions. Language AOT and batching are not qualified. The ordered A–J
screens are now complete; see the [results and figures](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/blob/dev/docs/abodybuilder3_ordered_optimisation_261026.md).
Optional chemistry keeps its established serial protocol because final
concurrency qualification found prepared-hydrogen differences.

## Deployment checks, in order

| Priority | Work | Evidence needed |
| --- | --- | --- |
| 1 | Integrate the approved route into the real dev CLI and Docker build | Build from a clean checkout without retained experimental images or caches. Prepare AOT packages on the GPU, then include them and the native writer in the final image. Check image/source/weight/precision identities and run the installed CLI offline, including output writing. |
| 2 | Cover small jobs, shape boundaries and mixed lengths | Test jobs below 100 inputs, partial microbatches, batch size 1 and the 2–32 package domain, and valid inputs around each declared length boundary. Include short and long H3s and heterogeneous length distributions. An input outside the exported domain must be handled explicitly; no truncation or silent precision/model change. |
| 3 | Extend numerical and scientific regression evidence | Add an independently selected experimental panel spanning sequence/loop diversity and lengths beyond the reused 100-target cohort. Preserve target pairing; compare raw arrays and precision-matched eager/AOT outputs, whole-Fv and framework-aligned CDR RMSDs, local distance scores, side-chain torsions/geometry and confidence. Biological targets, technical repeats and artificial engineering fixtures remain distinct. |
| 4 | Qualify failure, output and resource behaviour | Missing/corrupt/incompatible package, OOM batch splitting, unsupported lengths, invalid input, duplicate names, output counts, interrupted jobs and bounded queues. Verify that any strict-eager fallback is explicit and tested, stays strict FP32 and does not silently alter the model. Preserve the existing language/chemistry paths while separately checking their compatibility. |
| 5 | Check sustained operation | Use a sufficiently diverse long job to expose retained metadata, memory growth, queue stalls and filesystem effects. Verify complete output/identity accounting. Record allocated/reserved CUDA, RSS and container/system measures separately, with no addition of overlapping unified-memory quantities. |

These are focused gaps, not a request to repeat all completed tests. Reuse the
existing writer, cache-bypass and paired structure evidence for unchanged code.
Integration needs a bounded real-CLI check; widening shape or feature coverage
requires only the checks relevant to that extension.

### Existing confidence-output issue

Current dev's `scripts/predict.py::extract_plddt_scores` flattens the model's
per-residue confidence logits. The JSON mean and `--confidence-threshold` path
therefore do not compute the intended pLDDT score. PDB B-factors are currently
zero. The structural evaluation reports convert logits correctly; their
scientific confidence comparisons do not validate the production CLI output.

Review a separate correction that computes the expected score from the released
head's bin probabilities, exports one score per residue and applies the threshold
to that score. Define base-model behaviour when no confidence head exists. Test
known logits, masking, JSON and filtering; decide separately whether to populate
PDB B-factors. This changes output metadata and potentially which structures are
written, so it must be discussed as a correctness change rather than presented
as an invisible speed optimisation. No confidence correction is enabled here.

## Optimisation experiments after integration

| Order | Component or approach | Hypothesis and bounded comparison |
| --- | --- | --- |
| A | Profile the complete AOT pipeline | Trace startup, feature construction, pair projection, actual GPU execution, atom conversion, transfers/synchronisation, native formatting and filesystem writes. Include queue occupancy. Use CUDA events or a device timeline for GPU time: the AOT host-forward call returns asynchronously and cannot be read as model execution time. Preserve the normal benchmark path and measure profiling overhead separately. |
| B | Atom conversion and producer/consumer overlap | If the timeline shows conversion or synchronisation on the critical path, test an exported inference-plus-atom-conversion boundary, fewer unused output tensors, and bounded double buffering/asynchronous copies. Check raw arrays and masks against the existing conversion before changing output paths. Spark shared memory does not make ordinary PyTorch copies automatically free or coherent zero-copy accesses. |
| C | AOT shapes and batch scheduling | Compare the current dynamic package with a small bank of fixed/narrow-shape packages and length buckets, then tune microbatch caps and partial-batch handling for AOT. Test only a small profile-motivated grid first. Avoid excessive padding: pair work grows with the square of padded length. Any change in batching or compiler specialization needs paired numerical checks, even at strict FP32. |
| D | Output/metadata streaming | If writing or host-memory growth matters, test batched writes and streaming input/audit metadata while preserving every output name and record. Keep the existing per-PDB format available; an alternative archive or coordinate format would be an explicit output choice, not a silent default change. |
| E | Repeated small jobs | Compare several consecutive small submissions using one model-owning process with launching a fresh process each time. A persistent worker can retain weights and packages, but is useful primarily when that is the real workload. Do not build a network service merely to optimise a large single job. |
| F | Language model | Profile ProtT5/embedding generation separately before designing language batching/AOT. Qualify the language checkpoint independently of the base/pLDDT evidence. |
| G | Kernel/component rewrites | Only when profiling identifies a substantial remaining kernel or third-party bottleneck, consider a targeted Triton/C++ implementation. Preserve the original code on a dedicated component branch, compare original and new outputs, and run regression/accuracy checks. AOT already fuses operations, so another rewrite is not assumed to improve speed. |
| H | Reduce | Separate subprocess startup, dictionary loading, hydrogen placement and optional side-chain optimisation in the profile. Test bounded CPU parallelism across antibodies and reduced temporary-file/subprocess overhead where supported. Consider an internal rewrite only for a measured bottleneck, retaining identical hydrogen/orientation decisions and output semantics. |
| I | PDBFixer and hydrogen preparation | Profile parsing, topology/repair, missing-atom detection and hydrogen addition separately. Evaluate worker-local reuse of immutable force-field/template data and removal of redundant conversions or repeated scans, with equivalence checks. Preserve required repair, protonation and disulfide handling; do not skip chemistry checks to claim a speedup. |
| J | OpenMM and optional-stage scheduling | Profile system/context creation, force-field parameterisation, hydrogen optimisation and restrained minimisation. Test safe reuse of compatible systems/contexts with complete state reset, CPU worker/thread counts and explicit GPU scheduling. Compare bounded CPU preparation overlap with inference against phased GPU minimisation; measure whole-job throughput and peak memory, since OpenMM and inference share the GPU. |

Optional-stage optimisation (H–J) is appended at the user's request on
26 October 2026 and follows the prediction/deployment priorities above. Start
with a small matched profile of preparation-only and preparation-plus-relaxation
jobs, then prioritise changes within H–J by their measured contribution. Existing
dependency upgrades and CUDA enablement are the baseline, not new optimisation
results.

Keep the established chemical protocol, including pH, side-chain-flip policy,
force field, restraints, convergence criteria, seeds and precision, fixed in
performance comparisons. Retain raw predictions and compare prepared atoms,
coordinates, protonation/orientation decisions, energies, forces, convergence and
geometry. Use paired repeats to distinguish a candidate's differences from the
baseline's own variability. Any third-party code changes require a dedicated
component branch containing the original and modified code and comparison and
regression tests. Discuss changes that alter results before enabling them.
Optional preparation must remain optional and must not slow plain prediction.

Earlier eager CUDA-graph replay, larger-batch and extra-worker screens did not
show a dependable gain. AOT changes the balance, but those experiments should be
revisited only for a specific reason visible in the new profile. The strict-FP32
point-attention replacement had no consistent full-job benefit; TF32 and more
aggressive reduced precision are lower priorities than qualifying this default.

## Progress through the optimisation list — 26 October 2026

- **A:** completed a CUDA/NVTX/kernel and queue profile with uninstrumented and
  instrumented controls. All profiled outputs were exact. GPU model execution
  dominates; CPU atom-conversion wall time largely waits for GPU completion.
- **B:** deferred pinned transfers preserved all raw/PDB bytes. At 10,000 inputs
  the measured gain was only 0.6–1.2%; the extra machinery remains isolated.
- **C:** completed a six-condition batch/shape screen. Wider batch domains,
  caps 16/32/64/128 and fixed length banks gave no convincing speed gain.
  Batch 128 used about 9.3 GiB; no package/default change was retained.
- **D:** the profile shows output already overlaps inference. The 20,000-job
  trace found modest metadata growth, not a current output bottleneck. No new
  writer/metadata rewrite is justified by these measurements.
- **E:** a local resident-process prototype preserved all outputs and removed
  repeated startup. Ten five-input requests took about 5.8 seconds rather than
  52 seconds in fresh containers. It remains an isolated local-worker prototype;
  it does not accelerate an already combined CSV or create a network service.
- **F:** the [language lifecycle optimisation](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/blob/dev/docs/abodybuilder3_language_lifecycle_261026.md)
  passes exact embedding/structure/PDB comparisons and is integrated on dev.
  It keeps the original batch-1 encoder on GPU, replacing repeated Trainer
  teardown. No embedding batching, precision change or weights change is enabled.

- **G:** the profile-motivated geometry-update alternative was exact but about
  14–15× slower in a standalone microbenchmark. No kernel rewrite was retained.
- **H:** Reduce itself took about 0.7 s per antibody, versus 8–9 s in OpenMM
  hydrogen addition. Neither a Reduce rewrite nor caching its 0.002 s version
  subprocess is justified by the measured whole-job contribution.
- **I:** bounded independent CPU-1 preparation processes improved complete
  eight-input CLI jobs by about 3× at four workers and 4.4–4.6× at eight.
  Initial paired PDBs matched; an independent final check and float-array trace
  found small repaired-atom/hydrogen differences. The candidate remains isolated.
- **J:** two concurrent preparation/relaxation workers improved a five-input
  group about 1.8×, but changed relaxed outputs. More CPU threads within a single
  hydrogen engine were faster too, with much larger hydrogen differences.
  Both remain isolated; no chemistry engine/settings/default change is enabled.

The [ordered report](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/blob/dev/docs/abodybuilder3_ordered_optimisation_261026.md) links the
[pipeline component evidence](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/blob/components/aot-pipeline-261026/docs/aot_pipeline_optimisation_261026.md)
and [optional-stage component evidence](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/blob/components/structure-preparation-261026/docs/preparation_optimisation_261026.md).
Original and candidate code, tests, all images and local evidence are retained.

**Next decision:** review the optional-stage repeatability differences before
enabling concurrency or commissioning a deterministic-repair comparison.
Resident requests remain a separate local prototype. Confidence export remains
a separate correctness change for discussion. No result-changing candidate is
silently promoted.

## Comparisons and presentation

Freeze the integrated AOT default as the new optimisation reference. Change one
intended factor at a time, with matched inputs/order, checkpoint, precision,
writer and relevant batch shapes. Where shapes deliberately change, label that
factor rather than calling it an exact matched-shape comparison. Interleave
conditions, retain individual timings and increase repetitions only when a small
effect needs resolving. Choose completed-job latency/throughput and measured
memory trade-offs over CPU/GPU utilisation alone.

Use a CPU/GPU timeline for the initial profile, throughput-versus-memory plots
for scheduling, and paired per-antibody numerical/experimental-error
distributions for changes in arithmetic. Retain meaningful tables and CSV/JSON
data beside the figures. Score independent targets once in distribution plots;
fresh-process repetitions assess repeatability rather than double the cohort.
Local-structure and tail metrics must accompany a mean whole-Fv RMSD.

Keep result-changing candidates isolated until their differences are reviewed.
Do not describe small numerical differences as either proven accuracy gains or
universally harmless. Do not invent expected speedup percentages before a
controlled measurement.

## Primary technical references

- [PyTorch AOTInductor export and deployment](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_aot_inductor.html): the supported saved-executable route; the pinned runtime's APIs were checked directly during the completed study.
- [Nsight Systems CUDA tracing](https://docs.nvidia.com/nsight-systems/UserGuide/): distinguishes host API calls, device kernels and memory operations on the timeline.
- [NVIDIA NVTX](https://docs.nvidia.com/nvtx/overview/): annotations connect application stages with those traces.
- [DGX Spark memory reporting](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html): platform-specific unified-memory accounting limits.

This document records the accepted default and the scoped deployment validation.
The progress section distinguishes completed screens from work still proposed.
The separate confidence-output correction remains disabled.
