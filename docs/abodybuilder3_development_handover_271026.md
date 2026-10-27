# ABodyBuilder3 development handover — 27 October 2026

## Start here

The next useful step is **a clean rebuild and installed-CLI qualification of the
latest dev chemistry**, followed by broader chemical validation. The prediction
optimisations are already shipped on main. Do not restart the old optimisation
list from its beginning or assume that experimental chemistry is still awaiting
integration.

This handover synthesises decisions across the work, rather than reproducing
benchmark tables. It describes the state before its own documentation commit:

| Item | State |
|---|---|
| Main | `c20d119957bcbbd24d8534fddbdc3e9fc7d57795`: accepted prediction deployment and repertoire README |
| Dev | `bb0cb56715fb0c3fde34b1b9630e9d0a29cd1e99`: same prediction foundation, plus deterministic repair, GPU hydrogen geometry and default local rescue |
| Main image | `abodybuilder3-official-spark:latest`, also `:accepted-runtime-261026`; `sha256:ab981f1494f2c817995a3cf767702181330af8d3f97584120d533dedd9432fd0` |
| Dev image | `abodybuilder3-dev-spark:latest`, also `:rescue-271026` and `:rescue-integration-271026`; `sha256:deca5b8a477bbca7ed60b0f929efb4cb4e76df65d06097ae3cace2b99195cce2` |

No ABodyBuilder3 experiment is running or queued at handover. Proposed work below
has not been launched. Main remains unchanged by this handover. Further promotion
of dev chemistry to main has not been authorised.

**Historical reports describe their state at the time.** Statements such as
“main unchanged”, “chemistry stays serial”, or “all outputs/images retained” in
25–26 October documents are superseded by the main promotion, 27 October rescue
integration and subsequent user-authorised storage cleanup. Preserve those
reports as evidence; use this document, current code and the
[rescue integration record](abodybuilder3_rescue_integration_271026.md) for current
status.

## What we achieved, and why it worked

### Prediction: remove redundant work and deployment overhead

We first verified the historical installation against a pinned rebuild, then
modernised its dependency packaging. The original and modernised routes used
the same released base, pLDDT and language checkpoints. The main gains came
later from execution and lifecycle changes, not retraining or changing weights.
The original recipe remains on the GitHub tag `original-runtime-251026`.

The accepted base/pLDDT path has one GPU model process, length-aware batches,
bounded CPU feature/output queues, compact pair-feature projection lookups,
reuse of sequence-independent buffers, and side-chain computation only at the
last learned structure update. **All eight learned attention/backbone updates
remain.** A native writer replaces Python formatting with identical formatting
for the same supplied coordinates.

Cached `torch.compile` kernels still needed costly per-process setup. Saving an
AOT executable moved that work to image preparation and removed the earlier
job-size crossover as the deployment policy. Strict FP32 was chosen explicitly;
TF32 permission gave no compelling speed benefit in these comparisons. The
accepted implementation does not include the alternative point-attention rewrite.
AOT still changes floating-point rounding: do not promise bitwise identity with
eager execution or across independently exported packages.

Two different comparisons answer different questions:

- The installed strict-FP32 AOT/native combination took about 13 seconds for
  1,000 inputs versus about 30 seconds for matched strict-FP32 eager/Python.
  On the separate 150-antibody validation panel, mean aligned Cα RMSD was about
  `1.6e-6 Å`, with negligible paired changes in experimental error metrics.
- The complete new-main defaults were about 6.3× faster than previous-main
  defaults on the earlier 1,000-input workload. That comparison includes worker,
  precision, batching and writer changes; it is not an isolated AOT speedup.

The stronger scale result is **100,000 distinct native heavy/light pairs**:
approximately **7,726 structures/minute for base and 7,646 for pLDDT**, including
fresh-container startup and every PDB write. Additional unified-RAM pressure
peaked around 4.63 and 4.37 GiB. This is prediction only, not preparation. The
study-balanced panel is not a population-frequency sample; this was not a
million-antibody test or an independent structural-accuracy benchmark.

For language inference, the bottleneck was repeated Lightning teardown moving
the 1.2-billion-parameter FP32 ProtT5 encoder off the GPU between chain calls.
Keeping the same batch-1 encoder resident removed that overhead, with exact
checked embeddings and structures. The initial five-input comparison improved
about 19×; a separate installed-CLI comparison also passed. Language batching,
AOT and reduced precision are **not** implemented by that change.

### Optional chemistry: define the operation before making it faster

The old `--relaxation` flag added hydrogens; it did not perform full structural
minimisation. It now fails with an explanatory error. Preparation
(`--add-hydrogens`) and genuine whole-structure restrained minimisation
(`--relax restrained`) are separate operations with separate output files and
audits. Raw predictions remain untouched.

Modernising Reduce changed optional output because the newer executable
succeeded where the old wrapper had silently fallen back to PDBFixer. The
supported pipeline therefore uses explicit methods and reports failures rather
than silently substituting a backend. Reduce is an optional orientation step;
its hydrogens are discarded and OpenMM supplies the final topology/protonation
and hydrogen placement.

Profiling redirected effort away from rewriting Reduce: it took roughly
0.7 seconds per antibody versus 8–9 seconds in the old OpenMM hydrogen step.
Preparation concurrency initially looked exact, but an independent check traced
a tiny repaired-atom difference to PDBFixer's CUDA missing-atom minimisation.
That difference could cross a PDB rounding boundary. Dev now uses seeded,
one-thread deterministic CPU missing-atom repair. Version/hash-checked
OpenMM/PDBFixer API patches expose the required platform controls; this is not
a wholesale rewrite of either package. Original and changed third-party code
and regression evidence remain on component branches.

Hydrogen failures then exposed a deeper issue. Severely strained, fixed heavy
atoms can create enormous nonbonded energies that hydrogen movement cannot fix.
More minimisation iterations alone did not solve stretched bonds, wrong-side
hydrogens or out-of-plane sp² hydrogens. The adopted pH-7 geometry objective uses
fixed force-field X–H lengths, angular variables, plane/outward constraints,
chirality checks and bounded sterics. **It omits Coulomb and GB terms and is not
physically equivalent to full Amber minimisation.** Checked geometry does not
establish experimentally optimal hydrogen positions or a clash-free structure.

CPU and GPU implementations passed the geometry screens on 509/512 distinct
pairs. Three had incompatible heavy geometry and were explicitly rejected.
The frequently quoted 127 is the compatible subset of the original 128-input
panel, not an arbitrary sample size. The complete 512 contains that panel plus
256 and 128 additional distinct pairs; later tuning used their failures, so it
is not an untouched final test set.

GPU execution retained a substantial full-job advantage over corrected CPU
preparation. Worker and streaming experiments led to **phased preparation after
inference**, with up to 12 isolated preparation workers, rather than trying to
maximise simultaneous use of the same GPU by every stage. CPU/GPU prepared
hydrogens can differ, including local outliers; raw neural predictions and
compatible heavy atoms remain unchanged.

Local rescue now repairs only the rejected heavy-atom neighbourhood. Restrained
rescue is an explicit, more extensive alternative; local failure never silently
escalates. Both rescued all three known rejected cases. In two-repeat, full
512-input comparisons, default local rescue achieved about **146 prepared
structures/minute** for OpenMM and **129/minute** for Reduce + OpenMM. It added
less than 1% median time over controls that succeeded on only 509/512. The other
509 outputs stayed byte-identical. These rates must not be confused with the
~7,700/minute prediction-only result.

The installed integration passed 39 focused tests, six-case checks containing
all three rejected cases, and raw-prediction preservation checks. Restrained
results are checked by geometry/topology rather than demanded to repeat bytes.
Unrelated CUDA-initialisation failures remain in the record; unchanged successful
reruns do not erase them or justify automatic backend fallback.

## Operational contract to preserve

- **Model precision:** strict FP32 by default. Precision is explicit; no automatic
  TF32/BF16/FP16 switch. Chemistry has its own recorded numerical policy.
- **AOT coverage:** base/pLDDT, batch 1 and batches 2–32, padded lengths 200–272.
  Out-of-domain shapes use the same strict-FP32 eager model with explicit warning
  and counts. Missing/corrupt/incompatible packages fail; they are not silently
  compiled or replaced. No input truncation.
- **Batching:** automatic cap 16 for at most 128 inputs, otherwise 32, with a
  memory estimate that can reduce it. One GPU model process; two CPU workers per
  prediction stage. A memory target is not a total-system hard limit.
- **Language:** original batch-1 model; `--embedding-runtime trainer` remains the
  reference lifecycle. ProtT5 is a separate pinned download.
- **Chemistry:** optional and lazily imported; plain prediction must stay fast.
  At pH 7 without whole-structure relaxation, local rescue is the default,
  `--rescue restrained` is explicit, and `--rescue off` retains explicit rejection.
  Other pH values require rescue off and use the earlier CPU protonation path.
  The new pH-7 geometry path requires CUDA even with rescue off.
- **Rescue versus relaxation:** rejection-only restrained rescue has convergence,
  chirality and paired geometry gates. Existing whole-structure
  `--relax restrained` minimises every input and does not inherit all those guards.
  Its portable platform default is CPU; CUDA is explicit on Spark.
- **Outputs/failures:** raw `.pdb`, optional `.prepared.pdb`/`.relaxed.pdb`, and
  `.preparation.json` are distinct. Preserve partial failure audits and nonzero
  exit status. Do not hide parameterisation, Reduce or CUDA errors as successful
  alternative preparation. Use explicit `--seed 42` for the qualification protocol;
  preparation otherwise defaults to seed 0.

The main unresolved output-correctness issue is **confidence export**:
`scripts/predict.py::extract_plddt_scores` flattens 50-bin logits instead of
computing per-residue expected pLDDT. JSON means/thresholding are consequently
not calibrated pLDDT, and PDB B-factors are zero. Structural evaluation used the
proper conversion; that does not validate the production exports. Keep this a
separate, explicitly reviewed output change.

## What not to repeat without new evidence

The A–J optimisation list was worked through; it is not a queue of untouched
ideas. Larger batch banks, fixed-length specialisations, extra workers, eager
CUDA graphs and transfer changes did not establish a useful full-job improvement
for the accepted default. Deferred transfers preserved output but gained only
about 1% on the tested large job. A standalone geometry-update rewrite was
exact but much slower. Revisit these only for a changed workload or a new profile.

CPU “conversion time” in a trace largely waited for GPU completion. It was not
all CPU arithmetic waiting to be rewritten. Likewise, higher CPU/GPU utilisation
is not itself evidence of greater completed-job throughput.

A resident-process prototype did remove repeated startup for separate small
requests, with exact tested output. It remains a prototype, not a production
service. Combining a known workload into one CSV already avoids repeated startup.
Do not build a network service unless that is the actual requested use case.

## Suggested next work, in order

These are recommendations for the next conversation, not jobs already approved
or started.

1. **Close the latest-dev rebuild gap.** Earlier prediction deployment had a
   no-cache build qualification. Rescue integration reused its verified runtime
   and AOT packages; the latest complete dev recipe has not had a new cold
   download/export build. Clone dev into a fresh directory, freeze its commit,
   choose a new image tag and empty AOT build directory, and retain both working
   images. Check pinned dependencies, guarded patches, all package identities,
   offline installed prediction and the retained six-case rescue panel. Reuse
   applicable numerical evidence; do not repeat every old benchmark. Fresh AOT
   exports may change rounded coordinates within qualified limits.
2. **Review and correct confidence exports.** Define the logits-to-score
   conversion, residue ordering and output/threshold semantics first. Test against
   the scoring implementation and ensure inference coordinates and throughput
   remain unchanged. Discuss the result-changing annotations before enabling.
3. **Qualify chemistry beyond its development panel.** Freeze a genuinely new,
   diverse panel spanning all three checkpoints, long loops/length tails and
   topology/protonation edge cases. Compare local versus restrained rescue using
   failure/coverage rates, per-target displacement and stereochemistry/contact
   tails. Experimental structure error is a separate endpoint from geometry.
   Treat the current 512 as development data. Check whether the older
   whole-structure relaxation needs the newer chirality safeguards.
4. **Expand preparation coverage selectively.** Test non-pH-7/CPU operation only
   if needed; neither is qualified by the new pH-7 CUDA results. Investigate
   context/startup failures and worker robustness before adding more concurrency.
   Keep normal inference and GPU preparation scheduling in the same end-to-end
   benchmark. Consider persistent compatible contexts only with complete state
   reset and geometry/output checks.
5. **Choose the next speed target from the intended workload.** For many small
   requests, qualify the resident-worker lifecycle, cancellation and isolation.
   For language throughput, batching/compiling ProtT5 is a separate model-specific
   project with exact embedding checks. For much larger repertoires, test
   metadata growth, length tails and filesystem/output organisation before
   assuming the 100k throughput extrapolates to millions.

For a later authorised rebuild, the existing script supports a separate tag:

```bash
# In a fresh dev clone; not run as part of this handover.
NO_CACHE=1 IMAGE_TAG=cold-qualification \
  AOT_BUILD_DIR="$PWD/.aot-build/cold-qualification" ./docker_build.sh build
IMAGE_TAG=cold-qualification ./docker_build.sh test
```

The import test is only a smoke test. The Dockerfile installs the pinned public
runtime/checkpoints, then the script prepares AOT packages in a GPU container and
assembles the final image. Normal Docker build itself has no GPU. The recorded
foundation is NGC PyTorch 25.11, CUDA 13.0, OpenMM 8.6.1 and PDBFixer 1.12.0;
last qualified driver was 580.178.04. Check the actual driver/runtime identities
rather than substituting newer CUDA wheels or assuming AOT portability. CUDA
13.4 NVRTC previously emitted PTX that this driver could not load; retain the
CUDA-13.0 dependency pins unless deliberately qualifying an upgrade.

The Dockerfile pins the model foundation at `SOURCE_REF` and copies an explicit
set of current CLI/model-adapter files into it. It does not install every local
checkout change automatically. Include new runtime modules in that copy list,
apply guarded preparation patches and regenerate the checked AOT packages when
their source/runtime identity changes.

## Where to resume and what survives locally

Canonical server paths:

- Main: `/home/adrian/codex_wd/projects/abodybuilder3/repos/ABodyBuilder3-DGX-Spark`.
- Dev: `/home/adrian/codex_wd/projects/abodybuilder3/repos/abodybuilder3-dev`.
- Inputs, truth, frozen source, records and retained references:
  `/home/adrian/codex_wd/projects/abodybuilder3/runs`.
- Original rebuild audit:
  `/home/adrian/codex_wd/reports/abodybuilder3_251026`.
- Storage inventories, exact deletion journals and receipts:
  `/home/adrian/codex_wd/reports/abodybuilder3_storage_271026`.

Only main/dev checkouts and their two working images remain. Component branches
and historical recipes remain on GitHub; recreate worktrees/clones when needed.
The local source copies were clean and backed by verified GitHub refs before
removal. Main/dev's shared Git store and branch refs were retained.

User-authorised cleanup removed historical PDB/raw-array outputs, stopped
containers, obsolete images and their unused cache, experimental compiler
artifacts, the derived selection database, redundant checkouts and small cache
metadata/empty directories. **Old report paths may therefore identify deleted
payloads.** Inputs, selection/provenance manifests, simulation/experimental truth,
benchmark records, frozen source and compact Git evidence remain. Old predictions
or abandoned images require regeneration; exact original timing cannot be
recreated, and numerical replay is not guaranteed across rebuilds.

The four complete current reference studies remain protected:
`hydrogen_geometry_271026`, `heavy_atom_rescue_271026`,
`rescue_pipeline512_271026` and `dev_rescue_integration_271026`.
Together they contain about 17.23 GB of failure examples, controls, predictions,
prepared outputs and audits. The reusable six-case panel is in
`runs/heavy_atom_rescue_271026`; `tools/check_rescue_integration.py` compares it
with the installed runtime when the tool and panel are mounted explicitly.
The latest integration records are in `runs/dev_rescue_integration_271026`.

ProtT5 remains at `runs/rebuild_251026/prott5` (about 11.28 GB). The original
upstream data archive remains at
`runs/structural_checks_251026/data/upstream_data.tar.gz` (about 3.02 GB).
The large source OAS corpus is on the external `/mnt/sequences` SSD; local frozen
selections remain. None of that external data was cleaned up.

After housekeeping, dedicated ABodyBuilder3 storage is approximately 38.5 GB,
plus about 20.3 GB of Docker layers shared with other projects. These are dated
storage observations, not quotas. The last cleanup removed about 0.60 GB of
approved allocation even though global free space rose by 3.83 GB during its
window; the larger net change must not all be credited to this project.
Further deletion needs a new discussion. Keep sequences, raw structures and
private per-target manifests local; this handover does not authorise uploading
payloads or publishing audit inventories.

## Evidence and code map

Use these entry points for details instead of treating this summary as a test log:

| Question | Entry point |
|---|---|
| Main/dev interface and constraints | [Dev README](../README.md), `scripts/predict.py`, `scripts/batch_pipeline.py`, `src/abodybuilder3/aot_runtime.py` |
| Numerical gates, independent implementation panel and failure tests | [Installed AOT validation](abodybuilder3_dev_aot_validation_261026.md), `tools/score_deployment.py`, `tests/test_aot_runtime.py`, `tests/check_deployment_failures.py` |
| Real repertoire scaling and RAM accounting | [Repertoire study](abodybuilder3_repertoire_scale_261026.md), [unified RAM protocol](abodybuilder3_unified_memory_261026.md) |
| ProtT5 lifecycle and exactness | [Language report](abodybuilder3_language_lifecycle_261026.md), `src/abodybuilder3/language/direct_inference.py` |
| Current chemistry, dispatch and failure contract | [Preparation guide](abodybuilder3_preparation_251026.md), [rescue integration](abodybuilder3_rescue_integration_271026.md), `scripts/structure_preparation.py`, `scripts/preparation_pool.py`, `scripts/rescue_policy.py` |
| Geometry and both rescue implementations | `scripts/hydrogen_geometry.py`, `scripts/heavy_atom_rescue.py`, `scripts/rejected_rescue.py`, `scripts/rescue_geometry.py`; qualified [component code and reports](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/tree/da2672e33dd1afdc4f1b3fadf5d997ef9f8fdd99) |
| Completed/rejected performance ideas | [Ordered results](abodybuilder3_ordered_optimisation_261026.md), [historical development plan](abodybuilder3_aot_development_plan_261026.md), component branches `components/aot-pipeline-261026`, `components/openfold-structure-251026`, `components/pdb-formatting-261026`, `components/prott5-runtime-261026` |
| Third-party reference/candidate patches | `patches/preparation/`, the qualified component branch above and `components/structure-preparation-261026`; originals and regression evidence stay with their component |

## Working conventions that matter next time

The user prioritises **completed-job speed on Spark**, with reasonable memory,
not memory reduction for its own sake or maximum utilisation on every core.
Benchmark the full requested pipeline, including startup and output writing.
Keep prediction-only, hydrogen preparation and full relaxation rates separate.

Spark has unified memory: primary physical-headroom estimates use a calibrated
idle-adjusted drop in host `MemAvailable`. Do not add RSS, PSS, CUDA allocator and
cgroup numbers. Report baseline drift, individual technical repeats and metric
scope; use paired local-structure/tail metrics alongside whole-Fv RMSD. Two runs
are not biological replicates, and geometry is not experimental accuracy.

The user is a data scientist/statistician: use informative plots plus concise
tables, readable labels and underlying data, rather than large static tables or
HTML-only reports. Stamp chat messages in Europe/London time and date report
filenames `_ddmmyy`. Apply the shared `/home/adrian/codex_wd/guidance` documents and
set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before CUDA initialisation.

Keep future third-party changes on dedicated component branches with original
and candidate code and meaningful regressions. Discuss changes that alter
results before enabling them. Commit/push accepted work; keep main's deployment
documentation concise and detailed research on dev. Do not rerun GPU studies for
this documentation change or remove retained resources without new authority.
