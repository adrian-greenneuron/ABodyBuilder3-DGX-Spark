# Isolated OpenFold-derived structure-module experiments

This branch contains only the component under investigation, its untouched
reference, candidate code and comparison/regression tests. It is not a new
trained model or a replacement for ABodyBuilder3's default release.

`reference/structure_module.py` is copied byte for byte from ABodyBuilder3 dev
commit `07bfe1e`, whose core model is pinned to
`a323ab3abe48440305e9b4baf1b5a393781d7d78`. The file retains the original Exscientia,
AlQuraishi Laboratory and DeepMind copyright and Apache 2.0 licence notices.
`accepted/structure_module.py` is the exact-output subset deployed to dev, with
`accepted/features.py` as its compact-feature adapter. `candidate/` also retains
the nonidentical experiments for review. Both retain the original licence
notices and default to reference behaviour. Explicit inference-only arguments
enable one experiment at a time:

- `_inference_pair_projected`: consume the projected pair representation made
  by `candidate/features.py`, using the original learned projection weights.
- `_inference_point_matmul`: replace the weighted point broadcast/reduction
  with three batched matrix multiplications.
- `_inference_point_ieee`: run those replacement products with TF32 disabled
  locally, restoring the previous setting afterwards. This diagnostic assumes
  one inference thread; it is not installed in dev.
- `_inference_final_only`: retain all learned backbone/attention updates but
  omit intermediate side-chain outputs which do not feed the next update.
- `_inference_pair_buffer`: copy the static pair channels once and update only
  the distance channel at each block, preserving the attention inputs exactly.

The unchanged dependency modules and released checkpoints come from the pinned
ABodyBuilder3 runtime. The retained reference image is
`abodybuilder3-dev-spark:batched-251026`, image
`sha256:0a3ffb4edcef8d608d65eca49787fcb4ea820098a63e6dfdca853baee0527afa`.
Its public build recipe is in the main repository's dev branch at `07bfe1e`.
No checkpoint is copied to this branch.

## Comparison protocol — 25 October 2026

Use unchanged FP32 settings, checkpoint weights, allocator, padding/batch groups
and input order. Run changes separately, first on the five public sequence pairs
and artificial length/substitution stress cases. These synthetic cases evaluate
implementation agreement; they do not establish biological accuracy.

The compatibility control must match the reference exactly. A candidate is
eligible for dev only when its final raw coordinates, confidence logits, final
states and PDB bytes match exactly and a matched complete-job comparison shows
a speed improvement. Numerical differences require discussion with the user;
the earlier batching tolerance is not permission to accept new differences.

Component measurements use one unmeasured/warm-up repetition and two timed
repetitions, reversing control/candidate order. Final decisions use fresh
complete-job containers and current overlapping-pipeline code. Report CUDA
allocated/reserved peaks, sampled process RSS and container accounting
separately; none is a total physical-memory bound on the Spark.

Inside that runtime, mount this branch at `/component` and the dev repository's
`tools/` at `/measurement`. Run:

```bash
PYTHONPATH=/measurement python /component/tools/compare_component.py \
  --models base plddt --count 64 --batch-size 32 --output /results/comparison
python -m unittest discover -s /component/tests -p 'test_*.py'

# Compare only the accepted subset, with the same exact-output gate.
PYTHONPATH=/measurement python /component/tools/compare_component.py \
  --implementation accepted --variants exact_all --models base plddt \
  --count 64 --batch-size 32 --output /results/accepted
```

All comparison records include source hashes. `requires_user_discussion` means
the change is experimental and must not be treated as accepted or enabled.

## Outcome

Projection lookups, final-only side-chain outputs and pair-buffer reuse passed
the exact-output gate. The deployed combination also passed 960 GPU regression
comparisons and 18,610 matched complete-job prediction comparisons, including
warm-ups. Complete 1,000-input jobs used about 14% less time, 57% less process RSS
and 19% less peak CUDA allocated memory than the preceding dev version.

The initial point-matmul experiment was faster in its small isolated screen but changed Cα
coordinates by up to 0.0443 Å. The compiler experiment was faster in its warmed
component screen but changed Cα coordinates by up to 0.0093 Å. Neither is
deployed or accepted; both require discussion before further adoption. The
compiler screen excludes compilation latency and is not a complete-job result.

The follow-up [structural accuracy and clustering report](docs/abodybuilder3_attention_compiler_accuracy_251026.md)
uses all 100 published experimental test antibodies, both base/pLDDT checkpoints,
strict-precision controls and all 12,000 within-antibody prediction pairs. Accuracy
changes vary by antibody; no candidate is consistently better. CDR-backbone
changes were tiny, while some side-chain atoms moved by up to about 0.5 Å.
Neither attention replacement nor compilation has been enabled on dev. Main and
dev remain unchanged; this component branch contains the experimental evidence.

The runtime uses FP32 tensors/weights with unchanged NGC defaults, including
`float32_matmul_precision='high'` and `allow_tf32=True`. Thus FP32 here must not
be read as a claim that every matrix multiplication uses strict IEEE FP32.

The follow-up [attention-plus-compilation comparison](docs/abodybuilder3_combined_compilation_251026.md)
tests both combinations against compilation alone at matched precision on the
same 100 experimental antibodies. Normal precision reduced warmed model-stage
time by 3.7–5.1%; strict FP32 increased it by 1.2–1.6%. Peak CUDA allocation
changed by about 4.8 MiB. CDR-H3 backbone accuracy changed by at most 0.0051 Å,
but normal-precision side-chain atoms moved by up to 0.775 Å. These are model-stage
measurements, excluding compilation and CLI overhead; neither combination is
enabled on dev. The report includes all eight conditions, local accuracy,
pairwise comparisons and resource definitions.

The [complete-pipeline follow-up](docs/abodybuilder3_pipeline_compilation_261026.md)
includes startup and writing every PDB in 48 fresh-container jobs. Empty compiler
caches made both compiled approaches substantially slower for 100- and 1,000-input
jobs. Peak CUDA allocation fell but CPU memory increased; fresh compiled
processes did not consistently give bit-identical predictions, even with strict
FP32. Neither option is enabled on dev. Reused caches and persistent workers
remain untested in this comparison.

[Component measurements](docs/component_comparisons_251026.json) and
[source provenance](docs/provenance_251026.json) retain the accepted and held
results separately. The production pipeline and complete-job measurements are
in the main repository's [dev report](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/blob/dev/docs/abodybuilder3_dev_optimisation_251026.md). Main was unchanged.
