# Isolated ABodyBuilder3 PDB writer optimisation

This branch compares the original Python PDB formatter with a C++ implementation
of the same text output. It contains the original, candidate, Docker extension,
regression tests and measurements. It does not replace the model or change its
weights, precision or inference equations.

On the DGX Spark, the native writer gave **1.19–1.22× complete-job throughput**
for 10,000 inputs. All **88,820 matched raw predictions and PDB files were
identical** in the dedicated study. Main and dev remain unchanged; this candidate
is available for review and is not enabled there.

See the [full speed, memory and exactness report](docs/abodybuilder3_pdb_formatting_261026.md)
and [measurement archive](docs/data/abodybuilder3_pdb_formatting_evidence_261026.zip).

## Contents

| Path | Purpose |
| --- | --- |
| `reference/format_pdb.py` | Original `to_pdb_fast`, copied verbatim from dev `bd8a12e` |
| `candidate/format_pdb.py` | Input checks, native call and original-writer fallback |
| `candidate/pdb_native.cpp` | PDB formatting in C++, releasing the Python GIL |
| `native_activation.py` | Explicit opt-in hook for the experimental Docker image |
| `tests/check_formatter.py` | Byte comparisons, boundary cases and concurrent writers |
| `tools/run_formatter_study.py` | Matched full-CLI speed/memory/prediction comparison |
| `docs/provenance_261026.json` | Original source and function hashes |

The original function's historical speed comment is retained with its source;
our measured formatter-only ratio is **5.40×**, distinct from full-job throughput.
The native path uses a thread-local C numeric locale and preserves PDB spacing,
rounding, atom order, chain separators and line endings. Inputs outside its
array contract fall back to the original formatter. Exactness is a tested result,
not a claim that arbitrary malformed Protein objects are supported.

## Build the experimental image

The parent runtime and released checkpoints are supplied by the pinned dev
recipe. From separate directories:

```bash
git clone --branch dev https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark.git abodybuilder3-runtime
git -C abodybuilder3-runtime checkout bd8a12ea9f58fa3577215532653a05c47e90e256
docker build -t abodybuilder3-dev-spark:optimised-251026 abodybuilder3-runtime

git clone --branch components/pdb-formatting-261026 https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark.git abodybuilder3-pdb
docker build -t abodybuilder3-component-spark:native-formatter-261026 abodybuilder3-pdb
```

The second build compiles the extension against the parent image's Python ABI
and existing pybind11. It does not install a different PyTorch or model. A fresh
parent build requires network access to the pinned upstream sources and public
checkpoint archive. The measured image identities are in the report; a later
build is not asserted to produce the same Docker image digest.

The native writer is **off by default** in this image. To test it with your CSV:

```bash
mkdir -p results
docker run --rm --gpus all --ipc=host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e ABB3_NATIVE_FORMATTER=1 \
  -e ABB3_FORMATTER_USAGE=/results/formatter_usage.json \
  -v "$PWD/input.csv:/data/input.csv:ro" \
  -v "$PWD/results:/results" \
  --entrypoint python \
  abodybuilder3-component-spark:native-formatter-261026 \
  /opt/abodybuilder3/scripts/predict.py predict \
  --csv /data/input.csv --output /results/pdbs/ --model base \
  --batch-size auto --cpu-workers 2 --workers 1 --seed 42 --quiet
```

Set `ABB3_NATIVE_FORMATTER=0` for the matched original-writer control. Both use
the same model and runtime. `ABB3_STRICT_FP32=1` explicitly disables TF32 matrix
multiplication; the normal parent setting permits TF32. These settings do not
enable the separate attention/compiler experiments.

## Run the regression checks

The generated edge cases and eight-thread test need no GPU or downloaded test
predictions. Mount the checkout at `/formatter` so the local tests are visible:

```bash
docker run --rm --network none \
  -v "$PWD/abodybuilder3-pdb:/formatter" \
  -v "$PWD/results:/results" \
  --entrypoint sh abodybuilder3-component-spark:native-formatter-261026 \
  -c 'sh /formatter/tools/build_native.sh && python /formatter/tests/check_formatter.py --output /results/formatter_regression.json'
```

For the extended 413-fixture screen, add `--predictions /predictions` and mount a
retained pipeline study containing its first 100 raw predictions per checkpoint
at `n1000/{base,plddt}/eager_normal/run1/raw/`. These local raw predictions are not
bundled in this code branch. The report records the completed extended screen
and full-CLI comparisons; they are not inferred from the small generated cases.

The licence is [Apache 2.0](LICENSE.txt), retaining the source project's terms.
The component is intended for review before any deployment to dev.
