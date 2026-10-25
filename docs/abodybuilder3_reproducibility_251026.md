# Rebuilding and verifying ABodyBuilder3 on DGX Spark

Verified on 25 October 2026. The original remains on main. This branch now includes explicit chemical preparation and restrained relaxation; see [the preparation guide](abodybuilder3_preparation_251026.md) and [updated benchmarks](abodybuilder3_preparation_benchmarks_251026.md). No existing image, container, checkpoint or output was removed.

## What is pinned

- Base: NVIDIA NGC PyTorch 25.11, manifest digest `417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21`.
- Original image source: deployment commit `8e4b5af7c347dadcc52678c7ee61ea975eeef1a8`. All 88 captured files match the retained image. The repository main CLI at `a323ab3abe48440305e9b4baf1b5a393781d7d78` was frozen and used across all measured runtimes; it differs slightly from the older CLI baked into the original image.
- Original installed Python package versions: `runtime-constraints.txt` on main; all 322 installed distribution versions matched after the original clean rebuild. This branch contains the deliberate modernised dependency pins instead.
- Original OpenMM source `07b738c560f31f539674528c3a8ca629fc8320ba`, PDBFixer `94cfa4c0ca551cdc5f13320f9a658efd59f2b881`, Reduce `9d4ac1f5f103daa18ea471f11fe1cf9b2cd8ca1f`. The rebuilt Reduce binary and dictionary also matched their original SHA-256 hashes.
- Three ABodyBuilder3 checkpoints from [Zenodo 11354577](https://zenodo.org/records/11354577); archive MD5 and each checkpoint SHA-256 are checked during the build. There is no dependence on a local copy of the old image or weights.
- Language embedding model: [Rostlab/prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50), revision `973be27c52ee6474de9c945952a8008aeb2a1a73`. This is the original full ProtT5 model, not the half-precision encoder alternative. Its five inference assets occupy 11.28 GB and are downloaded separately.

The latest official ABodyBuilder3 source checked was [18e4058](https://github.com/Exscientia/abodybuilder3/commit/18e4058015a39c5405c08a0d5629cf302627b253), dated 23 October 2024. It is already an ancestor of the deployment. The latest official weights record remains 11354577. Modernisation therefore does not imply new trained weights.

## Rebuild the original from public sources

```bash
git clone --branch main https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark.git
cd ABodyBuilder3-DGX-Spark
NO_CACHE=1 IMAGE_TAG=rebuild-local ./docker_build.sh build
IMAGE_TAG=rebuild-local ./docker_build.sh test
```

This uses a separate tag. No old local build cache, checkpoint directory or project checkout is required; Docker still obtains the pinned public base image. Apt packages come from Ubuntu's current repository, so image bytes are not promised to be identical across dates. The checked source, model parameters and pinned Python runtime are the reproducibility targets.

To capture and check the runtime without a GPU:

```bash
mkdir -p verification
docker run --rm --network none   -e CAPTURE_UID="$(id -u)" -e CAPTURE_GID="$(id -g)"   -v "$PWD/tools:/tools:ro" -v "$PWD/verification:/capture"   abodybuilder3-spark:rebuild-local python /tools/capture_runtime.py
python3 tools/verify_runtime.py   --reference tests/reference_runtime_251026.json   --candidate verification/original_runtime_manifest_251026.json
```

The identity check includes source, checkpoints, installed Python versions, OpenMM source revision and Reduce artefacts. After a compiler or Ubuntu update, a binary checksum may change; investigate rather than assuming identical predictions or deleting the old image.

## Build this modernised branch

```bash
git clone --branch official/abodybuilder3-spark https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark.git
cd ABodyBuilder3-DGX-Spark
NO_CACHE=1 IMAGE_TAG=modernised-local ./docker_build.sh build
IMAGE_TAG=modernised-local ./docker_build.sh test
```

This branch defaults to the separate image name `abodybuilder3-official-spark`; it does not replace the original tag. Its OpenMM 8.6.1 CUDA 13 wheel dependencies and Reduce revision are pinned. The current recipe uses CUDA 13.0 wheel dependencies compatible with tested driver 580.178.04, and copies this checkout's prediction/preparation scripts over the pinned model checkout. The preparation study verified actual CUDA minimisation; the earlier CUDA-13.4-wheel image could not load its PTX on this driver. Capture the current image as above and use `tools/verify_runtime.py --reference tests/reference_preparation_runtime_251026.json --candidate verification/original_runtime_manifest_251026.json` for its strict identity check. `tests/reference_modernised_runtime_251026.json` retains the earlier deployment identity. Separately, `--modernised` with the original reference checks shared model source and checkpoints while allowing the deliberate runtime upgrades.

## Prepare the public inputs and run

```bash
python3 tools/prepare_benchmark_inputs.py --output benchmark_inputs
docker run --rm --gpus all --ipc=host   -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   -v "$PWD/benchmark_inputs:/inputs:ro" -v "$PWD/predictions:/predictions"   abodybuilder3-official-spark:modernised-local   python /opt/abodybuilder3/scripts/predict.py predict --csv /inputs/antibodies_5.csv     --output /predictions/ --model plddt --workers 1 --seed 42
```

`--model base`, `plddt` and `language` select distinct trained checkpoints. Compare each against its own original counterpart. The default remains `plddt`. For language inference, the CLI normally fetches ProtT5 on first use; offline measurement instead mounts the pinned local snapshot prepared below.

## Reproduce the earlier staged measurements

The earlier source/runtime comparison used CLI `a323ab3` and the published harness at commit `3e64fab`. Check out that commit in a separate checkout when reproducing those historical tables; the current branch's CLI and CUDA dependency pins are intentionally different. For the current comparison, use `tools/run_preparation_study.py` and its frozen source/image manifest.

Prepare ProtT5 with a new output directory:

```bash
mkdir -p prott5
docker run --rm -v "$PWD/tools:/tools:ro" -v "$PWD/prott5:/prott5"   abodybuilder3-spark:rebuild-local python /tools/download_prott5.py --output /prott5
```

Download time is preparation, not inference. Loading and fresh embedding generation are included in the language measurements.

The published measurement controller defaults to the image tags retained in this study: `abodybuilder3-spark:latest`, `abodybuilder3-spark:rebuild-251026`, and `abodybuilder3-official-spark:modernised-251026`. Build those separate tags when reproducing the full comparison. For modernised, the final fully pinned recipe is runtime-identical to that benchmarked image; the local final tag is `abodybuilder3-official-spark:modernised-pinned-251026`. Neither replaced `latest`.

```bash
git clone --branch dev https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark.git ../abodybuilder3-dev
python3 tools/run_comparison.py --models base plddt --count 5   --frozen benchmark_inputs --prott5 prott5 --dev-source ../abodybuilder3-dev --output results_small
python3 tools/run_comparison.py --conditions dev --models base plddt   --mode dev_batch --batch-size 3 --count 5   --frozen benchmark_inputs --prott5 prott5 --dev-source ../abodybuilder3-dev --output results_small
```

The dev comparison was pinned to `01047c73f7505d46db808f42aa32f1c624a3e21e`; check out that commit for reproduction. `run_comparison.py` retains its containers and refuses to overwrite an existing case directory. Compare small outputs with `tools/compare_predictions.py` before increasing `--count` to 100 or 1,000. The larger study screened dev batch sizes 16, 32 and 64 at 100 inputs, then tested the fastest passing size, 32, at 1,000 inputs. Language used 5 then 20 inputs to bound the expensive embedding preparation, with `--models language`. The existing dev helper does not support language embeddings; `--mode dev_language_adapter` is a benchmark-only extension that reuses unchanged ProtT5 embeddings.

For the unchanged eight-worker CLI, use `tools/benchmark_cli.py --condition original --model plddt --count 1000 --workers 8 --frozen benchmark_inputs --output results_cli`. Add `--condition rebuilt` or `modernised` in fresh output directories for their matched measurements. These runs instrument every worker and check that all expected PDB files exist.

## Interpretation and limitations

The five public example pairs contain 227–231 residues in total. Input sizes are counted from the actual strings, rather than historical comments. Repeating them measures throughput, not accuracy on 1,000 distinct antibodies. Prediction comparisons use raw atom coordinates, masks and confidence arrays, with Cα aligned RMSD and local-distance agreement as additional summaries. A low aligned RMSD cannot hide raw-coordinate differences.

The rebuilt and modernised unrelaxed predictions match the original exactly for these inputs. Dev batching produces small numerical differences; its existing criterion is maximum absolute Cα coordinate difference below 0.01 Å. This is an implementation-agreement check, not experimental-structure validation or a universal tolerance guarantee.

Language runtimes vary substantially between repetitions. No reliable language speed ranking is claimed. The language adapter was slower in this study. The original remains the working default, and the existing dev CLI is unchanged; its parsed `--batch-size` currently does not invoke the batching helper.

The frozen historical CLI's `--relaxation` flag performed hydrogen addition, not energy minimisation. The current branch rejects it with an error; use `--add-hydrogens` or genuine `--relax restrained` instead. Optional hydrogen-added files need not be byte-identical after updating Reduce: for the checked example, legacy Reduce's nonzero exit caused a PDBFixer fallback; Reduce 4.17 succeeds and the CLI uses its output. See the benchmark report for atom counts. The underlying unrelaxed model predictions were unchanged.

CPU checks: `python3 -m unittest discover -s tests -p 'test_deployment_comparison.py' -v`. Full measurements, limits and the static figure are in [the benchmark report](abodybuilder3_benchmarks_251026.md).

Confidence reporting is also an existing CLI limitation: the model head emits 50-bin logits; true pLDDT is their softmax-weighted expectation over bin centres. The benchmark comparison derives that score separately. The current CLI JSON/filter path flattens logits, and PDB B-factors are zero. These annotations were not changed in this deployment comparison.
