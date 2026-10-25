# ABodyBuilder3 on NVIDIA DGX Spark — batched development branch

Predict antibody variable-region structures from paired heavy- and light-chain sequences on the DGX Spark's ARM64 CPU and NVIDIA GB10 GPU. This repository packages [ABodyBuilder3](https://github.com/Exscientia/abodybuilder3) in a reproducible Docker environment, with a parallel prediction CLI and optional chemical preparation and restrained energy minimisation.

**This is `dev`: the modernised main runtime with integrated GPU batching.** The model architecture, released weights and FP32 settings are unchanged. Bounded CPU preparation and output queues overlap prediction in one GPU-owning process.

The latest dev implementation completed 1,000 base/pLDDT inputs in **28–29 seconds**, using about **14% less time**, **57% less sampled process RSS** and **19% less peak CUDA allocated memory** than the preceding dev version. All matched raw predictions and PDB bytes were identical. Similar gains held for 1,000 distinct synthetic sequence variants. See the [current measurements and validation](docs/abodybuilder3_dev_optimisation_251026.md).

[Main](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/tree/main) remains the stable worker-based deployment. The original rebuild recipe is preserved at [`original-runtime-251026`](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/tree/original-runtime-251026). This dev branch includes main’s explicit chemical preparation and restrained minimisation.

## Requirements

- NVIDIA DGX Spark running Linux, with Docker and the NVIDIA Container Toolkit configured for GPU containers.
- A working NVIDIA driver. This deployment was tested with **580.178.04**; its CUDA dependencies are pinned to 13.0 for that driver.
- Internet access during the build to obtain the public base image, source and checkpoints. No existing local ABodyBuilder3 image or weights are needed.
- Space for an approximately **22 GB logical Docker image**, plus build cache and outputs. Docker layers may be shared with other images. The language model needs a separate **11.28 GB** ProtT5 download.

Memory use depends on the model, worker count and sequence lengths. The measurements below cover five public antibody pairs of 227–231 residues in total; they do not establish the largest supported input.

## Quick start

### 1. Build and check the environment

```bash
git clone --branch dev https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark.git
cd ABodyBuilder3-DGX-Spark
./docker_build.sh build
./docker_build.sh test
```

The default image name is `abodybuilder3-dev-spark:latest`. The build verifies the downloaded checkpoint hashes; the test checks imports and reports GPU availability. To rebuild without using Docker's build cache:

```bash
NO_CACHE=1 ./docker_build.sh build
```

### 2. Predict the five public examples

The input preparation command uses only Python's standard library. Run it once in a fresh output directory:

```bash
python3 tools/prepare_benchmark_inputs.py --output benchmark_inputs --counts 5
mkdir -p predictions

docker run --rm --gpus all --ipc=host \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$PWD/benchmark_inputs:/inputs:ro" \
    -v "$PWD/predictions:/predictions" \
    abodybuilder3-dev-spark:latest \
    python3 scripts/predict.py predict \
        --csv /inputs/antibodies_5.csv --output /predictions/ \
        --model plddt --batch-size auto --seed 42
```

Raw heavy-atom PDB files are saved in `predictions/` on the host. The image runs as root by default, so files written through the mount may be owned by root.

### 3. Use your own sequences

Provide a CSV with `name,heavy,light` columns, one paired antibody per row. The generated example CSV shows the expected format. Mount its directory as `/inputs`, then change the `--csv` filename above. Keep the trailing `/` in `--output /predictions/` to request a directory.

The CLI also accepts `--heavy`/`--light` sequences and paired FASTA files. To see all options:

```bash
docker run --rm abodybuilder3-dev-spark:latest \
    python3 scripts/predict.py predict --help
```

## Models and parallel prediction

| Selection | Model | Additional assets |
|---|---|---|
| `--model plddt` | Default checkpoint, trained with the pLDDT confidence head | Included in the image |
| `--model base` | Base checkpoint | Included in the image |
| `--model language` | Checkpoint using ProtT5 sequence embeddings | Separate ProtT5 download |

These are different trained checkpoints. Deployment checks compare each with its own original counterpart; they do not rank their accuracy against experimental structures.

Base and pLDDT use GPU batching by default. Antibodies are grouped by combined chain length into eight-residue bins, reducing padding, then processed by a single model. Results are transferred to CPU before formatting and writing. Input names remain associated with their predictions even though execution is reordered.

The inference path uses compact projection lookups, computes side-chain outputs only at the final model block, and reuses the static pair-feature buffer. All eight learned attention/backbone updates remain. The isolated [OpenFold-derived component branch](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/tree/components/openfold-structure-251026) retains the original code, accepted implementation and regression tests. Faster experiments that changed predictions remain disabled.

| Option | Meaning |
|---|---|
| `--batch-size auto` | Default: cap at 16 for at most 128 inputs, otherwise 32; the memory estimate can reduce batches further |
| `--batch-size 16` | Explicit cap with lower measured CUDA allocation than 32 on the public workload |
| `--batch-size 1` | Disable GPU batching and use the worker-based path |
| `--cpu-workers 2` | Default: two workers per preparation/output stage, with bounded queues |
| `--batch-memory-gib 6` | Estimated inference-allocation target; not a hard total-memory limit |
| `--batch-report /predictions/batching.json` | Optional record of actual batches, CUDA OOM splits and completion |

Batching accepts `--workers auto` or `--workers 1`, using one GPU model process. To compare the worker approach, explicitly select `--batch-size 1 --workers 8`. Invalid worker counts are rejected; the old DataLoader worker-selection helper is not used by the current CLI.

The language checkpoint keeps its established single-model path: `auto` selects batch size 1 and one worker. Explicit language batching is rejected until a qualified implementation demonstrates an improvement. Its separate ProtT5 download is described below.

**Confidence export limitation:** the current CLI's JSON and `--confidence-threshold` paths use flattened confidence logits rather than properly converted pLDDT scores; PDB B-factors are zero. Do not interpret those exported values as calibrated pLDDT or use the threshold for scientific filtering. This existing limitation is documented in the [reproducibility notes](docs/abodybuilder3_reproducibility_251026.md#interpretation-and-limitations); the model's confidence head itself is unchanged.

### Prepare a reusable ProtT5 download

The CLI can download ProtT5 on first use, but a host-mounted snapshot avoids downloading it again in disposable containers. The helper pins the revision and downloads only the five inference assets:

```bash
mkdir -p prott5
docker run --rm \
    -v "$PWD/tools:/tools:ro" -v "$PWD/prott5:/prott5" \
    abodybuilder3-dev-spark:latest \
    python3 /tools/download_prott5.py --output /prott5

mkdir -p language_predictions
docker run --rm --gpus all --ipc=host \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$PWD/benchmark_inputs:/inputs:ro" \
    -v "$PWD/prott5:/opt/abodybuilder3/Rostlab/prot_t5_xl_uniref50:ro" \
    -v "$PWD/language_predictions:/predictions" \
    abodybuilder3-dev-spark:latest \
    python3 scripts/predict.py predict \
        --csv /inputs/antibodies_5.csv --output /predictions/ \
        --model language --workers 1 --seed 42
```

Embedding generation is included in the language timings below. Download time is excluded.

## Optional chemical preparation and relaxation

Plain prediction is the fast default. Preparation runs only when requested, saves separate outputs and preserves the raw PDB. Add the desired options to the prediction command inside the container:

| Options | Behaviour |
|---|---|
| `--add-hydrogens --ph 7` | Repair missing heavy/terminal atoms and add hydrogens at pH 7 |
| `--add-hydrogens --hydrogen-method reduce --sidechain-flips` | Also allow Reduce to optimise Asn/Gln/His side-chain orientations |
| `--relax restrained --relax-platform cuda` | Prepare, then perform restrained energy minimisation on the Spark GPU |

For example, keep the quick-start Docker command and change its final line to:

```bash
        --model plddt --workers 1 --seed 42 --relax restrained --relax-platform cuda
```

| Output | Contents |
|---|---|
| `.pdb` | Unmodified raw prediction |
| `.prepared.pdb` | Repaired structure with hydrogens |
| `.relaxed.pdb` | Restrained energy-minimised structure, when requested |
| `.preparation.json` | Options, repairs, protonation, inferred disulfides, warnings, timings and minimisation results |

PDBFixer repairs missing heavy/terminal atoms; OpenMM places the final hydrogens. Optional Reduce processing changes side-chain orientations only when requested, then OpenMM rebuilds hydrogens against the repaired topology. There is no silent fallback between methods. Preparation does not reconstruct missing sequence stretches, and unsupported protonation/force-field combinations fail explicitly.

Relaxation uses Amber ff14SB with GBn2 implicit solvent. It first optimises hydrogen positions with heavy atoms fixed, then minimises with positional restraints on heavy atoms. Select `--relax-platform cuda` on Spark: the portable default is `cpu`, which was substantially slower. Inspect the audit's convergence and geometry warnings before downstream use. Lower energy or fewer overlaps does not establish greater accuracy against an experimental structure.

**Changed options:** `--relaxation` and `-r` have been removed and now fail with an explanation. Use `--add-hydrogens` for preparation or `--relax restrained` for genuine minimisation. `-H` selects the heavy-chain sequence; it is not a hydrogen-addition alias. Preparation currently requires PDB output.

See the [preparation guide](docs/abodybuilder3_preparation_251026.md) for settings, units, failure handling and scientific limits.

## Measured performance

Measurements on **DGX Spark / GB10, 25 October 2026**, comparing the preceding dev commit `07bfe1e` with this implementation. Each complete job ran in a fresh offline container and included imports, model loading, feature preparation, inference and PDB writing, with the same raw-output hashing instrumentation. One warm-up preceded two measured runs, with conditions interleaved and the final order reversed. Both used automatic batching and one GPU model.

| Model / 1,000 inputs | Previous dev | Current dev | Current sampled process RSS, GiB | Current peak CUDA allocated, GiB |
|---|---:|---:|---:|---:|
| Base, five pairs repeated | 33.08 s | **28.43 s** | 1.789 | 3.782 |
| pLDDT, five pairs repeated | 33.79 s | **28.93 s** | 1.789 | 3.783 |
| Base, distinct synthetic variants | 33.38 s | **28.56 s** | 1.790 | 3.782 |
| pLDDT, distinct synthetic variants | 33.96 s | **29.12 s** | 1.800 | 3.783 |

Times are medians; memory columns give the higher measured peak. RSS is sampled process-tree memory, while CUDA values are the exact per-process PyTorch allocator peaks. **Do not add them** to estimate physical unified-memory use. Full ranges, the 100-input checks, component ablations and memory definitions are in the [current dev benchmark report](docs/abodybuilder3_dev_optimisation_251026.md).

The distinct inputs are deterministic substitutions in the five public sequence pairs, retaining their 227–231 combined lengths. They test implementation and throughput rather than biological accuracy or broad real-antibody scaling. The [earlier report](docs/abodybuilder3_dev_batching_251026.md) retains the matched main comparison and language measurements; the language path is unchanged and its timings remain highly variable.

All **18,610 matched candidate predictions**, including warm-ups, had identical raw coordinate/mask/confidence fingerprints and PDB bytes. Another **960 strict GPU regression comparisons** covered multiple batch sizes, padding and substitutions, with zero new coordinate differences. The earlier introduction of batching produced differences up to **0.008 Å** in Cα coordinates relative to main; these new optimisations preserve the matched previous-dev batches exactly. These checks establish implementation agreement on the tested inputs, not accuracy against experimental structures.

The installed CLI also passed the three chemical preparation/relaxation workflows. Those stages remain separate from raw prediction and retain geometry warnings and audit records. For matched per-antibody chemistry timings, see the [preparation benchmarks](docs/abodybuilder3_preparation_benchmarks_251026.md).

## Reproducible builds and verification

The Dockerfile pins the NGC base image by digest, the source foundation by commit, Reduce by revision and Python dependencies by version. It verifies all three ABodyBuilder3 checkpoint hashes. This checkout's scripts, batching/feature adapters and reviewed OpenFold-derived inference module are copied into the image separately. The trained weights and all other captured runtime sources/dependencies match the preceding dev runtime.

| Component | Tested runtime |
|---|---|
| Base image | NVIDIA NGC PyTorch 25.11, pinned digest |
| Python / PyTorch / CUDA | 3.12 / NGC 2.10 prerelease / 13.0 |
| OpenMM / PDBFixer | 8.6.1 / 1.12.0, published ARM64 packages |
| OpenMM CUDA dependencies | Published wheels pinned to CUDA 13.0 |
| Reduce | 4.17, pinned source revision |
| ABodyBuilder3 weights | Three checkpoints from [Zenodo 11354577](https://zenodo.org/records/11354577); approximately 441 MB compressed archive |

The deployment retains the Flash Attention import compatibility fix, PyTorch checkpoint-loading compatibility and fast PDB writer. It replaces the historical custom OpenMM/PDBFixer builds with published packages. CUDA dependency pins avoid the unsupported-PTX failure observed with newer NVRTC wheels on the tested driver.

Ubuntu packages and build tools can change, so an identical Docker image digest is not promised across build dates. The [rebuild and verification guide](docs/abodybuilder3_reproducibility_251026.md) explains source/checkpoint/runtime checks, prediction comparisons, the original build tag and the limits of reproducibility. For this dev image, use `tests/reference_dev_optimised_runtime_251026.json` as the strict runtime reference. The older batching reference remains attached to its historical results. The retained NGC settings allow TF32 for FP32 matrix multiplications; details are in the current report. `--seed` alone does not guarantee identical prepared or relaxed coordinates across platforms.

CPU-only interface and measurement checks:

```bash
python3 -m unittest discover -s tests -p 'test_batch_pipeline.py' -v
python3 -m unittest discover -s tests -p 'test_structure_preparation.py' -v
python3 -m unittest discover -s tests -p 'test_deployment_comparison.py' -v
```

The measurement checks require NumPy. `tests/test_batch_cli.py` and `tests/test_inference_features.py` run inside the deployment image. `tests/check_optimised_gpu.py` checks exact agreement for the new inference path; the older `tests/check_batch_gpu.py` checks batching against serial execution with its original tolerance. Mount the repository when running its tests/tools inside Docker. Real chemistry tests run inside the Docker environment with saved public predictions, as described in the preparation guide. An import test alone does not verify CUDA minimisation or prediction agreement.

## Citation and licence

Please cite the original ABodyBuilder3 work when using the model:

> Kenlay H, Dreyer FA, Cutting D, Nissley D, Deane CM. **ABodyBuilder3: improved and scalable antibody structure predictions.** *Bioinformatics* 40(10), btae576 (2024). [doi:10.1093/bioinformatics/btae576](https://doi.org/10.1093/bioinformatics/btae576).

This repository is distributed under [Apache 2.0](LICENSE.txt). External components and downloaded assets retain their own licences; see [upstream ABodyBuilder3](https://github.com/Exscientia/abodybuilder3), the [checkpoint record](https://zenodo.org/records/11354577) and [ProtT5](https://huggingface.co/Rostlab/prot_t5_xl_uniref50).
