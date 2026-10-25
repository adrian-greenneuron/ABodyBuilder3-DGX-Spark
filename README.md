# ABodyBuilder3 on NVIDIA DGX Spark

Predict antibody variable-region structures from paired heavy- and light-chain sequences on the DGX Spark's ARM64 CPU and NVIDIA GB10 GPU. This repository packages [ABodyBuilder3](https://github.com/Exscientia/abodybuilder3) in a reproducible Docker environment, with a parallel prediction CLI and optional chemical preparation and restrained energy minimisation.

**Main now contains the tested modernised deployment.** It uses published OpenMM/PDBFixer packages and updated Reduce, while retaining the original model settings and weights. The latest official model source and released weights checked on 25 October 2026 were already incorporated in this deployment; modernisation does not introduce a newly trained model.

The original build recipe remains available at [`original-runtime-251026`](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/tree/original-runtime-251026). Experimental GPU batching remains on [`dev`](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/tree/dev) and is not part of the default CLI.

## Requirements

- NVIDIA DGX Spark running Linux, with Docker and the NVIDIA Container Toolkit configured for GPU containers.
- A working NVIDIA driver. This deployment was tested with **580.178.04**; its CUDA dependencies are pinned to 13.0 for that driver.
- Internet access during the build to obtain the public base image, source and checkpoints. No existing local ABodyBuilder3 image or weights are needed.
- Space for an approximately **22 GB logical Docker image**, plus build cache and outputs. Docker layers may be shared with other images. The language model needs a separate **11.28 GB** ProtT5 download.

Memory use depends on the model, worker count and sequence lengths. The measurements below cover five public antibody pairs of 227–231 residues in total; they do not establish the largest supported input.

## Quick start

### 1. Build and check the environment

```bash
git clone https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark.git
cd ABodyBuilder3-DGX-Spark
./docker_build.sh build
./docker_build.sh test
```

The default image name is `abodybuilder3-official-spark:latest`. The build verifies the downloaded checkpoint hashes; the test checks imports and reports GPU availability. To rebuild without using Docker's build cache:

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
    abodybuilder3-official-spark:latest \
    python3 scripts/predict.py predict \
        --csv /inputs/antibodies_5.csv --output /predictions/ \
        --model plddt --workers 1 --seed 42
```

Raw heavy-atom PDB files are saved in `predictions/` on the host. The image runs as root by default, so files written through the mount may be owned by root.

### 3. Use your own sequences

Provide a CSV with `name,heavy,light` columns, one paired antibody per row. The generated example CSV shows the expected format. Mount its directory as `/inputs`, then change the `--csv` filename above. Keep the trailing `/` in `--output /predictions/` to request a directory.

The CLI also accepts `--heavy`/`--light` sequences and paired FASTA files. To see all options:

```bash
docker run --rm abodybuilder3-official-spark:latest \
    python3 scripts/predict.py predict --help
```

## Models and parallel prediction

| Selection | Model | Additional assets |
|---|---|---|
| `--model plddt` | Default checkpoint, trained with the pLDDT confidence head | Included in the image |
| `--model base` | Base checkpoint | Included in the image |
| `--model language` | Checkpoint using ProtT5 sequence embeddings | Separate ProtT5 download |

These are different trained checkpoints. Deployment checks compare each with its own original counterpart; they do not rank their accuracy against experimental structures.

For large base/pLDDT batches, **`--workers 8`** was effective on the Spark. `--workers auto` is the default and selects eight on the tested system. Each prediction worker loads its own model copy, so reducing the worker count can reduce memory use. Use **`--workers 1` for the language model** to match the tested configuration. More workers do not necessarily improve throughput.

**Confidence export limitation:** the current CLI's JSON and `--confidence-threshold` paths use flattened confidence logits rather than properly converted pLDDT scores; PDB B-factors are zero. Do not interpret those exported values as calibrated pLDDT or use the threshold for scientific filtering. This existing limitation is documented in the [reproducibility notes](docs/abodybuilder3_reproducibility_251026.md#interpretation-and-limitations); the model's confidence head itself is unchanged.

### Prepare a reusable ProtT5 download

The CLI can download ProtT5 on first use, but a host-mounted snapshot avoids downloading it again in disposable containers. The helper pins the revision and downloads only the five inference assets:

```bash
mkdir -p prott5
docker run --rm \
    -v "$PWD/tools:/tools:ro" -v "$PWD/prott5:/prott5" \
    abodybuilder3-official-spark:latest \
    python3 /tools/download_prott5.py --output /prott5

mkdir -p language_predictions
docker run --rm --gpus all --ipc=host \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$PWD/benchmark_inputs:/inputs:ro" \
    -v "$PWD/prott5:/opt/abodybuilder3/Rostlab/prot_t5_xl_uniref50:ro" \
    -v "$PWD/language_predictions:/predictions" \
    abodybuilder3-official-spark:latest \
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

Measurements on **DGX Spark / GB10, 25 October 2026**. Each complete prediction job ran in a fresh container and included imports, model loading, inference and PDB writing. One warm-up preceded two measured runs. The workloads cycle five public antibody pairs; these are throughput measurements rather than an accuracy study of distinct antibodies.

| Model | Inputs | Workers | Complete time, median (range) | Peak summed RSS, GiB | Peak PyTorch CUDA allocation, GiB |
|---|---:|---:|---|---:|---:|
| Base | 1,000 | 8 | 82.18 s (81.96–82.41) | 14.577 | 1.018 |
| pLDDT | 1,000 | 8 | 83.32 s (83.18–83.45) | 14.620 | 0.978 |
| Language | 20 | 1 | 328.83 s (314.65–343.00) | 8.894 | 4.552 |

Memory columns give the higher measured peak across the two runs. Summed process-tree RSS can count shared pages more than once; CUDA allocation is a sampled concurrent PyTorch measurement and can miss short peaks. **Do not add these columns** to estimate physical memory use on the Spark's unified-memory system.

All **12,120 raw PDB files**, including warm-ups, matched the retained original reference byte for byte. Base/pLDDT median runtimes were within 1.3% of the previous modernised CLI. Language timings were variable; two repetitions do not establish a reliable small speed difference.

Optional preparation adds a separate per-antibody cost. For the pLDDT examples, median preparation took **8.46 s** with OpenMM, **9.52 s** with Reduce orientation followed by OpenMM, and **10.66 s** including restrained CUDA relaxation. These exclude inference, startup and file writing. All 30 measured CUDA relaxations met the force criterion; this is a finite test set, not a guarantee for arbitrary structures.

- [Current prediction and preparation benchmarks](docs/abodybuilder3_preparation_benchmarks_251026.md), with protocol, memory definitions and raw measurement links.
- [Earlier original/rebuilt/modernised and dev-batching comparison](docs/abodybuilder3_benchmarks_251026.md), using the frozen earlier CLI.

## Reproducible builds and verification

The Dockerfile pins the NGC base image by digest, the model source by commit, Reduce by revision and Python dependencies by version. It verifies all three ABodyBuilder3 checkpoint hashes. The model source is kept at the tested commit; this checkout's prediction and preparation scripts are copied into the image separately.

| Component | Tested runtime |
|---|---|
| Base image | NVIDIA NGC PyTorch 25.11, pinned digest |
| Python / PyTorch / CUDA | 3.12 / NGC 2.10 prerelease / 13.0 |
| OpenMM / PDBFixer | 8.6.1 / 1.12.0, published ARM64 packages |
| OpenMM CUDA dependencies | Published wheels pinned to CUDA 13.0 |
| Reduce | 4.17, pinned source revision |
| ABodyBuilder3 weights | Three checkpoints from [Zenodo 11354577](https://zenodo.org/records/11354577); approximately 441 MB compressed archive |

The deployment retains the Flash Attention import compatibility fix, PyTorch checkpoint-loading compatibility and fast PDB writer. It replaces the historical custom OpenMM/PDBFixer builds with published packages. CUDA dependency pins avoid the unsupported-PTX failure observed with newer NVRTC wheels on the tested driver.

Ubuntu packages and build tools can change, so an identical Docker image digest is not promised across build dates. The [rebuild and verification guide](docs/abodybuilder3_reproducibility_251026.md) explains source/checkpoint/runtime checks, prediction comparisons, the original build tag and the limits of reproducibility. `--seed` alone does not guarantee identical prepared or relaxed coordinates across platforms.

CPU-only interface and measurement checks:

```bash
python3 -m unittest discover -s tests -p 'test_structure_preparation.py' -v
python3 -m unittest discover -s tests -p 'test_deployment_comparison.py' -v
```

The second command requires NumPy. Real chemistry tests run inside the Docker environment with saved public predictions, as described in the preparation guide. An import test alone does not verify CUDA minimisation or prediction agreement.

## Citation and licence

Please cite the original ABodyBuilder3 work when using the model:

> Kenlay H, Dreyer FA, Cutting D, Nissley D, Deane CM. **ABodyBuilder3: improved and scalable antibody structure predictions.** *Bioinformatics* 40(10), btae576 (2024). [doi:10.1093/bioinformatics/btae576](https://doi.org/10.1093/bioinformatics/btae576).

This repository is distributed under [Apache 2.0](LICENSE.txt). External components and downloaded assets retain their own licences; see [upstream ABodyBuilder3](https://github.com/Exscientia/abodybuilder3), the [checkpoint record](https://zenodo.org/records/11354577) and [ProtT5](https://huggingface.co/Rostlab/prot_t5_xl_uniref50).
