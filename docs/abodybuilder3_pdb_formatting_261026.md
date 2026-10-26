# Faster PDB output without changing predictions — 26 October 2026

The isolated C++ PDB writer reduced complete 10,000-input job times by **16–18%**, giving **1.19–1.22× throughput**. It passed every raw-prediction fingerprint and PDB-byte comparison for **88,820 matched outputs**, covering both base and pLDDT checkpoints at both precisions. This is an output-formatting change: model weights, inference operations and prediction settings are unchanged. Main and dev have not been modified.

## What changed

The reference constructs NumPy metadata arrays, Python strings and one formatted line per atom. The candidate formats those same fields in C++, releases the Python GIL while building the text, and preserves chain separators, numbering, spacing, rounding and final newlines. It uses a thread-local C numeric locale. Unsupported array types/shapes and non-finite coordinates use the original Python writer. The original function is retained verbatim in `reference/format_pdb.py`; provenance records its source hash.

The CPU-only screen covered **413 fixtures**, including 200 real predictions, empty inputs, signed zero, rounding/large-number cases, unknown residues and fallback behaviour. A further **200 outputs** were checked with eight concurrent writing threads. Median formatting time for 200 real predictions fell from **0.910 to 0.168 seconds (5.40×)**. This microbenchmark measures formatting, not disk writes or full inference.

## Complete-job measurements

Each table value is the median of two fresh-container runs. The five-input startup smoke test is retained separately. Both sides used the same experimental image, with the native writer explicitly off or on; order reversed for the second repetition. Times include container startup/exit, imports, weights, features, inference, coordinate conversion, raw fingerprints and writing every PDB. The fixed seed was 42; optional chemistry was disabled. Bold identifies the lower displayed median, not statistical significance.

The 100 distinct targets are published experimental-test antibodies, 219–242 residues in total. Larger jobs repeat those targets with distinct names and **no prediction caching**; they do not supply additional independent accuracy observations. Length groups, automatic GPU batch policy, two preparation/writing threads and precision settings matched within each comparison.

| Inputs | Checkpoint | Precision | Python writer (s) | Native writer (s) | Throughput ratio | Less elapsed time |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | base | TF32 permitted | 7.27 | **6.73** | 1.081× | 7.5% |
| 100 | base | Strict FP32 | 7.41 | **6.98** | 1.062× | 5.8% |
| 100 | plddt | TF32 permitted | 7.57 | **7.04** | 1.075× | 7.0% |
| 100 | plddt | Strict FP32 | 7.56 | **6.82** | 1.109× | 9.8% |
| 1,000 | base | TF32 permitted | 29.53 | **25.57** | 1.155× | 13.4% |
| 1,000 | base | Strict FP32 | 29.30 | **25.82** | 1.135× | 11.9% |
| 1,000 | plddt | TF32 permitted | 29.98 | **25.63** | 1.170× | 14.5% |
| 1,000 | plddt | Strict FP32 | 30.52 | **26.04** | 1.172× | 14.7% |
| 10,000 | base | TF32 permitted | 252.43 | **211.88** | 1.191× | 16.1% |
| 10,000 | base | Strict FP32 | 255.94 | **214.39** | 1.194× | 16.2% |
| 10,000 | plddt | TF32 permitted | 259.26 | **213.15** | 1.216× | 17.8% |
| 10,000 | plddt | Strict FP32 | 260.86 | **214.67** | 1.215× | 17.7% |

## Peak memory for 10,000 inputs

The higher peak of each matched pair is shown, in GiB. CUDA allocated memory was unchanged. RSS variations of a few tens of MiB do not establish a physical-memory improvement. Process-tree RSS may double-count shared mappings; container accounting is separate. On the Spark's unified-memory system, **do not add these columns** to estimate physical RAM consumption.

| Checkpoint | Precision | Writer | CUDA allocated | Process-tree RSS | Container lifetime |
| --- | --- | --- | --- | --- | --- |
| base | TF32 permitted | original | 3.992 | 1.814 | 2.888 |
| base | TF32 permitted | native | 3.992 | 1.828 | 2.916 |
| base | Strict FP32 | original | 3.992 | 1.821 | 2.904 |
| base | Strict FP32 | native | 3.992 | 1.829 | 2.903 |
| plddt | TF32 permitted | original | 3.992 | 1.825 | 2.926 |
| plddt | TF32 permitted | native | 3.992 | 1.832 | 2.917 |
| plddt | Strict FP32 | original | 3.992 | 1.834 | 2.911 |
| plddt | Strict FP32 | native | 3.992 | 1.864 | 2.938 |

## Scope and decision

All 28 matched condition/replicate pairs passed, including four five-input smoke pairs. The full comparison checked masks, coordinates, confidence fingerprints and every PDB byte. These checks demonstrate exactness on the tested workloads; the original writer remains the fallback for inputs outside the native contract. This is the strongest exact-output optimisation from this study and is ready for review. It remains isolated on this component branch; it is **not enabled on dev**. The language checkpoint and optional hydrogenation/minimisation paths were not benchmarked.

The pinned parent is dev `bd8a12ea9f58fa3577215532653a05c47e90e256`, image `sha256:18456b5147ea1ec35d8de4f664962baeea6af1bb473ad9107bb1cbfa37fd844d`. The measured formatter image is `sha256:a53b5613e495565b4ebf71eab5659a5d81b8ecb09378fba6819a3cc82d40b8bf`. Runtime: DGX Spark GB10, NGC PyTorch 2.10.0a0+b558c986e8.nv25.11, CUDA 13.0. Every GPU job used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before CUDA initialisation. TF32-permitted means FP32 tensors with matrix precision `high`/`allow_tf32=True`; strict means `highest`/TF32 matrix multiplication disabled. No BF16 or FP16 inference was used.

The [measurement archive](data/abodybuilder3_pdb_formatting_evidence_261026.zip) contains individual timings, memory, source provenance and all validation outcomes. Raw structures, checkpoints and compiler caches remain local.
