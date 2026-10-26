# Detailed overnight benchmark tables — 26 October 2026

These are the unchanged numerical tables supporting the [main report and figures](abodybuilder3_overnight_optimisation_261026.md). Complete-job times are medians of two fresh runs. Memory is the higher peak of those runs; accuracy changes cover both repetitions, with 100 distinct experimental targets rather than 200 independent samples. All distances are Å and lDDT changes are percentage points. Bold timing cells identify the lowest displayed median within each row, not significance.

## Complete-job times at 10,000 inputs

| Checkpoint | Precision | Current dev (s) | Native writer (s) | Cached compiler + writer (s) | Cached attention + compiler + writer (s) | Combined vs dev |
| --- | --- | --- | --- | --- | --- | --- |
| base | TF32 permitted | 253.52 | 213.58 | 86.20 | **84.87** | 2.99× |
| base | Strict FP32 | 254.53 | 216.22 | **88.48** | 88.72 | 2.87× |
| plddt | TF32 permitted | 257.34 | 213.96 | 87.04 | **86.36** | 2.98× |
| plddt | Strict FP32 | 258.72 | 215.63 | 89.18 | **88.90** | 2.91× |

## Paired experimental accuracy and numerical changes

References use current dev at the same precision. H3 RMSD follows heavy-chain framework Cα alignment. H3 means are averaged over the two identical fresh-process results; deviation columns show the larger value across the repetitions. Raw atom displacement measures disagreement between predictions, not experimental side-chain error.

| Checkpoint | Precision | Candidate | Mean H3 RMSD to experiment | Largest H3 RMSD change | Largest raw Cα displacement | Largest raw atom displacement | Largest Cα-lDDT change (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | TF32 permitted | Cached compiler + native writer | 2.913793 | 0.00248318 | 0.023713 | 1.403312 | 0.013080 |
| base | TF32 permitted | Cached attention + compiler + native writer | 2.913700 | 0.00271644 | 0.030447 | 0.293824 | 0.022641 |
| base | Strict FP32 | Cached compiler + native writer | 2.914036 | 0.00000434 | 0.000031 | 0.001338 | 0.000000 |
| base | Strict FP32 | Cached attention + compiler + native writer | 2.914036 | 0.00000795 | 0.000032 | 0.000401 | 0.004186 |
| plddt | TF32 permitted | Cached compiler + native writer | 2.799059 | 0.00297330 | 0.070995 | 0.224426 | 0.016504 |
| plddt | TF32 permitted | Cached attention + compiler + native writer | 2.799105 | 0.00338361 | 0.062709 | 0.186948 | 0.025130 |
| plddt | Strict FP32 | Cached compiler + native writer | 2.799067 | 0.00000462 | 0.000063 | 0.000242 | 0.004188 |
| plddt | Strict FP32 | Cached attention + compiler + native writer | 2.799067 | 0.00000732 | 0.000113 | 0.000458 | 0.004219 |

## Peak memory at 10,000 inputs

Values are GiB. CUDA, process-tree RSS and container accounting are different measures and must not be added on unified memory. RSS can double-count shared compiler-worker mappings. The two-process raw measurements remain in the evidence archive linked from the main report.

| Checkpoint | Precision | Approach | CUDA allocated | CUDA reserved | Process-tree RSS | Container lifetime |
| --- | --- | --- | --- | --- | --- | --- |
| base | TF32 permitted | Current dev | 3.992 | 4.195 | 1.835 | 2.924 |
| base | TF32 permitted | Native writer | 3.992 | 4.195 | 1.836 | 2.908 |
| base | TF32 permitted | Cached compiler + native writer | 2.259 | 2.359 | 4.643 | 3.472 |
| base | TF32 permitted | Cached attention + compiler + native writer | 2.268 | 2.359 | 4.666 | 3.457 |
| base | Strict FP32 | Current dev | 3.992 | 4.195 | 1.847 | 2.921 |
| base | Strict FP32 | Native writer | 3.992 | 4.195 | 1.874 | 2.937 |
| base | Strict FP32 | Cached compiler + native writer | 2.259 | 2.359 | 4.646 | 3.497 |
| base | Strict FP32 | Cached attention + compiler + native writer | 2.268 | 2.359 | 4.646 | 3.495 |
| plddt | TF32 permitted | Current dev | 3.992 | 4.197 | 1.859 | 2.945 |
| plddt | TF32 permitted | Native writer | 3.992 | 4.197 | 1.849 | 2.923 |
| plddt | TF32 permitted | Cached compiler + native writer | 2.260 | 2.361 | 4.652 | 3.487 |
| plddt | TF32 permitted | Cached attention + compiler + native writer | 2.268 | 2.361 | 4.657 | 3.484 |
| plddt | Strict FP32 | Current dev | 3.992 | 4.197 | 1.861 | 2.951 |
| plddt | Strict FP32 | Native writer | 3.992 | 4.197 | 1.876 | 2.929 |
| plddt | Strict FP32 | Cached compiler + native writer | 2.260 | 2.361 | 4.638 | 3.521 |
| plddt | Strict FP32 | Cached attention + compiler + native writer | 2.268 | 2.361 | 4.639 | 3.524 |
