# Accepted runtime promoted to main — 26 October 2026

Main now ships strict FP32, saved AOT executables, length-aware batching, the native PDB writer and the direct batch-1 ProtT5 lifecycle. These are the accepted dev implementations, copied without model-code changes. Optional preparation remains serial. Experimental candidates and all research reports, tests and benchmark tools remain on dev/component branches.

The main README describes installation and shipped behaviour. A single public CSV example replaces the build dependency on the test cohort. Main retains a short operational preparation guide; its benchmark reports and upstream training launch/config/data scaffolding are retained on dev rather than shipped as deployment documentation. The full model library remains available.

## Verification

The image was built from the promotion checkout using the pinned public source/dependency foundation, then exported all four AOT packages afresh. Existing dependency layers were reused; the preceding no-cache qualification still applies to the unchanged foundation. The native formatter built successfully, and the installed README example completed as a non-root user.

Fresh offline containers compared 100 distinct antibodies per base/pLDDT checkpoint with accepted dev. Peak CUDA allocation was unchanged: 1.213 GiB for both. Complete-container times were 6.33 versus 5.97 seconds for base and 6.14 versus 6.37 seconds for pLDDT (dev versus main). These single runs qualify packaging and do not establish a small speed difference.

Rebuilt AOT packages are not byte-identical to the previous packages. Mean unaligned Cα RMSD was 0.000000984 Å for base and 0.000000967 Å for pLDDT. Maximum atom displacement was 0.000922 Å and 0.000124 Å respectively. Both pass the existing numerical gates: maximum Cα displacement 0.001 Å, aligned Cα RMSD 0.0001 Å and maximum atom displacement 0.01 Å; using unaligned RMSD here is stricter. Atom masks, sequence identities and PDB non-coordinate fields matched. PDB coordinate changes were at most one 0.001 Å rounding unit. No change to structural accuracy is claimed.

Eight installed language predictions matched the accepted embeddings, raw outputs and PDB bytes exactly. The direct language check took 11.89 seconds including container startup and evidence saving. Its peak CUDA allocation was 4.706 GiB. The shipped one-input README command, relative documentation links, syntax and missing-GPU build guard also passed.

An initial exact-fingerprint comparison detected the expected AOT rounding differences. A follow-up diagnostic initially confused prior observed extrema with acceptance limits; qualification then used the predeclared gates in `tools/score_deployment.py`. No model was altered to pass a check, and the completed runs were retained.

## Release and records

- Promotion commit: `78163003c6c18aa0557b23aae91848911d40553b`.
- Rollback tag: `pre-aot-261026`; accepted deployment tag: `optimised-runtime-261026`.
- Verified main image: `abodybuilder3-official-spark:accepted-runtime-261026`, ID `sha256:ab981f1494f2c817995a3cf767702181330af8d3f97584120d533dedd9432fd0`.
- [Compact source map, commands and measurements](data/abodybuilder3_main_promotion_261026.json).
- Local detailed evidence: `/home/adrian/codex_wd/projects/abodybuilder3/runs/main_promotion_261026`.

No Docker images, host datasets, model assets or audit records were removed. The previous main remains reachable through its rollback tag; dev retains all removed research files and its existing runtime.
