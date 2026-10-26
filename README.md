# Deterministic atom-repair experiments

**CPU-one-thread repair + eight preparation workers was 5.2–5.4× faster than
current serial preparation**, with ~2.29 GiB additional unified RAM for OpenMM
hydrogens and ~3.39 GiB with Reduce. CPU repair and CUDA-double repair repeated
exactly in the tests. Their prepared hydrogen coordinates differ from the
current repair, so **neither candidate is enabled on dev or main**.

See the [report, figures and comparison tables](docs/deterministic_repair_261026.md).
The earlier [preparation-concurrency findings](docs/preparation_optimisation_261026.md)
are retained alongside them.

- `reference/`: original wrapper and exact installed PDBFixer source with its MIT notice.
- `candidate/`: explicit repair Context properties, wrapper, scheduler and reviewable patches.
- `Dockerfile.deterministic`: adds only the experimental repair code to the retained candidate runtime.
- `tools/`: repeatability screens, full-CLI/shared-RAM measurements, coordinate/chemistry diagnostics and plots.
- `tests/`: policy, source-scope and bounded-scheduler regression checks.

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Actual chemistry/plotting tools use the pinned application runtime. Input
sequences, predictions, full atom arrays and per-residue identity lists remain
local. Compact evidence and checksums are included in the report. The image
`abodybuilder3-dev-spark:deterministic-repair-screen2-261026` is experimental;
it does not replace `latest` or change either production branch.
