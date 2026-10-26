# Optional structure-preparation experiments

This isolated component branch contains the original preparation wrapper,
candidate process scheduler, integration patch, comparison tools and tests.
The [report and figures](docs/preparation_optimisation_261026.md) include the
coordinate differences found during final qualification. **The candidate is not
enabled on dev or main.**

`reference/structure_preparation.py` is the unchanged chemistry implementation
from dev `49080c3`. `candidate/preparation_pool.py` schedules independent CPU-1
preparation jobs; `candidate/dev_integration.patch` records the experimental CLI
and Docker changes. No internal OpenMM, PDBFixer or Reduce source is changed.
The retained candidate Docker images contain the complete runnable experiment;
they are distinct from `abodybuilder3-dev-spark:latest`.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

These policy tests need only Python. Actual chemistry checks and comparison
tools require the pinned application runtime and the retained public raw PDBs.
No structures, sequences, weights or private data are committed here. Compact
measurements, protocols and checksums accompany the report. The original
failed test fixture and all later qualification evidence remain in the local
study, with their correction recorded explicitly.
