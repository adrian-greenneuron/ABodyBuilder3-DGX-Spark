# ProtT5 prediction lifecycle component

This isolated branch retains the original upstream wrapper, the candidate,
comparison/contract tests and [measured results](docs/prott5_lifecycle_261026.md).
The candidate preserves the original batch-1 encoder arithmetic and outputs;
it removes repeated Trainer teardown and GPU/CPU model transfers. Main is unchanged.

The passing adapter is installed on [dev](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/tree/dev)
with its original reference backend available explicitly. No weights or datasets
are stored on this branch. See `provenance_261026.json` for the source identity.
