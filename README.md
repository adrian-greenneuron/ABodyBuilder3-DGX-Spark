# AOT pipeline experiments

This component branch preserves the original adapters, experimental alternatives,
tests and [results for priorities A–E and G](docs/aot_pipeline_optimisation_261026.md).
It is an experiment record, not a replacement application.

The production reference is [dev](https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark/tree/dev)
at `d3c7444`, with the pinned AOT/native image recorded in each protocol.
`reference/` contains the original source; `candidate/` contains deferred output
transfers, alternative shape routing and a resident-process prototype. No model
weights, executable packages, sequences or predicted coordinates are committed.

CPU tests run in the pinned application runtime:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

`tools/check_deferred_gpu.py` is the additional device test. The measurement
controllers retain their image and source identities in the evidence archive;
generated packages and biological inputs are external. `tools/publish_pipeline_results.py`
regenerates the figures from the retained local study directory. See the report
for the distinction between exact-output changes, new compiler rounding and
startup amortisation.
