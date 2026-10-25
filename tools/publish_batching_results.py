"""Publish the completed dev batching study without changing measurements."""
import argparse,hashlib,json,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--docs',type=Path,default=Path('docs'))
    args=p.parse_args();root=args.root;docs=args.docs;data=docs/'data';data.mkdir(parents=True,exist_ok=True)
    assert json.loads((root/'completed_251026.json').read_text())['completed']
    audit=json.loads((root/'output_audit_251026.json').read_text());assert audit['passed']
    rows=[]
    for path in sorted(root.glob('*/*/measurements.json')):
        for r in json.loads(path.read_text()):
            assert r['completed']
            rows.append(dict(case=path.parent.parent.name,**r))
    qualification=json.loads((root/'qualification_final/agreement.json').read_text())
    chemistry={}
    for method in ['openmm','reduce','relax']:
        r=json.loads(next((root/f'e2e_{method}').glob('*.preparation.json')).read_text())
        chemistry[method]={k:r[k] for k in ['status','warnings','options','versions']}
        if 'restrained_minimisation' in r:
            chemistry[method]['restrained_minimisation']=r['restrained_minimisation']
    identities={name:hashlib.sha256((root/name/'predict_main.py').read_bytes()).hexdigest() for name in ['frozen_main','frozen_dev','frozen_final']}
    result={'main_commit':'a5deeb1b73994491f94b5724d2f2baa9993d47a6','previous_dev_commit':'01047c73f7505d46db808f42aa32f1c624a3e21e',
            'measured_pipeline_commit':'230ef920bf4879d1b398704b487cca322cd8d895',
            'cli_sha256':identities,'measurements':rows,'qualification':qualification,'chemistry':chemistry,
            'output_audit':audit,'small_screen_selection':json.loads((root/'screen_selection_251026.json').read_text()),
            'final_configuration':{'batch_size':'auto','small_job_cap':16,'large_job_cap':32,'small_job_max_inputs':128,
                                  'cpu_workers_per_stage':2,'length_bin_residues':8,'sorting_window':512,'memory_target_gib':6}}
    (data/'abodybuilder3_dev_batching_measurements_251026.json').write_text(json.dumps(result,indent=2)+'\n')
    def select(case,model):return [r for r in rows if r['case']==case and r['model']==model and not r['warmup']]
    def median(rs):return statistics.median(r['complete_process_seconds'] for r in rs)
    lines=['# Dev batching: design and measurements — 25 October 2026','',
        'Dev now includes main’s modernised runtime and explicit chemistry pipeline, with an integrated length-aware prediction pipeline. Main was left unchanged. Model source, released checkpoints and FP32 settings are unchanged. This is an implementation/throughput comparison, not a new model or an experimental-structure accuracy benchmark.','',
        '## Complete-job performance','',
        'Each measurement is a fresh offline container, including imports, model loading, feature preparation, prediction and writing every PDB. Each condition has one whole-job warm-up and two measured repetitions. The 1,000-input workloads cycle the same five public antibody pairs (227–231 residues); they are not 1,000 independent sequences. Main used eight GPU-owning workers; dev batching uses one model process. Runs were sequential; the first large comparison interleaved conditions and reversed the final order. The batch-cap-32 follow-up and final-image default checks were performed afterwards.','',
        '| Model | Condition | Complete time, median (range), s | Predictions/min | Peak summed RSS, GiB | Sampled concurrent CUDA allocation, GiB | Container accounted peak, GiB |',
        '|---|---|---:|---:|---:|---:|---:|']
    for model in ['base','plddt']:
        for case,label in [('large-main','Main, 8 workers'),('large-serial','Dev, serial CPU formatting'),('final-default','Dev, automatic batching')]:
            rs=select(case,model);m=median(rs)
            lines.append(f'| {model} | {label} | {m:.2f} ({min(r["complete_process_seconds"] for r in rs):.2f}–{max(r["complete_process_seconds"] for r in rs):.2f}) | {60000/m:.0f} | {max(r["sampled_process_tree_rss_peak_bytes"] for r in rs)/2**30:.3f} | {max(r["sampled_concurrent_cuda_allocated_peak_bytes"] for r in rs)/2**30:.3f} | {max(r["container_lifetime_memory_peak_bytes"] for r in rs)/2**30:.3f} |')
    ratios={m:median(select('large-main',m))/median(select('final-default',m)) for m in ['base','plddt']}
    lines += ['',f'Final default throughput improved **{ratios["base"]:.2f}× for base and {ratios["plddt"]:.2f}× for pLDDT** versus the measured main CLI. The serial control shows that moving coordinate formatting to CPU contributes separately from GPU batching.','',
        'Memory columns are the higher measured peak across repetitions. Summed RSS double-counts some shared pages; sampled PyTorch allocations can miss short peaks. Container accounting includes warm-up/loading activity and differs from RSS. These quantities must not be added to estimate physical unified-memory usage. Dev has lower measured RSS/container accounting here, but higher CUDA allocation than main’s sampled concurrent total.','',
        '## Batch and CPU-worker selection','',
        '| Workload | Batch cap | CPU workers per stage | Complete time, median (range), s |','|---|---:|---:|---:|']
    configs=[('screen-b16-c2',100,16,2),('screen-b32-c1',100,32,1),('screen-b32-c2',100,32,2),('screen-b32-c4',100,32,4),('screen-b64-c2',100,64,2),('large-batched',1000,16,2),('full-screen-b32c1',1000,32,1),('full-screen-b32c2',1000,32,2)]
    for case,count,batch,cpu in configs:
        rs=select(case,'plddt');lines.append(f'| pLDDT, {count:,} inputs | {batch} | {cpu} | {median(rs):.2f} ({min(r["complete_process_seconds"] for r in rs):.2f}–{max(r["complete_process_seconds"] for r in rs):.2f}) |')
    lines += ['',
        'The small screen favoured a cap of 16; the full-size follow-up slightly favoured 32 with two CPU workers per stage. Differences between nearby settings are small and two repetitions do not establish a universal optimum. `auto` therefore caps jobs of at most 128 inputs at 16 and larger jobs at 32. The memory estimate can reduce actual batches further: a requested cap of 64 in the screen did not guarantee 64 inputs per batch under the 6-GiB target. Explicit `--batch-size 16` remains a useful lower-allocation option.','',
        '## Language checkpoint','',
        'Language uses the existing single-model path, with fresh embeddings from the same pinned offline ProtT5 snapshot. The test has five inputs rather than 1,000. It has no new GPU embedding batching or prediction cache.','',
        '| Condition | Complete time, median (range), s | Peak summed RSS, GiB | Sampled CUDA allocation, GiB |','|---|---:|---:|---:|']
    for case in ['language-main','language-dev']:
        rs=select(case,'language');lines.append(f'| {case} | {median(rs):.2f} ({min(r["complete_process_seconds"] for r in rs):.2f}–{max(r["complete_process_seconds"] for r in rs):.2f}) | {max(r["sampled_process_tree_rss_peak_bytes"] for r in rs)/2**30:.3f} | {max(r["sampled_concurrent_cuda_allocated_peak_bytes"] for r in rs)/2**30:.3f} |')
    lines += ['',
        'These language timings remain highly variable. They do not establish a reliable speed improvement; new language batching is explicitly rejected rather than silently applying base-model features.','',
        '## Architecture and limits','',
        '- One process owns the GPU model. CPU feature preparation and output formatting/writing overlap inference through bounded queues; `--cpu-workers 2` permits two workers per CPU stage. Native PyTorch CPU threads are limited to one in this mode to avoid nested oversubscription.',
        '- Within windows of 512 inputs, sequences are grouped by combined heavy/light length into eight-residue bins. Batches also obey a count cap and an approximate pair-feature budget. Input identities and unique output names survive sorting; duplicate names cannot overwrite a naturally occurring suffixed name.',
        '- Dense features are constructed directly in their final FP32 layout, with exact agreement against main’s CPU features. Padding is masked. Atom conversion is batched and coordinates are transferred to CPU before formatting; full GPU activations and all PDB strings are not retained for the whole workload.',
        '- `--batch-memory-gib 6` is an estimated inference-allocation target, not a hard process/GPU/system-memory ceiling. Its coefficient was calibrated on the small public workload. A CUDA out-of-memory error splits the affected batch, logs retries and eventually fails clearly if a single item cannot fit. Other exceptions propagate. There is no fallback to lower precision or another model.',
        '- Ordinary output uses bounded queues, but CSV/FASTA sequence metadata and batch-audit metadata are still retained in host memory. A single very long sequence can exceed the estimate. Broader loop lengths and diverse real sequences need additional evaluation.',
        '- Chemistry remains optional and follows raw prediction, using main’s explicit PDBFixer/OpenMM/Reduce pipeline and separate raw/prepared/relaxed files. OpenMM GPU work does not overlap structure-model inference. There is no new claim of improved experimental-structure accuracy.',
        '- No CUDA graphs, model compilation, mixed-precision change or output caching is enabled. High CPU/GPU percentages are not an acceptance criterion: completed predictions per minute is the objective.','',
        '## Validation','',
        f'- Every saved benchmark PDB was audited: **{audit["files_checked"]:,} files**, including warm-ups and screens. Counts, atom/residue identities and finite coordinates passed. All serial main/dev comparisons were byte-identical. Batched PDBs had a maximum Cα coordinate difference of {max(r["max_ca_coordinate_difference_angstrom"] for r in audit["cases"]):.3f} Å and maximum atom coordinate difference of {max(r["max_atom_coordinate_difference_angstrom"] for r in audit["cases"]):.3f} Å versus matched main predictions.',
        f'- Fourteen raw-array comparisons covered base/pLDDT, the five public pairs and artificial shorter/longer padding tests under the actual bucketed scheduler. Maximum Cα difference was {max(r["max_ca_coordinate_difference"] for r in qualification):.6f} Å, maximum all-atom coordinate RMS difference {max(r["atom_coordinate_rms"] for r in qualification):.6f} Å, and maximum derived pLDDT difference {max(r["max_plddt_score_difference"] for r in qualification if r["max_plddt_score_difference"] is not None):.4f} points. Raw Cα acceptance remained below 0.01 Å. The PDB audit allows rounding of the three-decimal coordinates.',
        '- An early unbounded mixed-length stress test exceeded the Cα tolerance; this motivated the narrow length bins. The passing claim applies to the final scheduler and tested inputs, not arbitrary batch compositions.',
        '- Twenty-two CPU/interface tests passed: scheduling/queue boundedness, failure propagation, output formats, duplicate naming, finite options, simulated CUDA OOM splitting, preparation policy and measurement correctness. The final installed CLI also completed real OpenMM preparation, Reduce orientation plus OpenMM preparation, and restrained CUDA minimisation on the public example. Geometry warnings were retained; minimisation met its convergence criterion.',
        '- The 54 core model-source files and all three checkpoints match the retained main runtime. Numeric kernels/features are unchanged between the final default confirmation and published image; the final CLI also adds the small-job cap and collision-safe names. Frozen CLI hashes and image IDs identify each measurement stage.',
        '- Confidence export remains main’s existing limitation: the CLI JSON/filter path contains flattened logits and PDB B-factors are zero. Derived pLDDT agreement above converts the model logits correctly and does not qualify the CLI confidence filter.','',
        '## Reproduce','',
        'Build this dev checkout with `./docker_build.sh build`; its default image name is `abodybuilder3-dev-spark:latest`. A fresh build uses the public pinned model/runtime recipe and copies the dev CLI, scheduler and batching module into the image. `NO_CACHE=1` disables the Docker build cache.','',
        'For strict identity verification, capture the built image with `tools/capture_runtime.py` and compare against `tests/reference_dev_batch_runtime_251026.json`. That manifest describes the final qualified image, including both new modules. The stable preparation reference describes main and is intentionally different. Commit `230ef920bf4879d1b398704b487cca322cd8d895` preserves the pipeline source used by the screen and serial controls; the final image differs in the adaptive small-job cap and collision-safe filename allocation.','',
        'The study uses `tools/benchmark_cli.py`, `tools/run_batching_study.py`, `tests/check_batch_gpu.py` and `tools/audit_batching_outputs.py`. The initial controller performs the small screen and large 16-cap comparison; the 32-cap follow-up and final-default rows are separate explicit invocations of the same benchmark tool. All arguments/configurations and warm-up flags are retained in the measurement JSON. `tools/publish_batching_results.py` regenerates this report from the local completed study.','',
        '[Measurements and validation records](data/abodybuilder3_dev_batching_measurements_251026.json) · [Main preparation guide](abodybuilder3_preparation_251026.md) · [Historical batching comparison](abodybuilder3_benchmarks_251026.md)','']
    (docs/'abodybuilder3_dev_batching_251026.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
