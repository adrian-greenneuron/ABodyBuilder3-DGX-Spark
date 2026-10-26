"""Render the completed 26 October comparisons from their retained records."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt'] = 'abodybuilder3-overnight-261026'

DATE = '261026'
URL = 'https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark'
PRECISION = {'normal':'TF32 permitted', 'strict':'Strict FP32'}
LABELS = {'current_dev':'Current dev', 'native_writer':'Native writer',
          'cached_compiled_native':'Cached compiler + native writer',
          'cached_combined_native':'Cached attention + compiler + native writer'}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip()+'\n')


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def best(values, digits=2):
    formatted=[f'{x:.{digits}f}' for x in values]
    lowest=min(float(x) for x in formatted)
    return [f'**{x}**' if float(x)==lowest else x for x in formatted]


def compact(row):
    """Retain resource/identity evidence; omit large per-forward timing lists."""
    value=dict(row)
    if 'metrics' in value:
        value['metrics']={k:v for k,v in value['metrics'].items() if k!='forward_calls'}
    if 'batching' in value:
        value['batching']={k:v for k,v in value['batching'].items() if k!='batches'}
    return value


def evidence(path, payloads):
    path.parent.mkdir(parents=True,exist_ok=True)
    entries={name:json.dumps(payload,indent=2,allow_nan=False)+'\n' for name,payload in payloads.items()}
    if path.exists():
        with zipfile.ZipFile(path) as archive:
            if (set(archive.namelist())==set(entries) and archive.testzip() is None and
                    all(archive.read(name)==value.encode() for name,value in entries.items())):
                return
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name, value in entries.items():
            archive.writestr(name,value)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--formatter',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    runs=args.project/'runs'; docs=repo/'docs'
    bulk=read(runs/'cached_native_bulk_261026/bulk_analysis_261026.json')
    cached=read(runs/'cached_pipeline_261026/analysis_261026.json')
    cached2=read(runs/'cached_pipeline_261026/analysis_rep2_261026.json')
    policy=read(runs/'overnight_261026/batch_policy_analysis_261026.json')
    formatter_rows=read(runs/'native_formatter_pipeline_261026/measurements_261026.json')
    native_checks=read(runs/'native_formatter_pipeline_261026/validation_261026.json')
    native_done=read(runs/'native_formatter_pipeline_261026/completed_261026.json')
    assert native_done['matched_predictions']==88820
    assert all(r['raw_exact']==r['pdb_exact']==r['count'] for r in bulk['exact_writer'])

    # Dedicated formatter component, including its unmodified Python reference.
    grouped=defaultdict(list)
    for row in formatter_rows:
        grouped[row['count'],row['model'],row['precision'],row['mode']].append(row)
    native_resources=[];native_times=[];native_memory=[]
    for count in [100,1000,10000]:
        for model in ['base','plddt']:
            for precision in ['normal','strict']:
                rows=[grouped[count,model,precision,mode] for mode in ['original','native']]
                times=[statistics.median(x['container_seconds'] for x in rr) for rr in rows]
                native_times.append([f'{count:,}',model,PRECISION[precision],*best(times),
                                     f'{times[0]/times[1]:.3f}×',f'{100*(1-times[1]/times[0]):.1f}%'])
                for mode,rr in zip(['original','native'],rows):
                    value={'count':count,'model':model,'precision':precision,'mode':mode,
                           'seconds_median':statistics.median(x['container_seconds'] for x in rr),
                           'seconds_runs':[x['container_seconds'] for x in rr],
                           'cuda_allocated_peak_gib':max(x['cuda_peak_allocated_bytes'] for x in rr)/2**30,
                           'process_tree_rss_peak_gib':max(x['sampled_process_tree_rss_peak_bytes'] for x in rr)/2**30,
                           'container_memory_peak_gib':max(x['container_lifetime_memory_peak_bytes'] for x in rr)/2**30}
                    native_resources.append(value)
                    if count==10000:
                        native_memory.append([model,PRECISION[precision],mode,
                            f"{value['cuda_allocated_peak_gib']:.3f}",f"{value['process_tree_rss_peak_gib']:.3f}",
                            f"{value['container_memory_peak_gib']:.3f}"])
    writer_doc=f'''# Faster PDB output without changing predictions — 26 October 2026

The isolated C++ PDB writer reduced complete 10,000-input job times by **16–18%**, giving **1.19–1.22× throughput**. It passed every raw-prediction fingerprint and PDB-byte comparison for **88,820 matched outputs**, covering both base and pLDDT checkpoints at both precisions. This is an output-formatting change: model weights, inference operations and prediction settings are unchanged. Main and dev have not been modified.

## What changed

The reference constructs NumPy metadata arrays, Python strings and one formatted line per atom. The candidate formats those same fields in C++, releases the Python GIL while building the text, and preserves chain separators, numbering, spacing, rounding and final newlines. It uses a thread-local C numeric locale. Unsupported array types/shapes and non-finite coordinates use the original Python writer. The original function is retained verbatim in `reference/format_pdb.py`; provenance records its source hash.

The CPU-only screen covered **413 fixtures**, including 200 real predictions, empty inputs, signed zero, rounding/large-number cases, unknown residues and fallback behaviour. A further **200 outputs** were checked with eight concurrent writing threads. Median formatting time for 200 real predictions fell from **0.910 to 0.168 seconds (5.40×)**. This microbenchmark measures formatting, not disk writes or full inference.

## Complete-job measurements

Each table value is the median of two fresh-container runs. The five-input startup smoke test is retained separately. Both sides used the same experimental image, with the native writer explicitly off or on; order reversed for the second repetition. Times include container startup/exit, imports, weights, features, inference, coordinate conversion, raw fingerprints and writing every PDB. The fixed seed was 42; optional chemistry was disabled. Bold identifies the lower displayed median, not statistical significance.

The 100 distinct targets are published experimental-test antibodies, 219–242 residues in total. Larger jobs repeat those targets with distinct names and **no prediction caching**; they do not supply additional independent accuracy observations. Length groups, automatic GPU batch policy, two preparation/writing threads and precision settings matched within each comparison.

{table(['Inputs','Checkpoint','Precision','Python writer (s)','Native writer (s)','Throughput ratio','Less elapsed time'],native_times)}

## Peak memory for 10,000 inputs

The higher peak of each matched pair is shown, in GiB. CUDA allocated memory was unchanged. RSS variations of a few tens of MiB do not establish a physical-memory improvement. Process-tree RSS may double-count shared mappings; container accounting is separate. On the Spark's unified-memory system, **do not add these columns** to estimate physical RAM consumption.

{table(['Checkpoint','Precision','Writer','CUDA allocated','Process-tree RSS','Container lifetime'],native_memory)}

## Scope and decision

All 28 matched condition/replicate pairs passed, including four five-input smoke pairs. The full comparison checked masks, coordinates, confidence fingerprints and every PDB byte. These checks demonstrate exactness on the tested workloads; the original writer remains the fallback for inputs outside the native contract. This is the strongest exact-output optimisation from this study and is ready for review. It remains isolated on this component branch; it is **not enabled on dev**. The language checkpoint and optional hydrogenation/minimisation paths were not benchmarked.

The pinned parent is dev `bd8a12ea9f58fa3577215532653a05c47e90e256`, image `sha256:18456b5147ea1ec35d8de4f664962baeea6af1bb473ad9107bb1cbfa37fd844d`. The measured formatter image is `sha256:a53b5613e495565b4ebf71eab5659a5d81b8ecb09378fba6819a3cc82d40b8bf`. Runtime: DGX Spark GB10, NGC PyTorch 2.10.0a0+b558c986e8.nv25.11, CUDA 13.0. Every GPU job used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before CUDA initialisation. TF32-permitted means FP32 tensors with matrix precision `high`/`allow_tf32=True`; strict means `highest`/TF32 matrix multiplication disabled. No BF16 or FP16 inference was used.

The [measurement archive](data/abodybuilder3_pdb_formatting_evidence_261026.zip) contains individual timings, memory, source provenance and all validation outcomes. Raw structures, checkpoints and compiler caches remain local.
'''
    write(args.formatter/'docs/abodybuilder3_pdb_formatting_261026.md',writer_doc)
    evidence(args.formatter/'docs/data/abodybuilder3_pdb_formatting_evidence_261026.zip',{
        'measurements_261026.json':formatter_rows,'resources_261026.json':native_resources,
        'validation_261026.json':native_checks,'completion_261026.json':native_done,
        'provenance_261026.json':read(args.formatter/'docs/provenance_261026.json'),
        'screen_261026.json':read(args.formatter/'docs/formatter_screen_261026.json')})

    # Cache packaging resolves cold code generation, not all fresh-process tracing.
    cache_times=[]
    for n in [100,1000]:
        for model in ['base','plddt']:
            for precision in ['normal','strict']:
                rr=[next(r for r in cached['resources'] if (r['count'],r['model'],r['precision'],r['mode'])==(n,model,precision,mode))
                    for mode in ['eager','compiled','combined']]
                cache_times.append([f'{n:,}',model,PRECISION[precision],*best([r['complete_seconds_median'] for r in rr])])
    cache_doc=f'''# Docker-packaged compilation caches — 26 October 2026

Packaging the prepared PyTorch/Triton caches cut complete compiled jobs from roughly **89–104 seconds to 28–39 seconds**, compared with the earlier empty-cache study. It still did not beat current dev on **100 or 1,000 inputs**. Large jobs subsequently became much faster when combined with the native writer; those separate measurements are in the [overnight results](abodybuilder3_overnight_optimisation_261026.md).

## What was packaged

Eight banks cover base/pLDDT × compilation-only/attention-plus-compilation × TF32-permitted/strict-FP32. Retained cold-study caches seeded the banks. Each bank was then prepared on the GPU for the 100- and 1,000-input workloads, hashed and copied into a Docker layer, together with frozen component and CLI code. Preparation and image building are deployment costs outside the measured fresh jobs. The preparation records and image/cache-manifest identities are retained in the evidence.

Measured compiled containers read their code, input CSVs and caches from the image, with **no host compiler-cache mount**. Only the output directory was mounted, at `/results`, leaving the baked `/output` cache paths visible. The initial eager control failed before inference because its user could not create the default cache directory; it was rerun with a writable `/results` cache. That failed attempt remains in the local audit and is excluded from successful timings.

These are disk/kernel caches, not a serialised whole-model executable. Fresh processes still import libraries, load weights, reconstruct Dynamo graphs/guards and cross existing Rigid/Rotation graph breaks. The first forward can therefore take around 16–20 seconds even when kernel artefacts are available. On-disk cache hits and cache-manifest identities were recorded; process-local `async_compile_cache_miss` counters must not be interpreted as proof that an on-disk kernel was regenerated. Other AOT cache misses can still occur. Caches are specific to the tested software, model, precision and GPU environment.

## Full CLI timings with the original writer

Medians of two fresh containers per cell, including startup and all PDB output. Bold marks the lowest displayed time, not statistical significance. Same 100 distinct experimental targets; 1,000 inputs repeat them ten times. No prediction caching, weight changes or optional chemistry. Normal is TF32-permitted FP32 tensors; strict disables TF32 matrix multiplication. Standard automatic batch caps and two CPU preparation/writing threads were preserved. All matched batch shapes agreed and there were no OOM retries.

{table(['Inputs','Checkpoint','Precision','Current dev (s)','Cached compiler (s)','Cached attention + compiler (s)'],cache_times)}

## Predictions and memory

Both fresh repetitions were scored on the first occurrence of all 100 distinct experimental targets. All 26,400 PDBs were written, and every raw fingerprint/PDB was identical between the two fresh repetitions of each condition in this cached study. This differs from the earlier cold-cache result; it is evidence for this pinned cache bank, not a universal determinism guarantee.

Compilation and attention still change floating-point arithmetic. Across these cached jobs, TF32-permitted candidates changed CDR-H3 experimental backbone RMSD by at most **0.003384 Å** relative to matched dev; a side-chain atom moved by up to **1.984 Å**. Strict-FP32 H3 changes were at most **0.00000839 Å**, with atom displacements up to **0.001382 Å**. Backbone changes were tiny, but atom displacement is not itself experimental error, and large side-chain movements should not be dismissed as uniformly harmless.

Peak CUDA allocation was lower for compiled paths, while process RSS was higher. The full resource tables, reserved CUDA peaks and container accounting are in the evidence and the bulk report. These measures must not be summed on the Spark's unified memory.

## Reproduction and evidence

`tools/run_pipeline_study.py --cache-source ...` prepares and packages banks from a retained cold comparison. `--resume` reuses frozen sources, finished jobs and bank identities after a controller failure. Build first from the pinned dev recipe, then perform the GPU preparation; a conventional GPU-less Docker build alone does not execute the model to discover all workload shapes. The measured runtime was NGC PyTorch 2.10.0a0/CUDA 13.0 on GB10, with the expandable-segments allocator enabled.

The [cache evidence archive](data/abodybuilder3_cached_pipeline_evidence_261026.zip) contains both structural analyses, per-target scores, compact individual measurements, image identities and preparation records. Original source/runtime identities and all local caches/outputs are retained. Main and dev are unchanged; neither result-changing candidate is enabled there.
'''
    write(docs/'abodybuilder3_cached_pipeline_261026.md',cache_doc)
    cache_payload={name:read(runs/'cached_pipeline_261026'/name) for name in [
        'analysis_261026.json','analysis_rep2_261026.json','per_target_scores_261026.json',
        'per_target_scores_rep2_261026.json','cached_images_261026.json','cache_preparation_261026.json']}
    cache_payload['measurements_261026.json']=[compact(r) for r in read(runs/'cached_pipeline_261026/measurements_261026.json')]
    evidence(docs/'data/abodybuilder3_cached_pipeline_evidence_261026.zip',cache_payload)

    # Main comparative table, drawn directly from the same scored resource data.
    timing_rows=[];memory_rows=[];accuracy_rows=[]
    labels=list(LABELS)
    for model in ['base','plddt']:
        for precision in ['normal','strict']:
            rr=[next(r for r in bulk['resources'] if (r['model'],r['precision'],r['label'])==(model,precision,label)) for label in labels]
            times=[r['seconds_median'] for r in rr]
            timing_rows.append([model,PRECISION[precision],*best(times),f'{times[0]/times[-1]:.2f}×'])
            for row in rr:
                memory_rows.append([model,PRECISION[precision],LABELS[row['label']],
                    f"{row['cuda_allocated_peak_gib']:.3f}",f"{row['cuda_reserved_peak_gib']:.3f}",
                    f"{row['process_tree_rss_peak_gib']:.3f}",f"{row['container_memory_peak_gib']:.3f}"])
            for label in labels[2:]:
                cc=[r for r in bulk['comparisons'] if (r['model'],r['precision'],r['label'],r['comparison'])==(model,precision,label,'vs_matched_dev')]
                accuracy_rows.append([model,PRECISION[precision],LABELS[label],
                    f"{statistics.mean(r['h3_mean_rmsd'] for r in cc):.6f}",
                    f"{max(r['h3_max_absolute_delta'] for r in cc):.8f}",
                    f"{max(r['max_raw_ca_displacement'] for r in cc):.6f}",
                    f"{max(r['largest_atom']['displacement'] for r in cc):.6f}",
                    f"{max(r['ca_lddt_max_absolute_delta_percentage_points'] for r in cc):.6f}"])
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
    colours=['#546477','#009e73','#2c7bb6','#a65eae']
    for ax,(model,precision) in zip(axes.flat,[(m,p) for m in ['base','plddt'] for p in ['normal','strict']]):
        rr=[next(r for r in bulk['resources'] if (r['model'],r['precision'],r['label'])==(model,precision,label)) for label in labels]
        values=[r['seconds_median'] for r in rr]
        ax.bar(range(4),values,color=colours,width=.68)
        for i,(r,v) in enumerate(zip(rr,values)):
            lo,hi=min(r['seconds_runs']),max(r['seconds_runs'])
            ax.errorbar(i,v,yerr=[[v-lo],[hi-v]],fmt='none',ecolor='black',capsize=4)
            ax.text(i,v+7,f'{v:.1f}s',ha='center',fontsize=11)
        ax.set_xticks(range(4),['Current\ndev','Native\nwriter','Cached\ncompiler','Cached compiler\n+ attention'])
        ax.set_title(f'{model} · {PRECISION[precision]}',fontsize=12)
        ax.set_ylim(0,292);ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Complete job time (seconds)')
    fig.suptitle('ABodyBuilder3: 10,000-input jobs, including startup and PDB output',fontsize=14)
    fig.text(.5,.01,'100 distinct experimental antibodies repeated 100 times · medians and ranges of two fresh jobs\nCached compiler conditions also use the native writer; model weights and learned updates are unchanged.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.94])
    assets=docs/'assets';assets.mkdir(exist_ok=True)
    for suffix in ['png','svg']:
        target=assets/f'abodybuilder3_overnight_optimisation_261026.{suffix}'
        fig.savefig(target,dpi=170,metadata={'Date':None} if suffix=='svg' else None)
        if suffix=='svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)
    caprows=[];workerrows=[]
    for policy_name in ['batch16_workers2','batch32_workers2','batch64_workers2','batch128_workers2']:
        rr=[r for r in policy['resources'] if r['policy']==policy_name]
        caprows.append([policy_name.split('_')[0].replace('batch',''),
                       f"{statistics.mean(r['seconds_median'] for r in rr):.2f}",
                       f"{max(r['cuda_allocated_peak_gib'] for r in rr):.3f}",
                       f"{max(r['max_ca_displacement_vs_batch32'] for r in policy['comparisons'] if r['policy']==policy_name):.6f}"])
    for workers in [1,2,4,8]:
        rr=[r for r in policy['resources'] if r['policy']==f'batch16_workers{workers}']
        workerrows.append([workers,f"{statistics.mean(r['seconds_median'] for r in rr):.2f}"])
    crossover = docs/'abodybuilder3_job_size_crossover_261026.md'
    job_size_status = ('The subsequent [64-job size calibration](abodybuilder3_job_size_crossover_261026.md) found eager inference faster at 1,000 inputs and cached execution faster at 1,500, 2,000 and 5,000 inputs, in both precision profiles and checkpoints. That brackets the crossover for this length range; wider workloads still need qualification before an automatic selector is deployed.' if crossover.exists() else
                       'The exact crossover with the native writer has **not** been measured; an automatic job-size selector needs intermediate matched measurements before choosing a threshold.')
    overall=f'''# Batching optimisation results — 26 October 2026

**Two useful gains emerged.** Replacing Python PDB formatting with an exact-output native writer gives **1.19–1.22× throughput** on 10,000-input jobs. Combining that writer with Docker-packaged compilation gives approximately **2.9–3.0× throughput**, including fresh startup and writing every PDB. Compilation changes floating-point results, so it remains a candidate for review. **Main and dev are unchanged; no new option is enabled there.**

## Full-pipeline results for 10,000 inputs

Each value is the median of two fresh-container jobs; the evidence retains both times. The four approaches used matched input order and GPU batch shapes. All times include imports, weight loading, CPU preparation, inference, coordinate conversion, common raw fingerprints, PDB output and container exit. The compiler conditions also use the native writer. Bold marks the lowest displayed median within each row, not significance.

{table(['Checkpoint','Precision','Current dev (s)','Native writer (s)','Cached compiler + writer (s)','Cached attention + compiler + writer (s)','Combined vs dev'],timing_rows)}

![Complete 10,000-input job times](assets/abodybuilder3_overnight_optimisation_261026.png)

These throughput jobs repeat **100 distinct published experimental antibodies** 100 times, using unique output names and **no prediction cache**. They are not 10,000 independent antibodies or a broader accuracy test. Both released base/pLDDT checkpoints, seed 42 and eight learned structure updates were preserved. Optional chemistry and the language checkpoint are outside the comparison. Two preparation and two writer threads overlap one model-owning GPU process; the automatic GPU batch cap is 32 here. Earlier tests use 16 for 100 inputs. Every job completed without OOM retries.

## Precision is an explicit choice

The two tested modes both store weights and tensors in FP32. **TF32 permitted** is the existing NGC/dev setting: `float32_matmul_precision='high'`, CUDA matrix `allow_tf32=True`. **Strict FP32** uses `highest` and disables TF32 matrix multiplication before inference; compiler fusion/reassociation can still change the final bits. These experiments did not use BF16, FP16 or autocast. No precision is changed automatically.

Strict-FP32 cached compilation is only a few percent slower here than TF32-permitted compilation, while its numerical changes are much smaller. Attention adds around 1–2% to normal-precision bulk throughput; its strict-FP32 benefit is inconsistent between checkpoints. Two repetitions do not establish that these small differences are repeatable. A sensible review candidate is **strict FP32 + cached compiler** for close arithmetic agreement, with **TF32 + cached compiler/attention** as an explicitly selected faster mode. This is a proposal, not a change to the default.

## Prediction differences and experimental accuracy

The native writer alone passed every fingerprint and PDB-byte comparison in all **88,820** matched predictions of its original screen, plus **80,000** matched predictions in this bulk comparison. This is separate from the compiler's numerical differences. All 16 bulk condition pairs also reproduced every raw prediction fingerprint across their two fresh runs.

For scientific accuracy, the **first occurrence of each of the 100 distinct experimental targets** was scored in **both** fresh repetitions. Masks and experimental backbone coordinates came from the retained published test dataset. CDR-H3 RMSD follows alignment on the heavy-chain framework Cα atoms; Cα-lDDT measures local distances on all matched Cα atoms. Both frameworks, all six CDRs and confidence changes are retained in the per-target evidence. Distances are Å; lDDT changes are percentage points.

The table shows the larger deviation across the two fresh runs, always against current dev at the **same precision**. H3 mean is averaged across the two process replicates, not treated as 200 independent antibodies. Smaller accuracy changes do not establish a scientifically superior predictor.

{table(['Checkpoint','Precision','Candidate','Mean H3 RMSD to experiment','Largest H3 RMSD change','Largest raw Cα displacement','Largest raw atom displacement','Largest Cα-lDDT change (pp)'],accuracy_rows)}

At TF32-permitted precision, compiler-related H3 accuracy changes stayed below **0.0034 Å** in this bulk test. Individual side-chain atoms still moved by up to **1.403 Å versus dev**, or **1.560 Å between attention-plus-compiler and compiler alone**. Such movements are not backbone accuracy losses, and atom displacement alone cannot tell whether a side chain became more accurate. Strict-FP32 changes were much smaller: H3 error changes below **0.000008 Å**. Accuracy gains/losses varied by target; neither arithmetic approach is uniformly closer to experiment.

Comparing TF32-permitted to strict-FP32 arithmetic within each approach changed H3 experimental error by at most **0.0080 Å** on this set. These small changes support an explicit speed/precision choice; they do not qualify BF16/FP16 or establish behaviour for all antibodies. Detailed pairwise, local and confidence results are in the evidence. Result-changing options remain subject to user review before dev adoption.

## Memory at 10,000 inputs

Each entry is the higher peak across two jobs, in GiB. Compiler paths lower CUDA allocated memory from **3.99 to about 2.26–2.27 GiB**, but increase sampled process-tree RSS from **about 1.85 to 4.64–4.67 GiB**. They therefore do not demonstrate a universal system-memory saving. RSS can double-count shared compiler-worker mappings; container lifetime memory is a different accounting measure. On unified memory, **do not add these columns** to estimate physical usage.

{table(['Checkpoint','Precision','Approach','CUDA allocated','CUDA reserved','Process-tree RSS','Container lifetime'],memory_rows)}

## Other completed experiments

**Docker caches and job size.** The [packaged-cache comparison](abodybuilder3_cached_pipeline_261026.md) completed 48 fresh jobs with the original writer. At 100 inputs, dev took around 7 seconds and cached compilation 28 seconds. At 1,000 inputs, dev took around 29 seconds and cached compilation 38 seconds. Cache packaging removes much of cold kernel code generation, but fresh-process tracing/guards still cost time. The 10,000-input bulk results show that this cost can be amortised. {job_size_status} The number of antibodies in the job and the GPU microbatch size are different decisions.

**CUDA graph replay.** Small and large graph-only jobs and graph-plus-native-writer jobs completed. All **177,600** matched graph predictions and PDBs were exact. Graph replay added startup and private-buffer costs; it did not give a dependable full-pipeline gain. At 1,000 inputs it took around 53 seconds versus 29 seconds without the writer change; with the native writer, around 51 versus 26 seconds. At 10,000 inputs, graphs plus the writer took around 248–251 seconds versus 213–216 seconds for the writer alone. Graph-only CUDA peaks rose to about 5.15 GiB. This implementation is not recommended for adoption.

**GPU batch cap.** A 1,000-input screen used caps 16, 32, 64 and 128 with two CPU preparation/writer threads. The memory-estimate budget was 24 GiB to permit the largest cap; this was an explicitly labelled policy screen. The time column below averages the four model/precision medians, so it is a compact screen summary, not one job. Memory is the largest peak across those conditions. The small cap-16 timing advantage needs confirmation; it is not a demonstrated universal speed gain. Changing batch shape can alter floating-point output: cap comparisons reached about 0.0184 Å Cα displacement and 0.00384 Å H3 accuracy change versus cap 32. No policy was changed on dev.

{table(['GPU batch cap','Mean complete time (s)','Largest CUDA allocated peak (GiB)','Largest Cα displacement vs cap32 (Å)'],caprows)}

**CPU workers.** At cap 16, more than two preparation/writer threads did not yield a dependable gain; one was slower. All **24,000** predictions and PDBs from the worker-only variations were exact versus the two-thread control with the same batches. The summary averages model/precision medians.

{table(['Preparation and writer threads each','Mean complete time (s)'],workerrows)}

## Recommended next step

Review the exact-output writer for adoption first. Keep explicit strict-FP32 and TF32-permitted modes, optimise each separately, and qualify any automatic eager/compiled selector across the intended job sizes and wider length distributions. Keep the existing batch/worker policy while that qualification is incomplete. The language checkpoint, optional preparation stages, broad antibody-length coverage and cross-hardware cache portability need their own checks before claiming universal support.

The isolated [PDB formatter branch]({URL}/tree/components/pdb-formatting-261026) contains the original and C++ candidate, regression tests and [writer report]({URL}/blob/components/pdb-formatting-261026/docs/abodybuilder3_pdb_formatting_261026.md). The [overnight evidence archive](data/abodybuilder3_overnight_optimisation_evidence_261026.zip) contains per-job timing/memory, both-replicate per-target accuracy, repeatability, exactness, cache identities and policy results. Raw predictions, model weights, cache payloads and all Docker images remain local and intact.

Main is still `a5deeb1b73994491f94b5724d2f2baa9993d47a6`; dev is still `bd8a12ea9f58fa3577215532653a05c47e90e256`. The pinned parent runtime is image `sha256:18456b5147ea1ec35d8de4f664962baeea6af1bb473ad9107bb1cbfa37fd844d`, NGC PyTorch 2.10.0a0+b558c986e8.nv25.11/CUDA 13.0 on DGX Spark GB10. Every GPU job used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Sources, input identities, commands and cache preparation records were frozen for each study; compact evidence omits bulky per-forward logs, which remain retained locally.
'''
    write(docs/'abodybuilder3_overnight_optimisation_261026.md',overall)
    payload={'bulk_analysis_261026.json':bulk,
        'bulk_per_target_261026.json':read(runs/'cached_native_bulk_261026/bulk_per_target_261026.json'),
        'bulk_experimental_accuracy_261026.json':read(runs/'cached_native_bulk_261026/bulk_experimental_accuracy_261026.json'),
        'bulk_measurements_261026.json':[compact(r) for r in read(runs/'cached_native_bulk_261026/measurements_261026.json')],
        'bulk_images_261026.json':read(runs/'cached_native_bulk_261026/images_261026.json'),
        'bulk_protocol_261026.json':read(runs/'cached_native_bulk_261026/protocol_261026.json'),
        'batch_policy_analysis_261026.json':policy}
    for study in ['graph_pipeline_small_261026','graph_pipeline_large_261026','graph_native_pipeline_261026']:
        payload[study+'/measurements_261026.json']=[compact(r) for r in read(runs/study/'measurements_261026.json')]
        payload[study+'/exact_output_validation_261026.json']=read(runs/study/'exact_output_validation_261026.json')
    evidence(docs/'data/abodybuilder3_overnight_optimisation_evidence_261026.zip',payload)
    # A small manifest makes report/evidence changes independently verifiable.
    for destination in [docs,args.formatter/'docs']:
        paths=sorted([p for p in destination.rglob('*261026*') if p.is_file() and p.suffix in ['.md','.json','.zip','.png','.svg']])
        write(destination/'overnight_SHA256SUMS_261026.txt','\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(destination)) for p in paths))
    print('Rendered formatter, cached-pipeline and overnight reports with matching evidence.',flush=True)


if __name__=='__main__':
    main()
