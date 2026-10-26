"""Publish the completed eager/compiled job-size calibration, without deployment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from publish_overnight_results import read, write, table, best, compact, evidence, PRECISION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];runs=args.project/'runs';docs=repo/'docs'
    studies=[runs/'cached_native_crossover_261026',runs/'cached_native_crossover_boundary_261026']
    payload={};resources=[];analyses=[];jobs=0;predictions=0
    for study in studies:
        records=read(study/'measurements_261026.json')
        complete=read(study/'completed_261026.json')
        assert complete['completed'] and complete['jobs']==len(records)
        jobs+=len(records);predictions+=sum(r['count'] for r in records)
        payload[study.name+'/measurements_261026.json']=[compact(r) for r in records]
        payload[study.name+'/protocol_261026.json']=read(study/'protocol_261026.json')
        for path in sorted(study.glob('size_n*_analysis_261026.json')):
            count=int(path.name.split('_')[1][1:]);d=read(path)
            assert all(r['raw_exact']==r['count'] for r in d['repeatability'])
            analyses.append(d)
            resources.extend(dict(r,count=count) for r in d['resources'])
            payload[study.name+'/'+path.name]=d
            for suffix in ['per_target','experimental_accuracy']:
                name=f'size_n{count}_{suffix}_261026.json'
                payload[study.name+'/'+name]=read(study/name)
    assert jobs==64 and predictions==152000
    bulk=read(runs/'cached_native_bulk_261026/bulk_analysis_261026.json')
    resources.extend(dict(r,count=10000) for r in bulk['resources'] if r['label']=='native_writer' or
        (r['precision']=='normal' and r['label']=='cached_combined_native') or
        (r['precision']=='strict' and r['label']=='cached_compiled_native'))
    rows=[]
    for count in [1000,1500,2000,5000,10000]:
        for model in ['base','plddt']:
            values=[]
            for precision in ['normal','strict']:
                label='cached_combined_native' if precision=='normal' else 'cached_compiled_native'
                a=next(r for r in resources if (r['count'],r['model'],r['precision'],r['label'])==(count,model,precision,'native_writer'))
                b=next(r for r in resources if (r['count'],r['model'],r['precision'],r['label'])==(count,model,precision,label))
                values.extend([*best([a['seconds_median'],b['seconds_median']]),f"{a['seconds_median']/b['seconds_median']:.2f}×"])
            rows.append([f'{count:,}',model,*values])
    accuracy=[]
    for precision in ['normal','strict']:
        rr=[r for d in analyses for r in d['comparisons'] if r['precision']==precision and r['comparison']=='vs_matched_reference']
        accuracy.append([PRECISION[precision],f"{max(r['h3_max_absolute_delta'] for r in rr):.8f}",
            f"{max(r['max_raw_ca_displacement'] for r in rr):.6f}",
            f"{max(r['largest_atom']['displacement'] for r in rr):.6f}",
            f"{max(r['ca_lddt_max_absolute_delta_percentage_points'] for r in rr):.6f}"])
    doc=f'''# Choosing eager or compiled inference by job size — 26 October 2026

For the tested 219–242-residue antibodies, **eager inference was faster at 1,000 inputs; cached compilation was faster at 1,500, 2,000, 5,000 and 10,000 inputs**. Both strict-FP32 and TF32-permitted speed profiles showed this pattern on both checkpoints. This supports making the execution decision once at startup, while keeping precision an explicit user choice. Main and dev are unchanged; no automatic selector has been deployed.

![Job-size crossover in both precision profiles and checkpoints](assets/abodybuilder3_job_size_crossover_261026.png)

Above 1, cached execution is faster; below 1, eager inference is faster. Points are ratios of the two-job median times. Whiskers show the range of the two paired job ratios, **not confidence intervals**. The lines guide the eye rather than fit a cost model; [both individual timing pairs](data/abodybuilder3_crossover_plot_data_261026.csv) are available.

## Matched complete-job times

All approaches use the exact-output native PDB writer. The **TF32-permitted profile** uses cached compilation plus point-matmul attention. The **strict-FP32 profile** uses cached compilation alone: its bulk speed was similar without introducing another arithmetic replacement. These profiles were selected for calibration after the preceding bulk study, not as new models or an independent accuracy test. References use the same precision as their candidate.

Times are medians of two fresh-container commands including startup, imports, weights, all inference, atom conversion and every PDB write. Each row compares the same input order, batch shapes, two CPU preparation/writer threads and no OOM retries. Bold marks the lower displayed time within each precision pair; it does not imply significance. The ratio is eager time divided by cached time, so **above 1 favours compilation**.

{table(['Inputs','Checkpoint','Eager TF32 (s)','Cached TF32 profile (s)','Ratio','Eager strict FP32 (s)','Cached strict FP32 profile (s)','Ratio'],rows)}

The new calibration comprises **64 fresh jobs and 152,000 written PDBs** at 1,000/1,500/2,000/5,000 inputs; 10,000-input rows reuse the prior matched bulk measurements. The 1,500-input boundary was tested after the first screen bracketed the crossover between 1,000 and 2,000. Each workload repeats the same **100 distinct published experimental-test antibodies**, with unique output names and no prediction caching. This is a job-size calibration, not extra independent biological evidence.

## What a startup decision should consider

The total job size determines whether startup costs can be amortised. The **GPU microbatch cap** is a separate setting; it stayed at 32 throughout these jobs. The measured crossover lies between the 1,000- and 1,500-input points for this particular workload and cache bank. Compiled time need not rise monotonically with job size: partial batches, graph/guard reconstruction and available cached shapes affect startup.

Within this measured scope, choosing eager for the small jobs and cached execution from 1,500 inputs would select the faster measured route. The exact boundary inside that interval has not been established. A production selector should also consider checkpoint, requested precision, antibody-length distribution, memory budget, compatible cache availability and runtime/GPU identity. It should expose the selected route and an explicit override. A different precision should never be selected silently. Unknown length/cache regimes require further qualification rather than treating this one crossover as universal.

## Prediction and accuracy checks

All jobs wrote the expected number of PDBs; every native-writer call was counted. Batch shapes matched between eager and compiled conditions. Every raw fingerprint reproduced exactly across the two fresh processes for all 32 matched condition pairs. This repeatability result applies to these prepared caches and workload shapes; it does not promise bit-identical results on other software or hardware.

Both repetitions were scored against the retained experimental structures using the first occurrence of each of the 100 distinct targets. Distances below are Å, relative to eager inference at the same precision. H3 error is backbone RMSD after heavy-chain framework Cα alignment; Cα-lDDT changes are percentage points. The maxima cover both checkpoints and all four new job sizes, without treating repeated inputs as independent antibodies.

{table(['Profile','Largest H3 error change','Largest raw Cα displacement','Largest raw atom displacement','Largest Cα-lDDT change (pp)'],accuracy)}

The changes are consistent with the [bulk accuracy results](abodybuilder3_overnight_optimisation_261026.md): strict-FP32 changes are much smaller, while TF32-permitted results can move individual side-chain atoms more visibly despite tiny backbone-error differences. Neither profile is enabled on dev. The earlier matched all-PDB writer exactness checks remain separate from these deliberate compiler arithmetic changes.

Peak CUDA allocated/reserved memory, process-tree RSS, container memory, all individual times and per-target structural/confidence scores are in the [calibration evidence](data/abodybuilder3_job_size_crossover_evidence_261026.zip) and [resource CSV](data/abodybuilder3_job_size_resources_261026.csv). CUDA allocation remains around 4.0 GiB for eager and 2.26–2.27 GiB for compiled execution, with higher CPU RSS for compiled paths. These quantities must not be added on the Spark's unified-memory system.

## Reproduction

`tools/run_cached_formatter_study.py` now accepts `--reuse-images`, `--counts`, `--speed-profiles` and `--skip-probes`. The calibration reused the exact eight Docker image identities from the completed bulk study, with no rebuilding or changes to the baked kernel banks. New input CSVs are prefixes of the retained 10,000-row workload and are mounted read-only; model and component code remain in the prepared images. Existing passing 100-input qualification was reused, not rerun. `tools/analyse_bulk_pipeline.py --count ... --reference-label native_writer --expected-jobs 16 --output-prefix ...` scores each job size with the same structural metrics.

The pinned runtime, checkpoints, allocator and preserved main/dev commits are documented in the [overall results](abodybuilder3_overnight_optimisation_261026.md). All images, caches, raw outputs and frozen launcher/analysis identities remain retained locally.
'''
    write(docs/'abodybuilder3_job_size_crossover_261026.md',doc)
    csvpath=docs/'data/abodybuilder3_job_size_resources_261026.csv'
    columns=['count','model','precision','label','seconds_median','cuda_allocated_peak_gib',
             'cuda_reserved_peak_gib','process_tree_rss_peak_gib','container_memory_peak_gib']
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore',lineterminator='\n')
        writer.writeheader();writer.writerows(sorted(resources,key=lambda r:(r['count'],r['model'],r['precision'],r['label'])))
    evidence(docs/'data/abodybuilder3_job_size_crossover_evidence_261026.zip',payload)
    names=['abodybuilder3_job_size_crossover_261026.md','data/abodybuilder3_job_size_resources_261026.csv',
           'data/abodybuilder3_job_size_crossover_evidence_261026.zip']
    write(docs/'job_size_SHA256SUMS_261026.txt','\n'.join(hashlib.sha256((docs/name).read_bytes()).hexdigest()+'  '+name for name in names))
    print('Published 64-job crossover calibration and paired evidence.',flush=True)


if __name__=='__main__':main()
