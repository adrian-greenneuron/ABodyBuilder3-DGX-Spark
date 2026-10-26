"""Package the full-pipeline comparison, including fresh-compile variability."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--docs',type=Path,required=True)
    args=p.parse_args();root=args.study;docs=args.docs
    data=docs/'data/pipeline_compilation_261026';data.mkdir(parents=True,exist_ok=True)
    assets=docs/'assets';assets.mkdir(exist_ok=True)
    first=json.loads((root/'analysis_261026.json').read_text())
    second=json.loads((root/'analysis_rep2_261026.json').read_text())
    measurements=json.loads((root/'measurements_251026.json').read_text())
    resources={(r['count'],r['model'],r['precision'],r['mode']):r for r in first['resources']}
    accuracy=first['accuracy']+second['accuracy']
    scores=[]
    for name in ['per_target_scores_261026.json','per_target_scores_rep2_261026.json']:
        scores+=json.loads((root/name).read_text())
    protocol=json.loads((root/'protocol_251026.json').read_text())
    summary={'resources':first['resources'],'accuracy':accuracy,'repeatability':first['repeatability'],
             'validation':first['validation'],'scope':first['scope'],
             'protocol':protocol,'measurements':measurements}
    (data/'pipeline_summary_261026.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (data/'pipeline_measurements_261026.csv').open('w',newline='') as f:
        fields=['count','model','precision','mode','replicate','container_command_seconds',
                'cuda_allocated_peak_gib','sampled_process_tree_rss_peak_gib','container_lifetime_peak_gib']
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in measurements:
            writer.writerow({**{k:r[k] for k in fields[:6]},
                'cuda_allocated_peak_gib':r['metrics']['peak_cuda_allocated_bytes']/2**30,
                'sampled_process_tree_rss_peak_gib':r['metrics']['sampled_process_tree_rss_peak_bytes']/2**30,
                'container_lifetime_peak_gib':r['metrics']['container_lifetime_memory_peak_bytes']/2**30})
    with (data/'pipeline_accuracy_changes_261026.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['count','model','precision','mode','replicate','target','region',
                         'experimental_backbone_rmsd','change_vs_matched_eager','change_vs_matched_compiler'])
        for r in scores:
            for region, value in r['accuracy']['regions'].items():
                actual=value['framework_aligned_backbone_rmsd']
                baseline=r['eager_accuracy']['regions'][region]['framework_aligned_backbone_rmsd']
                compiled=r.get('compiler_accuracy',{}).get('regions',{}).get(region,{}).get('framework_aligned_backbone_rmsd')
                writer.writerow([r['count'],r['model'],r['precision'],r['mode'],r.get('replicate',1),r['target'],region,
                                 actual,actual-baseline,actual-compiled if compiled is not None else ''])
    # The figure uses complete command time, including startup, with linear axes.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained',sharex=True)
    colours={'eager':'#24679b','compiled':'#d58620','combined':'#8657a5'}
    labels=['Base · normal','Base · strict FP32','pLDDT · normal','pLDDT · strict FP32']
    for ax,count in zip(axes,[100,1000]):
        for j,mode in enumerate(['eager','compiled','combined']):
            selected=[resources[count,model,precision,mode] for model in ['base','plddt'] for precision in ['normal','strict']]
            centres=np.arange(4)+(j-1)*.23
            values=[r['complete_seconds_median'] for r in selected]
            errors=np.array([[r['complete_seconds_median']-r['complete_seconds_range'][0],
                              r['complete_seconds_range'][1]-r['complete_seconds_median']] for r in selected]).T
            ax.barh(centres,values,height=.21,color=colours[mode],xerr=errors,
                    label={'eager':'Uncompiled','compiled':'Compilation alone','combined':'Attention + compilation'}[mode])
            for y,v in zip(centres,values):ax.text(v+1,y,f'{v:.1f} s',va='center',fontsize=9)
        ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(0,119)
        ax.set_title(f'{count:,} inputs',fontsize=13);ax.set_xlabel('Complete job time (seconds)')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=10)
    fig.suptitle('Fresh compilation outweighs warmed-model gains\nTwo fresh containers per condition · every PDB written · empty compiler caches',fontsize=14)
    figure='abodybuilder3_pipeline_compilation_261026'
    fig.savefig(assets/(figure+'.png'),dpi=180)
    fig.savefig(assets/(figure+'.svg'));plt.close(fig)
    svg=assets/(figure+'.svg');svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    lines=['# Complete batching pipeline: attention and compilation — 26 October 2026','',
        '**Current dev remains faster for these fresh jobs.** Including imports, model loading, compilation, features, atom conversion and output writing, both compiled approaches took about **89–104 seconds**, compared with **7–8 seconds for 100 inputs** and **29 seconds for 1,000 inputs** without compilation. Adding attention did not give a dependable complete-job gain over compilation alone. Main and dev are unchanged, and neither experimental option is enabled.','',
        '## Protocol','',
        'The same 100 published experimental-test antibodies were used. The 1,000-input workload cycles them ten times, with distinct output names and **no prediction cache**; it is not 1,000 independent antibodies. Both base and pLDDT checkpoints ran uncompiled, compilation-only and attention-plus-compilation conditions, at normal and strict FP32 precision. Each condition had **two independent fresh-container repetitions**, giving **48 jobs and 26,400 PDBs**. Condition order was reversed between repetitions and checkpoints.','',
        'The prediction CLI, bounded CPU feature queues, length-aware scheduler, automatic batch caps, atom conversion and PDB writer were frozen from dev `bd8a12e`. Batch shapes matched across conditions and no OOM retries were needed. The automatic cap was 16 for 100 inputs and 32 for 1,000 inputs; two preparation and two writing threads overlapped one GPU-owning model process. Optional hydrogenation/relaxation and the language checkpoint were outside this comparison.','',
        'The retained optimised image was `sha256:18456b5147ea1ec35d8de4f664962baeea6af1bb473ad9107bb1cbfa37fd844d`: NGC PyTorch 2.10.0a0, CUDA 13.0, NVIDIA GB10. Every job was offline and used the expandable-segments allocator. No image was rebuilt and no model weights, learned updates or chemistry dependencies changed. Normal precision retained `high`/TF32-permitted settings; strict used `highest` with TF32 disabled before model loading.','',
        'Unlike the earlier fixed-shape model screen, this experiment used **`torch.compile(mode="default", dynamic=True)`** for the unchanged scheduler’s varying shapes. Each job started with fresh empty compiler/Triton cache directories. These are cold-compiler results, not a reused-cache or persistent-server benchmark. The experimental adapter was confined to this isolated component branch and local frozen runs.','',
        '## Complete-job speed','',
        'Times are medians of two fresh container commands, in seconds. They include startup, every PDB write, common prediction fingerprints and final raw-evidence writing. Evidence writing took only **0.014–0.019 seconds per job**. Bold marks the lowest recorded median in each matched row, not statistical significance. Individual runs and ranges are retained in the data.','',
        '| Inputs | Checkpoint | Precision | Uncompiled | Compilation alone | Attention + compilation |',
        '|---:|---|---|---:|---:|---:|']
    for count in [100,1000]:
        for model in ['base','plddt']:
            for precision in ['normal','strict']:
                values=[resources[count,model,precision,mode]['complete_seconds_median'] for mode in ['eager','compiled','combined']]
                cells=[f'**{v:.2f}**' if v==min(values) else f'{v:.2f}' for v in values]
                lines.append(f'| {count:,} | {model} | {precision} | '+' | '.join(cells)+' |')
    lines+=['',f'![Complete job timings](assets/{figure}.png)','',
        'Compilation dominates these job sizes. In one 1,000-input base combined run, the first forward took about **54.6 seconds** on the host; a later one-item partial batch took **21.0 seconds**, whereas ordinary already-compiled forwards took fractions of a second. These forward labels include compiler activity and are not standalone GPU-kernel timings.','',
        'Dynamo recorded multiple graphs and graph breaks around the existing Rigid/Rotation helper objects. This is **partial compilation with eager regions**, not one fully captured model graph. No cache-limit fallback or OOM fallback occurred. We did not rewrite those helpers, change batch policy, test a warmed disk cache or measure a long-lived prediction service. The earlier 4–5% warmed-model gain therefore should not be presented as a complete-pipeline speedup.','',
        '## Peak memory for 1,000-input jobs','',
        'Values are the higher peak across the two runs, in GiB; bold marks the lowest peak within each checkpoint/precision group. CUDA allocated peaks cover the whole process, including startup. RSS is sampled process-tree RSS and can count shared pages in compiler workers more than once. Container accounting is a separate lifetime measure. **Do not add these columns to estimate physical unified memory.**','',
        '| Checkpoint | Precision | Approach | Peak CUDA allocated | Sampled process-tree RSS | Container lifetime peak |',
        '|---|---|---|---:|---:|---:|']
    for model in ['base','plddt']:
        for precision in ['normal','strict']:
            for mode in ['eager','compiled','combined']:
                r=resources[1000,model,precision,mode]
                cells=[]
                for field in ['cuda_allocated_peak_gib','rss_peak_gib','container_peak_gib']:
                    best=min(resources[1000,model,precision,v][field] for v in ['eager','compiled','combined'])
                    value=f'{r[field]:.3f}'
                    cells.append(f'**{value}**' if round(r[field],3)==round(best,3) else value)
                lines.append(f'| {model} | {precision} | {mode} | '+' | '.join(cells)+' |')
    lines+=['','Compilation reduced peak PyTorch CUDA allocation from about **3.99 GiB to 2.26–2.27 GiB**, but compiler-process RSS increased from about **1.8 GiB to 4.5–5.8 GiB**. This does not establish a physical system-memory saving. All 100-input measurements and reserved-memory peaks are included in the machine-readable results.','',
        '## Prediction differences and experimental accuracy','',
        'Raw coordinates, masks and confidence fingerprints were captured for every output. Quantitative geometry and accuracy were scored for the **first occurrence of each of the 100 distinct targets in every job**, including both fresh repetitions: 4,800 saved raw predictions. The ten presentations in a 1,000-input job are not treated as independent biological samples. All six CDRs, both frameworks, Cα-lDDT, confidence and the established geometric screens were retained.','',
        'The table reports the largest absolute change in framework-aligned CDR-H3 backbone RMSD to experiment, and the largest raw atom displacement, across both workload sizes and both fresh repetitions. References use the **same precision**. Tiny backbone differences are not evidence of improved biological accuracy.','',
        '| Checkpoint | Precision | Maximum H3 accuracy change vs uncompiled | Maximum H3 accuracy change vs compiler only | Maximum atom displacement vs uncompiled | Maximum atom displacement vs compiler only |',
        '|---|---|---:|---:|---:|---:|']
    for model in ['base','plddt']:
        for precision in ['normal','strict']:
            v=[r for r in accuracy if r['model']==model and r['precision']==precision and r['mode']=='combined']
            lines.append(f"| {model} | {precision} | {max(r['h3_max_accuracy_change_vs_eager'] for r in v):.8f} Å | {max(r['h3_max_accuracy_change_vs_compiler'] for r in v):.8f} Å | {max(r['largest_atom_vs_eager']['displacement'] for r in v):.6f} Å | {max(r['max_raw_atom_displacement_vs_compiler'] for r in v):.6f} Å |")
    lines+=['','All combined H3 backbone accuracy changes were below **0.01 Å** on this test set, without a consistent winner across antibodies. Local atoms can still differ substantially: normal-precision combined predictions moved a side-chain atom by **1.219 Å versus uncompiled dev**, and the combined/compiler-only difference reached **2.358 Å**. These are atom displacements, not changes in experimental RMSD; they should not be equated with accuracy loss or dismissed as universally harmless. Strict FP32 differences were much smaller, but not zero.','',
        '## Fresh-process repeatability','',
        '**All uncompiled repeats matched exactly** in raw fingerprints and PDB bytes at both precisions. Fresh compiled repeats were not consistently bit-identical, even with strict FP32. The first-occurrence comparisons found up to **0.917 Å** atom displacement between normal combined runs and **0.000551 Å** between strict combined runs. These bounds describe the quantitatively retained 100 targets per job; fingerprints alone do not quantify the other presentations in a repeated workload.','',
        'A fixed seed and strict FP32 therefore do not establish deterministic fresh compilation. The source of this fresh-compile variability was not isolated in this study; different arithmetic/kernel choices are a hypothesis, not a measured attribution. The previous same-process fixed-shape repeatability checks remain valid within their narrower scope. Exact per-job fingerprint/PDB repeat counts are retained in the summary.','',
        '## Decision and limits','',
        '**Keep current dev unchanged.** Neither precision of attention-plus-compilation improves complete fresh-job latency here, and normal precision adds noticeable local atom changes. Strict precision limits numerical drift but does not remove compilation overhead or guarantee bit-identical fresh-process results.','',
        'A persistent model worker, reuse of validated compilation caches, and fewer special partial-batch shapes are possible future experiments. They could amortise compilation and must be measured separately before claiming a production throughput benefit. They were not run or enabled as part of this comparison. The result covers one Spark, two checkpoints and these workloads; it is not a claim that compilation is always slower.','',
        '## Evidence and reproduction','',
        '- [All 48 timings, memory peaks, compiler counters and accuracy summaries](data/pipeline_compilation_261026/pipeline_summary_261026.json)',
        '- [Per-job timing and memory CSV](data/pipeline_compilation_261026/pipeline_measurements_261026.csv)',
        '- [Per-antibody accuracy changes, both replicates](data/pipeline_compilation_261026/pipeline_accuracy_changes_261026.csv)',
        '- [Full comparison archive](data/pipeline_compilation_261026/abodybuilder3_pipeline_comparisons_261026.zip)',
        '- [Checksums](data/pipeline_compilation_261026/abodybuilder3_pipeline_checksums_261026.txt)','',
        'The archive contains both scoring passes, per-target scores, compact raw-output fingerprints, protocol, source hashes and all exact container commands. Raw NPZ/PDB predictions, compiler caches and verbose logs remain local. The second scoring pass was added after observing fresh-compile differences; no GPU job was rerun or replaced.','',
        '`tools/run_pipeline_study.py` freezes the dev CLI and component source and queues the finite comparison; `tools/measure_pipeline.py` is its measurement-only adapter. `tools/analyse_pipeline_study.py --replicate 1` and `--replicate 2` use the established structural scorer. `tools/write_pipeline_report.py` generates this report and evidence package. The retained image includes the required SciPy environment; no host dependencies were installed.','',
        'Main remains `a5deeb1b73994491f94b5724d2f2baa9993d47a6`; dev remains `bd8a12ea9f58fa3577215532653a05c47e90e256`. All experimental changes are confined to the component branch and frozen local study.','']
    (docs/'abodybuilder3_pipeline_compilation_261026.md').write_text('\n'.join(lines))
    archive=data/'abodybuilder3_pipeline_comparisons_261026.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in ['analysis_261026.json','analysis_rep2_261026.json','per_target_scores_261026.json',
                     'per_target_scores_rep2_261026.json','protocol_251026.json','analysis_source_261026.json',
                     'analysis_source_v2_261026.json','completed_251026.json']:
            z.write(root/name,name)
        for r in measurements:
            path=root/r['relative_output']
            for name in ['fingerprints.json','command_251026.json']:
                z.write(path/name,str(path.relative_to(root)/name))
        for path in data.glob('*.csv'):z.write(path,path.name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    checksums=[]
    for path in sorted(data.iterdir()):
        if path.is_file() and not path.name.endswith('checksums_261026.txt'):
            checksums.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.name+'\n')
    (data/'abodybuilder3_pipeline_checksums_261026.txt').write_text(''.join(checksums))
    print(json.dumps({'report':str(docs/'abodybuilder3_pipeline_compilation_261026.md'),
                      'archive_bytes':archive.stat().st_size,'completed':True}),flush=True)


if __name__ == '__main__':
    main()
