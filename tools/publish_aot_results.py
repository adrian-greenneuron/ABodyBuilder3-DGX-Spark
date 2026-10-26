"""Publish retained AOT/JIT measurements, paired plots and compact evidence.

Reads frozen experiment records; performs no inference and changes no defaults.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    root,repo=args.study,args.repo
    assets,data=repo/'docs/assets',repo/'docs/data'
    assets.mkdir(exist_ok=True);data.mkdir(exist_ok=True)
    records=json.loads((root/'measurements_261026.json').read_text())
    assert len(records)==24 and json.loads((root/'completed_261026.json').read_text())['completed']
    analyses={n:json.loads((root/f'aot_n{n}_analysis_261026.json').read_text()) for n in [100,1000,10000]}
    comparisons={n:json.loads((root/f'aot_n{n}_per_target_261026.json').read_text()) for n in analyses}
    index={}
    resources=[]
    for count,analysis in analyses.items():
        for row in analysis['resources']:
            row={'inputs':count,**row}
            index[count,row['model'],row['label']]=row
            resources.append({k:v for k,v in row.items() if k not in ('images','seconds_runs','evidence_writing_seconds')} |
                             {'seconds_run1':row['seconds_runs'][0],'seconds_run2':row['seconds_runs'][1]})
    paired=[]
    for count,rows in comparisons.items():
        for row in rows:
            numeric=row['numerical']
            a=row['accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd']
            b=row['reference_accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd']
            paired.append({'inputs':count,'model':row['model'],'replicate':row['replicate'],
                'target':row['target'],'aligned_ca_rmsd_vs_jit_A':numeric['aligned_ca_rmsd'],
                'aligned_backbone_rmsd_vs_jit_A':numeric['aligned_backbone_rmsd'],
                'framework_aligned_h3_backbone_rmsd_vs_jit_A':numeric['regions']['cdrh3']['framework_aligned_backbone_rmsd'],
                'max_raw_atom_displacement_A':numeric['raw_atom_max'],
                'h3_experimental_error_aot_A':a,'h3_experimental_error_jit_A':b,
                'h3_experimental_error_change_A':a-b,
                'ca_lddt_change_pp':row['accuracy']['ca_lddt']-row['reference_accuracy']['ca_lddt'],
                'max_confidence_change':row.get('max_confidence_difference',0.)})
    write_csv(data/'abodybuilder3_aot_resources_261026.csv',resources)
    write_csv(data/'abodybuilder3_aot_paired_predictions_261026.csv',paired)
    # Every repeated input must agree between fresh processes within condition.
    for analysis in analyses.values():
        assert all(r['raw_exact']==r['count'] for r in analysis['repeatability'])
    # Figures use one record per biological target; technical repeats are not pooled.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','savefig.facecolor':'white'})
    colours={'cached_compiled_native':'#CE7625','aot_native':'#246E9E'}
    names={'cached_compiled_native':'Cached JIT + native writer','aot_native':'Saved AOT + native writer'}
    def save(fig,name):
        for extension in ['png','svg']:
            path=assets/(name+'.'+extension)
            fig.savefig(path,dpi=180,bbox_inches='tight')
            if extension=='svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.9),sharex=True,sharey=True)
    for ax,model in zip(axes,['base','plddt']):
        for label in colours:
            rows=[index[n,model,label] for n in analyses]
            ax.plot(list(analyses),[r['seconds_median'] for r in rows],'-o',color=colours[label],label=names[label],lw=2,ms=7)
            for n,row in zip(analyses,rows):
                ax.scatter([n*.98,n*1.02],row['seconds_runs'],s=24,color=colours[label],alpha=.65,zorder=3)
        for n in analyses:
            a=index[n,model,'aot_native'];b=index[n,model,'cached_compiled_native']
            ax.annotate(f"{b['seconds_median']/a['seconds_median']:.2f}×",(n,a['seconds_median']),
                        xytext=(0,-22),textcoords='offset points',ha='center',fontsize=11,fontweight='bold')
        ax.set_xscale('log');ax.set_yscale('log')
        ax.set_xticks(list(analyses),['100','1,000','10,000'])
        ax.yaxis.set_major_locator(FixedLocator([5,10,20,50,100]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        ax.set_ylim(3.6,120);ax.set_xlim(70,14500)
        ax.grid(axis='y',alpha=.2);ax.set_title(model+' checkpoint')
        ax.set_xlabel('Antibodies processed in the job')
    axes[0].set_ylabel('Complete job time (seconds; log scale)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.92),ncol=2,frameon=False)
    fig.suptitle('Saving the executable removes most JIT startup cost',fontsize=16,fontweight='bold')
    fig.text(.06,.015,'Strict FP32 · two fresh containers per condition · every PDB written · ratios show JIT time / AOT time',fontsize=10)
    fig.subplots_adjust(top=.77,bottom=.19,wspace=.12)
    save(fig,'abodybuilder3_aot_runtime_261026')
    fig,axes=plt.subplots(1,3,figsize=(12,4.4))
    fields=[('cuda_allocated_peak_gib','PyTorch CUDA allocated'),('process_tree_rss_peak_gib','Process-tree RSS'),('container_memory_peak_gib','Container lifetime memory')]
    for ax,(field,title) in zip(axes,fields):
        for y,model in enumerate(['base','plddt']):
            for label,offset in [('cached_compiled_native',-.16),('aot_native',.16)]:
                row=index[10000,model,label]
                ax.barh(y+offset,row[field],height=.27,color=colours[label])
                ax.text(row[field]+.045,y+offset,f'{row[field]:.2f}',va='center',fontsize=10)
        ax.set_yticks([0,1],['base','pLDDT']);ax.invert_yaxis()
        ax.set_xlim(0,max(index[10000,m,l][field] for m in ['base','plddt'] for l in colours)*1.2)
        ax.set_xlabel('GiB');ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Peak memory for complete 10,000-input jobs',fontsize=15,fontweight='bold')
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
    fig.text(.05,.02,'Higher peak of two jobs. RSS can double-count shared compiler-worker mappings; these measures must not be added.',fontsize=10)
    fig.subplots_adjust(top=.75,bottom=.2,wspace=.35)
    save(fig,'abodybuilder3_aot_memory_261026')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    for model,colour in [('base','#246E9E'),('plddt','#8052A0')]:
        rows=[r for r in paired if r['inputs']==10000 and r['model']==model and r['replicate']==1]
        assert len(rows)==100 and len({r['target'] for r in rows})==100
        for ax,key,factor in [(axes[0],'h3_experimental_error_change_A',1e6),
                              (axes[1],'aligned_ca_rmsd_vs_jit_A',1e6)]:
            values=np.sort([r[key]*factor for r in rows])
            ax.step(values,np.arange(1,101),where='post',lw=2.2,label=model,color=colour)
            ax.grid(alpha=.17);ax.set_ylim(0,102)
    axes[0].axvline(0,color='#666666',ls='--',lw=1)
    axes[0].set_xlabel('Change in H3 experimental RMSD (millionths of an Å)')
    axes[0].set_title('Negative: lower experimental error')
    axes[0].set_ylabel('Antibodies at or below value (%)')
    axes[1].set_xlabel('Aligned Cα RMSD vs JIT (millionths of an Å)')
    axes[1].set_title('Prediction disagreement, not experimental error')
    axes[1].legend(frameon=False,loc='lower right')
    fig.suptitle('Strict-FP32 AOT and JIT predictions remain very close',fontsize=15,fontweight='bold')
    fig.text(.05,.02,'100 distinct experimental antibodies · one observation per target/checkpoint · second technical repetition was identical',fontsize=10)
    fig.subplots_adjust(top=.8,bottom=.2,wspace=.22)
    save(fig,'abodybuilder3_aot_accuracy_261026')
    runtime_rows=[]
    for count in analyses:
        for model in ['base','plddt']:
            a=index[count,model,'aot_native'];b=index[count,model,'cached_compiled_native']
            runtime_rows.append(f"| {count:,} | {model} | {b['seconds_median']:.2f} | **{a['seconds_median']:.2f}** | {b['seconds_median']/a['seconds_median']:.2f}× |")
    accuracy_rows=[]
    for count in analyses:
        for model in ['base','plddt']:
            rows=[r for r in paired if r['inputs']==count and r['model']==model and r['replicate']==1]
            accuracy_rows.append(f"| {count:,} | {model} | {np.mean([r['aligned_ca_rmsd_vs_jit_A'] for r in rows]):.3g} | {max(abs(r['h3_experimental_error_change_A']) for r in rows):.3g} | {max(r['max_raw_atom_displacement_A'] for r in rows):.6f} |")
    memory_rows=[]
    for model in ['base','plddt']:
        for label in colours:
            r=index[10000,model,label]
            memory_rows.append(f"| {model} | {'AOT' if label=='aot_native' else 'Cached JIT'} | {r['cuda_allocated_peak_gib']:.3f} | {r['cuda_reserved_peak_gib']:.3f} | {r['process_tree_rss_peak_gib']:.3f} | {r['container_memory_peak_gib']:.3f} |")
    prep={}
    manifest=json.loads((root/'aot_packages_261026.json').read_text())
    for model in manifest:
        prep[model]={domain:json.loads((root/Path(path).parent/'preparation_261026.json').read_text()) for domain,path in manifest[model].items()}
    report=[
        '# Ahead-of-time deployment results — 26 October 2026','',
        '**Saving and loading the complete executable works.** Strict-FP32 AOT deployment is **4.9× faster at 100 inputs, 2.6× at 1,000 and 1.07× at 10,000** than the existing cached-JIT compiler path. These are complete fresh-container jobs, including every PDB write. Both use the same native writer. Main and dev remain unchanged; the candidate is available only on this isolated component branch.','',
        '## Complete-job performance','',
        '![Fresh-container runtime comparison](assets/abodybuilder3_aot_runtime_261026.png)','',
        '| Inputs | Checkpoint | Cached JIT (s) | Saved AOT (s) | Speedup |',
        '| --- | --- | --- | --- | --- |',*runtime_rows,'',
        'Values are medians of two fresh containers. Small dots retain both measurements; ranges are not confidence intervals. Bold marks the lower measured median, not statistical significance. The comparison used the same input order, batch shapes, two CPU preparation/writer threads, native formatter, seed 42, checkpoint weights and eight learned structure updates. No job retried after OOM. All **24 jobs and 88,800 PDB outputs** completed. Jobs above 100 inputs repeat the same **100 distinct published test antibodies**, with unique output names and no prediction cache; they do not add independent biological targets.','',
        'The earlier 1,000–1,500-input crossover described the **disk-cached JIT implementation**. It should not be applied to this AOT implementation. We did not rerun eager inference or test jobs below 100 inputs in this study, so these results do not establish a new universal AOT/eager threshold.','',
        '## What was saved','',
        'A tensor-input wrapper exports the existing model with `torch.export.export(strict=False)`, then `torch._inductor.aoti_compile_and_package` produces a loadable executable package. Each checkpoint has separate packages for batch size 1 and batch sizes 2–32. The adapter selects a package by microbatch size; it never changes precision or silently falls back to another implementation. The packages declare length bounds 200–272; the actual experimental antibodies span 219–242 residues, so empirical qualification is narrower than the export bounds.','',
        'Fixed-shape export worked directly. Dynamic export initially failed because the original identity-tensor LRU caches cannot hash symbolic dimensions. During export only, the adapter calls the exact undecorated `identity_rot_mats`, `identity_trans` and `identity_quats` functions. It restores their original bindings afterwards. Focused CPU tests and all four preparation probes confirmed exact eager outputs for this cache bypass. No quaternion/rotation formula, checkpoint weight or third-party source file was rewritten.','',
        'The pinned NGC loader also requires an explicit `torch._inductor.codecache` import. This is handled in the isolated adapter. The failed direct-loader attempt, symbolic-cache failure and an initial Docker parent-reference packaging failure remain in the audit records. Completed preparation was reused after the controller fix.','',
        'Fresh deployment containers read both executable packages from Docker, with **no host model-package or compiler-cache mount**. Their compiler-cache paths were redirected to fresh unused directories. Every AOT job recorded **zero Dynamo graphs**; JIT controls still performed graph setup. Package loading took roughly **0.09–0.11 seconds per package**. Synchronized, fresh-process preparation probes took **0.20–0.45 seconds for the first model call**, depending on batch size. Those probes are distinct from complete CLI time and are not isolated kernel timings.','',
        f"Preparing the four executable packages required {sum(v['package_compile_seconds'] for m in prep.values() for v in m.values()):.1f} seconds of compilation in total, plus export/model setup. Each two-package checkpoint bank is about {sum(v['package_bytes'] for v in prep['base'].values())/2**20:.1f} MiB for base and {sum(v['package_bytes'] for v in prep['plddt'].values())/2**20:.1f} MiB for pLDDT. Preparation is outside timed inference and need not repeat for each job. The Docker layers also inherit the existing parent image; these sizes describe package payloads, not complete image downloads.",'',
        '## Predictions and experimental accuracy','',
        '![Paired structural agreement and experimental error changes](assets/abodybuilder3_aot_accuracy_261026.png)','',
        'AOT is **not bit-identical to JIT**. Across the 100 distinct targets, mean Cα RMSD after whole-Fv Cα alignment was about **0.0000014 Å**. The largest change in CDR-H3 experimental backbone RMSD was **0.00000479 Å**. Mean H3 errors round to the same six decimals (base 2.914036 Å; pLDDT 2.799067 Å), with no target changing by 0.01 Å. That is a descriptive reporting band, not a formal equivalence threshold or a proof of universal accuracy preservation.','',
        '| Inputs | Checkpoint | Mean Cα RMSD vs JIT (Å) | Largest absolute H3 error change (Å) | Largest valid-atom shift (Å) |',
        '| --- | --- | --- | --- | --- |',*accuracy_rows,'',
        'H3 error follows heavy-framework Cα alignment, using the same retained experimental coordinates and masks. Accuracy and disagreement use only valid atoms; larger preliminary differences in padded/nonexistent atom slots were excluded from scientific scoring. The plot uses one observation per target/checkpoint from the 10,000-input jobs. Both fresh repetitions were scored separately and gave identical raw fingerprints within every condition. Across all job sizes and both methods, **44,400 paired prediction fingerprints** matched between fresh repetitions; repetitions are not pooled as extra antibodies.','',
        'The largest valid-atom disagreement was 0.002394 Å (base, H66 CE1); this is a prediction displacement, not an experimental side-chain error. The largest Cα-lDDT change was 0.004219 percentage points and the largest pLDDT confidence change was below 0.00050 points. Compiler changes can alter rounding even under strict FP32. There is no evidence here of a material backbone-accuracy gain or loss, but neither atomic identity nor broader structural accuracy is established. Native-writer exactness relies on its existing regression evidence; this study verified use of that writer and all expected outputs rather than rerunning an unrelated formatter qualification.','',
        '## Memory','',
        '![Complete-job memory comparison](assets/abodybuilder3_aot_memory_261026.png)','',
        '| Checkpoint | Execution | CUDA allocated (GiB) | CUDA reserved (GiB) | Process-tree RSS (GiB) | Container lifetime (GiB) |',
        '| --- | --- | --- | --- | --- | --- |',*memory_rows,'',
        'These are the higher peaks of the two complete 10,000-input jobs. CUDA allocation increases by about **0.132 GiB (5.8%)**, while sampled process-tree RSS falls from about **4.64 to 1.60–1.62 GiB**. Container lifetime accounting also falls. RSS can double-count shared compiler-worker pages; container accounting includes a different set of memory. On the Spark’s unified memory, **do not add these columns** or interpret CUDA allocation alone as total physical usage. All job sizes and both individual timings are retained in the resource CSV.','',
        '## Scope, decision and reproduction','',
        'These results support reviewing **strict FP32 + saved AOT executable + native writer** as the next deployment candidate. They remove much of the earlier startup objection without requiring a geometry rewrite. Main/dev defaults remain unchanged pending discussion. The language checkpoint, TF32 AOT, optional chemistry, jobs below 100 inputs, wider sequence lengths, different runtimes/GPUs and complete-image download cost were not tested here. Exported input bounds must not be confused with validated biological coverage.','',
        'The experimental cohort has already been used during runtime optimisation. This is a regression assessment on those reused targets, not an independent new validation set.','',
        'Preparation: `tools/prepare_aot_package.py` exports and packages a checkpoint; `tools/check_aot_package.py` tests loading in a fresh process. `tools/aot_model.py` supplies the export wrapper and guarded deployment adapter. `tools/run_aot_study.py` freezes prepared packages into Docker layers and queues paired, fresh-container comparisons; its `--resume` path verifies completed preparation before reuse. The comparison uses `tools/measure_pipeline.py --mode aot --precision strict --aot-root /aot_packages`; this option belongs to the isolated measurement harness, not the production CLI.','',
        'The preparation/controller records retain exact commands, package hashes, source hashes and parent/new image identities. Fresh-study preparation requires the frozen component checkout, the `aot_packages_261026.json` package-path map and the retained input/cache study described in the preceding reports. CPU scoring uses `tools/analyse_bulk_pipeline.py --reference-label cached_compiled_native --expected-jobs 8` once per job size; `tools/publish_aot_results.py` regenerates this report from those records.','',
        'The runtime remains NGC PyTorch 2.10.0a0+b558c986e8.nv25.11, CUDA 13.0, DGX Spark GB10. Every GPU process set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before CUDA initialization, `float32_matmul_precision=highest` and CUDA matrix `allow_tf32=False`. Base/pLDDT parents are the same pinned cached-JIT/native images as before. Five focused CPU checks cover cache-binding restoration and fail-closed precision/path/shape guards; the real pipeline covered both exported batch domains.','',
        '- [Individual times and resource peaks](data/abodybuilder3_aot_resources_261026.csv)',
        '- [Per-target paired metrics, including H3 and confidence](data/abodybuilder3_aot_paired_predictions_261026.csv)',
        '- [Protocol, identities, commands, preparation, raw fingerprints and full scores](data/abodybuilder3_aot_deployment_evidence_261026.zip)','',
        'Large exported preparation graphs, executable packages, raw predictions, PDBs and every Docker image remain local and intact. The evidence archive contains no model weights, raw coordinates or biological sequences. Main remains `a5deeb1b73994491f94b5724d2f2baa9993d47a6`; dev remains `bd8a12ea9f58fa3577215532653a05c47e90e256`.','']
    (repo/'docs/abodybuilder3_aot_deployment_261026.md').write_text('\n'.join(report))
    evidence=data/'abodybuilder3_aot_deployment_evidence_261026.zip'
    files={}
    for p in root.glob('*.json'):files[str(p.relative_to(root))]=p
    for p in root.glob('aot_n*_*.json'):files[str(p.relative_to(root))]=p
    for p in root.glob('check_*/check_261026.json'):files[str(p.relative_to(root))]=p
    for p in root.glob('**/preparation_261026.json'):
        if 'package_context' not in str(p):files[str(p.relative_to(root))]=p
    for p in root.glob('n*/**/command_261026.json'):files[str(p.relative_to(root))]=p
    for p in root.glob('n*/**/fingerprints.json'):files[str(p.relative_to(root))]=p
    for p in root.glob('n*/**/metrics_261026.json'):files[str(p.relative_to(root))]=p
    for p in root.glob('n*/**/batching.json'):files[str(p.relative_to(root))]=p
    for p in root.glob('n*/**/formatter_usage.json'):files[str(p.relative_to(root))]=p
    for p in root.glob('base_dynamic_uncached_identities/source/**/*.py'):files[str(p.relative_to(root))]=p
    for p in root.glob('analysis_source_261026/*'):files[str(p.relative_to(root))]=p
    for p in root.glob('frozen_component/**/*.py'):files[str(p.relative_to(root))]=p
    for p in root.glob('controller_initial_261026.py'):files[str(p.relative_to(root))]=p
    for p in root.glob('*tests*.log'):files[str(p.relative_to(root))]=p
    for p in root.glob('controller*261026.log'):files[str(p.relative_to(root))]=p
    for p in root.glob('build_*attempt*.log'):files[str(p.relative_to(root))]=p
    for name in ['probe_fixed_export.log','base_dynamic.log','base_dynamic_uncached_identities.log',
                 'base_fresh_check.log','base_fresh_check_import_ready.log']:
        files[name]=root/name
    with zipfile.ZipFile(evidence,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,path in sorted(files.items()):archive.write(path,name)
        archive.writestr('SHA256SUMS', ''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {name}\n' for name,p in sorted(files.items())))
    print(json.dumps({'report':'docs/abodybuilder3_aot_deployment_261026.md','evidence_bytes':evidence.stat().st_size,
                      'evidence_files':len(files),'resource_rows':len(resources),'paired_target_rows':len(paired)}))


if __name__=='__main__':main()
