"""Plot the retained resource, paired accuracy and job-size measurements.

This is reporting only. It never launches inference or changes experiment data.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'svg.hashsalt':'abb3-comparisons-261026','font.size':11})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

COLORS={'native_writer':'#009e73','cached_compiled_native':'#2c7bb6',
        'cached_combined_native':'#a65eae'}
LABELS={'native_writer':'Native writer','cached_compiled_native':'Cached compiler\n+ native writer',
        'cached_combined_native':'Cached compiler + attention\n+ native writer'}
SHORT={'cached_compiled_native':'Compiler','cached_combined_native':'Compiler + attention'}


def read(path):
    return json.loads(path.read_text())


def save(fig, assets, stem):
    for suffix in ['png','svg']:
        path=assets/f'{stem}_261026.{suffix}'
        fig.savefig(path,dpi=180,metadata={'Date':None} if suffix=='svg' else None)
        if suffix=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def csv_write(path,rows):
    assert rows
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'docs')
    args=parser.parse_args();runs=args.project/'runs';docs=args.output
    assets=docs/'assets';data=docs/'data';assets.mkdir(exist_ok=True);data.mkdir(exist_ok=True)
    root=runs/'cached_native_bulk_261026'
    summary=read(root/'bulk_analysis_261026.json')
    target_rows=read(root/'bulk_per_target_261026.json')
    candidate_labels=['native_writer','cached_compiled_native','cached_combined_native']
    conditions=[(model,precision) for model in ['base','plddt'] for precision in ['normal','strict']]
    resources=[]
    for label in candidate_labels:
        for model,precision in conditions:
            row=next(r for r in summary['resources'] if (r['model'],r['precision'],r['label'])==(model,precision,label))
            control=next(r for r in summary['resources'] if (r['model'],r['precision'],r['label'])==(model,precision,'current_dev'))
            resources.append({'model':model,'precision':precision,'candidate':label,
                'elapsed_change_percent':100*(row['seconds_median']/control['seconds_median']-1),
                'cuda_allocated_change_percent':100*(row['cuda_allocated_peak_gib']/control['cuda_allocated_peak_gib']-1),
                'process_tree_rss_change_percent':100*(row['process_tree_rss_peak_gib']/control['process_tree_rss_peak_gib']-1)})
    fig,axes=plt.subplots(1,3,figsize=(12.4,5.2),sharey=True)
    metrics=[('elapsed_change_percent','Elapsed time',(-75,6)),
             ('cuda_allocated_change_percent','CUDA allocated peak',(-50,6)),
             ('process_tree_rss_change_percent','Process-tree RSS peak',(-8,165))]
    for ax,(metric,title,limits) in zip(axes,metrics):
        for i,label in enumerate(candidate_labels):
            rr=[r for r in resources if r['candidate']==label]
            for offset,r in zip(np.linspace(-.14,.14,4),rr):
                ax.scatter(r[metric],2-i+offset,s=62,marker='o' if r['model']=='base' else 's',
                           facecolor=COLORS[label] if r['precision']=='normal' else 'white',
                           edgecolor=COLORS[label],linewidth=1.4,zorder=3)
        ax.axvline(0,color='#48515a',ls='--',lw=1.1)
        ax.set(title=title,xlabel='Change relative to matched dev (%)',xlim=limits,ylim=(-.5,2.5))
        ax.grid(axis='x',alpha=.18);clean(ax)
    axes[0].set_yticks([2,1,0],[LABELS[x] for x in candidate_labels])
    legend=[Line2D([],[],marker='o' if model=='base' else 's',ls='none',markersize=6,
                   markerfacecolor='#546477' if precision=='normal' else 'white',markeredgecolor='#546477',
                   label=f"{model} · {'TF32' if precision=='normal' else 'FP32'}") for model,precision in conditions]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.58,.10),ncol=4,frameon=False,fontsize=10)
    fig.suptitle('Compilation reduces time and CUDA allocation, while process RSS rises',fontsize=14,y=.97)
    fig.text(.54,.02,'Each dot: one checkpoint × precision combination (four per approach). Runtime uses the two-job median;\nmemory uses the higher of two peaks. Offsets separate conditions; these are not confidence intervals. Do not add CUDA and RSS.',
             ha='center',fontsize=9.5)
    fig.tight_layout(rect=[0,.22,1,.91]);save(fig,assets,'abodybuilder3_resource_tradeoffs')
    csv_write(data/'abodybuilder3_resource_tradeoffs_261026.csv',resources)

    # Use one copy of each biological target. Both fresh process repetitions
    # produced the same fingerprints, so pooling them would duplicate the data.
    assert all(r['raw_exact']==r['count'] for r in summary['repeatability'])
    paired=[]
    for model,precision in conditions:
        for label in candidate_labels[1:]:
            rows=[r for r in target_rows if (r['model'],r['precision'],r['label'],r['replicate'],r['comparison'])==
                  (model,precision,label,1,'vs_matched_dev')]
            assert len(rows)==len({r['target'] for r in rows})==100
            for row in rows:
                h3=row['accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd']
                reference=row['reference_accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd']
                paired.append({'model':model,'precision':precision,'candidate':label,'target':row['target'],
                    'h3_experimental_rmsd':h3,'reference_h3_experimental_rmsd':reference,
                    'h3_error_delta_angstrom':h3-reference,
                    'largest_atom_displacement_angstrom':row['largest_atom']['displacement']})
    csv_write(data/'abodybuilder3_paired_accuracy_plot_data_261026.csv',paired)
    for kind in ['h3','atom']:
        fig,axes=plt.subplots(2,2,figsize=(11.5,7.4),sharey=True)
        for column,precision in enumerate(['normal','strict']):
            scale=1e3 if precision=='normal' else 1e6
            coldata=[r for r in paired if r['precision']==precision]
            if kind=='h3':
                limit=max(abs(r['h3_error_delta_angstrom'])*scale for r in coldata)*1.15
            else:
                smallest=min(r['largest_atom_displacement_angstrom'] for r in coldata)
                largest=max(r['largest_atom_displacement_angstrom'] for r in coldata)
            for row_index,model in enumerate(['base','plddt']):
                ax=axes[row_index,column]
                for label in candidate_labels[1:]:
                    selected=[r for r in paired if (r['model'],r['precision'],r['candidate'])==(model,precision,label)]
                    values=np.sort([r['h3_error_delta_angstrom']*scale if kind=='h3' else
                                    r['largest_atom_displacement_angstrom'] for r in selected])
                    assert len(values)==100 and np.isfinite(values).all()
                    if kind=='atom':assert (values>0).all()
                    ax.step(values,np.arange(1,101),where='post',lw=2.2,color=COLORS[label],label=SHORT[label])
                ax.set_title(f"{model} · {'TF32 permitted' if precision=='normal' else 'Strict FP32'}",fontsize=12)
                ax.set_ylim(0,101);ax.set_yticks([0,25,50,75,100]);ax.grid(alpha=.18);clean(ax)
                if column==0:ax.set_ylabel('Cumulative antibodies (%)')
                if kind=='h3':
                    ax.axvline(0,color='#48515a',ls='--',lw=1.1)
                    ax.set_xlim(-limit,limit)
                    ax.set_xlabel('Δ H3 experimental RMSD (×10⁻³ Å)' if precision=='normal' else
                                  'Δ H3 experimental RMSD (×10⁻⁶ Å)')
                else:
                    ax.set_xscale('log');ax.set_xlim(smallest*.75,largest*1.45)
                    ax.set_xlabel('Largest single-atom displacement (Å; logarithmic)')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.955),ncol=2,frameon=False)
        if kind=='h3':
            title='Paired H3 accuracy changes vary by antibody and remain small'
            footer='100 distinct experimental antibodies per curve; one copy per target. Negative = closer to experiment; positive = farther.\nComparisons use the same-precision dev reference. Note the 1,000× difference between the column scales.'
            stem='abodybuilder3_paired_h3_accuracy'
        else:
            title='Small backbone-error changes can coexist with larger local atom shifts'
            footer='100 distinct antibodies per curve; maximum valid-atom displacement within each target, relative to same-precision dev.\nDisplacement measures prediction disagreement, not experimental side-chain error. The native writer itself is exact.'
            stem='abodybuilder3_local_atom_displacements'
        fig.suptitle(title,fontsize=14,y=.995)
        fig.text(.53,.015,footer,ha='center',fontsize=9.5)
        fig.tight_layout(rect=[0,.085,1,.915]);save(fig,assets,stem)

    # Crossover ratios use matched complete-job timings, with both repetitions
    # visible. Their observed range is not a statistical confidence interval.
    cross=[]
    for study in ['cached_native_crossover_261026','cached_native_crossover_boundary_261026','cached_native_bulk_261026']:
        cross.extend(read(runs/study/'measurements_261026.json'))
    ratios=[];fig,axes=plt.subplots(1,2,figsize=(11.5,4.9),sharex=True,sharey=True)
    sizes=[1000,1500,2000,5000,10000]
    for ax,model in zip(axes,['base','plddt']):
        for precision,color,marker,name in [('normal','#a65eae','o','TF32 · compiler + attention'),
                                            ('strict','#2c7bb6','s','Strict FP32 · compiler')]:
            label='cached_combined_native' if precision=='normal' else 'cached_compiled_native'
            centre=[];low=[];high=[]
            for count in sizes:
                rows=[r for r in cross if (r['count'],r['model'],r['precision'])==(count,model,precision)]
                a=sorted([r for r in rows if r['label']=='native_writer'],key=lambda r:r['replicate'])
                b=sorted([r for r in rows if r['label']==label],key=lambda r:r['replicate'])
                assert [r['replicate'] for r in a]==[r['replicate'] for r in b]==[1,2]
                values=np.array([x['container_seconds']/y['container_seconds'] for x,y in zip(a,b)])
                ratio=float(np.median([x['container_seconds'] for x in a])/np.median([x['container_seconds'] for x in b]))
                assert values.min()-1e-12<=ratio<=values.max()+1e-12
                centre.append(ratio);low.append(ratio-values.min());high.append(values.max()-ratio)
                for rep,v in enumerate(values,1):
                    ratios.append({'model':model,'precision':precision,'count':count,'replicate':rep,
                        'eager_seconds':a[rep-1]['container_seconds'],'cached_seconds':b[rep-1]['container_seconds'],
                        'paired_speed_ratio':float(v),'ratio_of_time_medians':ratio})
            ax.errorbar(sizes,centre,yerr=[low,high],fmt=marker+'-',color=color,lw=2.2,
                        markersize=7,capsize=4,label=name)
        ax.axhline(1,color='#48515a',lw=1.1,ls='--');ax.axvspan(1000,1500,color='#708090',alpha=.09)
        ax.set_xscale('log');ax.set_xticks(sizes,['1,000','1,500','2,000','5,000','10,000'])
        ax.set(title=model,xlabel='Inputs per complete job (logarithmic spacing)',ylim=(.55,2.7))
        ax.grid(axis='y',alpha=.18);clean(ax)
        ax.text(1070,.6,'Eager faster',fontsize=9,color='#48515a')
        ax.text(1800,2.55,'Cached execution faster',fontsize=9,color='#48515a')
    axes[0].set_ylabel('Eager / cached elapsed time (×)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.93),ncol=2,frameon=False)
    fig.suptitle('The faster execution route changes between 1,000 and 1,500 inputs',fontsize=14,y=.995)
    fig.text(.54,.015,'Both routes use the native writer. Points = ratios of two-job median times; whiskers = range of paired ratios, not confidence intervals.\nLines guide the eye; they are not a fitted cost model. Shading marks the measured crossover bracket for these antibody lengths.',
             ha='center',fontsize=9.4)
    fig.tight_layout(rect=[0,.14,1,.86]);save(fig,assets,'abodybuilder3_job_size_crossover')
    csv_write(data/'abodybuilder3_crossover_plot_data_261026.csv',ratios)
    paths=[*assets.glob('abodybuilder3_resource_tradeoffs_261026.*'),
           *assets.glob('abodybuilder3_paired_h3_accuracy_261026.*'),
           *assets.glob('abodybuilder3_local_atom_displacements_261026.*'),
           *assets.glob('abodybuilder3_job_size_crossover_261026.*'),
           data/'abodybuilder3_resource_tradeoffs_261026.csv',data/'abodybuilder3_paired_accuracy_plot_data_261026.csv',
           data/'abodybuilder3_crossover_plot_data_261026.csv']
    (docs/'comparison_plots_SHA256SUMS_261026.txt').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(docs))+'\n' for p in sorted(paths)))
    print('Rendered four comparison figures; exact plotted values saved to CSV.',flush=True)


if __name__=='__main__':main()
