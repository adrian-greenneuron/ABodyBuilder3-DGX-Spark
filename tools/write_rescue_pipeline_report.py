"""Package anonymised full-pipeline evidence and render speed/RAM/geometry plots."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLOURS={'local':'#087f8c','restrained':'#8545a4','off':'#76818b'}
TARGETS=['tuned128_native_000004','validation256_validation_000255','fresh128_validation_000121']
SHORT={name:f'Antibody {i+1}' for i,name in enumerate(TARGETS)}
NAMES={'local':'Local rescue','restrained':'Restrained rescue','off':'No rescue'}


def read(p):return json.loads(p.read_text())


def collect(root):
    protocol=read(root/'protocol_271026.json');retry=read(root/'retry1_protocol_271026.json')
    geometry=read(root/'geometry_validation_271026.json');assert geometry['completed']
    measurements=read(root/'measurements_271026.json')['rows']+read(root/'measurements_retry1_271026.json')['rows']
    cases=[];rescued=[];comparisons={r['label']:r for r in geometry['comparisons']}
    for m in measurements:
        folder=root/m['directory'];completion=read(folder/'completion_271026.json');b=read(folder/'batching.json')
        assert b['deployment']['checkpoint_sha256']==protocol['inference']['model_checkpoint_sha256']
        assert b['deployment']['manifest_sha256']==protocol['inference']['aot_manifest_sha256']
        assert b['deployment']['eager_shape_fallback_batches']==0 and not b['oom_retries']
        row={k:v for k,v in m.items() if k not in ['verification_command','directory']}
        row.update(prepared=completion['successful'],rescued=completion['rescued'],raw_predictions=completion['targets'],
            complete_condition_summary=m['label']!='reduce_local_r1',
            validity='runtime_failure_retained' if m['label']=='reduce_local_r1' else 'reference_with_three_rejections' if m['policy']=='off' else 'complete',
            structures_per_minute=60*completion['successful']/m['command_seconds'],
            failures=[dict(target=r['target'],error=r['error']) for r in completion['rows'] if not r['prepared_sha256']],
            batch_count=len(b['batches']),batch_sizes=sorted(set(v['size'] for v in b['batches'])),
            deployment=b['deployment'],raw_identical_across_conditions=comparisons[m['label']]['raw_identical'],
            nonrescued_prepared_identical=comparisons[m['label']]['nonrescued_prepared_identical'])
        events=b['phase_events']
        if 'all_preparation_finished' in events:row['phased_preparation_seconds']=events['all_preparation_finished']-events['phased_preparation_start']
        cases.append(row)
        for c in completion['rows']:
            if not c['rescue_applied']:continue
            audit=read(folder/'pdbs'/(c['target']+'.preparation.json'));detail=audit['rescue']
            keep={k:detail[k] for k in ['applied','policy','trigger_seconds','core_seconds','displacement','geometry_gate','heavy_before','heavy_after']}
            keep['heavy_attempts']=[{k:v for k,v in a.items() if k!='intermediate_pdb'} for a in detail['heavy_attempts']]
            entry=next(r for r in geometry['records'] if r['label']==m['label'] and r['target']==c['target'])
            keep.update(label=m['label'],target=c['target'],route=m['route'],replicate=m['replicate'],
                seconds=audit['seconds'],prepared_sha256=c['prepared_sha256'],
                independent_validation=geometry['rescues'][entry['rescue_validation_index']])
            rescued.append(keep)
    index={r['target']:r['raw_sha256'] for r in geometry['records']}
    records=[{k:v for k,v in r.items() if k not in ['raw_sha256','rescue_validation_index','error']} for r in geometry['records']]
    return dict(scope=protocol['question'],input_manifest=protocol['input_manifest'],image_id=protocol['image_id'],
                parent_image=protocol['parent_image'],inference=protocol['inference'],preparation=protocol['preparation'],
                selection='All 512 retained native pairs; no rejection-based exclusion. Two fresh repeats per condition/route; second reverses condition order.',
                ram_scope=protocol['metrics'],cases=cases,targets=[dict(target=k,raw_sha256=v) for k,v in index.items()],
                outcomes=records,unique_geometry=geometry['scores'],rescue_calls=rescued,
                retry_reason=retry['reason'],all_raw_identical=True,all_nonrescued_prepared_identical=True,
                accounted_target_attempts=geometry['accounted_attempts'],unique_scored=geometry['unique_scored'])


def selected(data,route,policy):
    return [r for r in data['cases'] if r['route']==route and r['policy']==policy and r['complete_condition_summary']]


def save(fig,assets,name):
    for ext in ['png','svg']:
        path=assets/(name+'.'+ext);fig.savefig(path,dpi=180,bbox_inches='tight')
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def plots(data,assets):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(13,5.1),layout='constrained')
    groups=[(route,policy) for route in ['openmm','reduce'] for policy in ['local','restrained']]
    names=[]
    for i,(route,policy) in enumerate(groups):
        rows=selected(data,route,policy);rates=[r['structures_per_minute'] for r in rows];centre=512*60/st.median(r['command_seconds'] for r in rows)
        axes[0].plot([min(rates),max(rates)],[i,i],color=COLOURS[policy],lw=2)
        axes[0].scatter(rates,[i-.04,i+.04],color=COLOURS[policy],s=35,alpha=.6)
        axes[0].scatter([centre],[i],color=COLOURS[policy],s=100,zorder=4)
        axes[0].annotate(f'{centre:.1f}',(centre,i),xytext=(8,-3),textcoords='offset points',color=COLOURS[policy],fontweight='bold')
        for j,r in enumerate(rows):
            low,high=[x/2**30 for x in r['ram_sensitivity_bytes']];y=i+(j-.5)*.18
            axes[1].plot([low,high],[y,y],color=COLOURS[policy],lw=3,alpha=.55)
            axes[1].scatter([r['peak_additional_unavailable_bytes']/2**30],[y],color=COLOURS[policy],s=65)
        names.append(('OpenMM' if route=='openmm' else 'Reduce + OpenMM')+'\n'+NAMES[policy])
    failed=next(r for r in data['cases'] if r['validity']=='runtime_failure_retained')
    low,high=[v/2**30 for v in failed['ram_sensitivity_bytes']]
    axes[1].plot([low,high],[2.32,2.32],color='#bd433e',ls=':',lw=2)
    axes[1].scatter([failed['peak_additional_unavailable_bytes']/2**30],[2.32],marker='x',color='#bd433e',s=90)
    axes[1].annotate('CUDA-affected attempt\n508/512 prepared',(high,2.32),xytext=(-5,-14),textcoords='offset points',ha='right',va='top',fontsize=9,color='#bd433e')
    for axis in axes:
        axis.set_yticks(range(4),names if axis is axes[0] else ['']*4);axis.set_ylim(3.6,-.6);axis.grid(axis='x',alpha=.18)
    axes[0].set_xlim(0,168);axes[0].set_xlabel('Successful structures per minute (whole job)')
    axes[1].set_xlim(0,16);axes[1].set_xlabel('Additional unavailable unified RAM (GiB)')
    axes[0].set_title('512/512 prepared; exactly three rescued per job')
    axes[1].set_title('Observed RAM; memory drift limits comparisons')
    fig.suptitle('Full 512-antibody pipeline: local rescue preserves throughput',fontsize=15,fontweight='bold')
    fig.supxlabel('Strict FP32 saved AOT · 12 preparation workers · two fresh completed jobs per condition\nIncludes prediction, loading, preparation, failed trigger attempts and writing. RAM segments show pre/post baseline sensitivity, not confidence intervals.',fontsize=10)
    save(fig,assets,'rescue_pipeline512_performance_271026')
    scores={s['prepared_sha256']:s for s in data['unique_geometry']}
    outcomes={(r['label'],r['target']):r for r in data['outcomes']}
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
    x=[];y=[]
    for target in [r['target'] for r in data['targets']]:
        a=outcomes['openmm_local_r1',target];b=outcomes['openmm_restrained_r1',target]
        x.append(scores[a['prepared_sha256']]['probe']['H_clash_pairs']);y.append(scores[b['prepared_sha256']]['probe']['H_clash_pairs'])
    axes[0].plot([0,100],[0,100],ls='--',color='#8b929b',lw=1)
    axes[0].scatter(x,y,s=25,alpha=.2,color='#153d5d')
    for i,t in enumerate(TARGETS):
        a=outcomes['openmm_local_r1',t];b=outcomes['openmm_restrained_r1',t]
        xx=scores[a['prepared_sha256']]['probe']['H_clash_pairs'];yy=scores[b['prepared_sha256']]['probe']['H_clash_pairs']
        axes[0].scatter([xx],[yy],s=85,color=COLOURS['restrained'],zorder=4)
        axes[0].annotate(SHORT[t],(xx,yy),xytext=(3,9+12*(i==0)),textcoords='offset points',fontsize=9)
    axes[0].text(5,85,'509 byte-identical outputs\nlie on the diagonal',fontsize=11)
    axes[0].set(xlim=(-2,100),ylim=(-5,100),xlabel='Local rescue: Probe H-involving overlap pairs',ylabel='Restrained rescue: same diagnostic',title='Geometry changes are confined to three rescues')
    axes[0].set_aspect('equal',adjustable='box')
    for j,policy in enumerate(['local','restrained']):
        for i,t in enumerate(TARGETS):
            values=[r['displacement']['heavy_max_A'] for r in data['rescue_calls'] if r['target']==t and r['policy']==policy and r['route']=='openmm']
            centre=st.median(values);xx=i+(j-.5)*.24
            axes[1].bar(xx,centre,.23,color=COLOURS[policy],label=NAMES[policy] if i==0 else None)
            axes[1].errorbar(xx,centre,yerr=[[centre-min(values)],[max(values)-centre]],fmt='none',color='#343a40',capsize=4)
    axes[1].set(xticks=range(3),xticklabels=[SHORT[t] for t in TARGETS],ylabel='Largest original heavy-atom movement (Å)',title='Broader cleanup requires larger coordinate changes')
    axes[1].legend(frameon=False,loc='upper left');axes[1].grid(axis='y',alpha=.15);axes[1].set_axisbelow(True)
    fig.suptitle('Restrained rescue improves the three outliers; the other 509 are untouched',fontsize=14,fontweight='bold')
    fig.supxlabel('OpenMM route. Left: first completed repeat, all 512 antibodies; coincident count pairs overlap.\nRight: medians/ranges of two repeats; unfitted displacement. Probe is a separate diagnostic, not MolProbity clashscore or structural accuracy.',fontsize=10)
    save(fig,assets,'rescue_pipeline512_geometry_271026')


def report(data):
    lines=[]
    def add(s=''):lines.append(s)
    add('# Rejection-only rescue in the full 512-antibody pipeline — 27 October 2026')
    add();add('**Both rescue policies complete all 512 antibodies while changing only the three rejected cases.** Local rescue retains essentially the no-rescue runtime; guarded restrained rescue adds about 4–6% to wall time and gives much better heavy geometry on those three. All other 509 prepared outputs are byte-identical to their matched full-pipeline R8 controls within each route. Main and dev remain unchanged.')
    add();add('![Full-pipeline throughput and unified RAM](assets/rescue_pipeline512_performance_271026.png)')
    add();add('## Complete-job performance')
    add();add('These are complete fresh Docker CLI jobs, including model loading, saved-AOT execution, raw prediction, all chemical preparation, the failed H attempt that triggers rescue, and output writing. They are **not prediction-only repertoire rates**. Each condition has two complete technical repeats. The median wall time is used to calculate throughput as `512 × 60 / median seconds`; observed replicate ranges are not confidence intervals. Bold identifies the faster rescue within each route, not statistical significance.')
    add();add('| Route | Rescue policy | Median complete time | Structures/min | Additional unified RAM: two observed peaks | Prepared / rescued |')
    add('| --- | --- | ---: | ---: | ---: | ---: |')
    for route in ['openmm','reduce']:
        for policy in ['local','restrained']:
            rows=selected(data,route,policy);seconds=st.median(r['command_seconds'] for r in rows);rate=512*60/seconds
            ram=[r['peak_additional_unavailable_bytes']/2**30 for r in rows];bold='**' if policy=='local' else ''
            add(f"| {'OpenMM' if route=='openmm' else 'Reduce + OpenMM'} | {NAMES[policy]} | {bold}{seconds:.2f} s{bold} | {bold}{rate:.1f}{bold} | {min(ram):.2f}–{max(ram):.2f} GiB | 512 / 3 |")
    add();add('The matched no-rescue jobs take 210.20 s (OpenMM) and 235.87 s (Reduce + OpenMM), but prepare only 509/512 and retain the three explicit rejection records. Relative to those incomplete controls, local rescue changes median wall time by **+0.04% / +0.75%**, while restrained rescue changes it by **+5.58% / +4.29%**. With two repeats these are descriptive differences; the near-zero local overhead is within observed timing variation. Filling worker slots means the extra time is not the sum of the isolated per-antibody rescue durations.')
    add();add('## Unified RAM: measured peaks, with a material limitation')
    add();add('RAM is measured from host `/proc/meminfo` `MemAvailable` every 20 ms, relative to a 5 s pre-job median, followed by an 8 s post-job window. No RSS/CUDA values are added together. The completed rescue jobs have observed pre-baseline peaks of **8.04–10.76 GiB**. No swap growth was measured.')
    add();add('Post-idle memory drift exceeds 0.25 GiB in six of the 12 selected reference/rescue jobs. For example, the higher Reduce/local peak is 10.76 GiB from its pre-job baseline but 8.55 GiB from its post-job baseline. The plot preserves that sensitivity rather than presenting it as a confidence interval. The measured RAM burden is low relative to the Spark capacity, but **these data do not establish a reliable memory winner between rescue methods**. Host activity, caching and runtime retention can contribute to unavailable RAM; these are not isolated per-process physical-allocation measurements.')
    add();add('## Geometry and unchanged outputs')
    add();add('All 13 attempts account for their 512 raw predictions and preparation audits: 6,656 target attempts and 6,640 written prepared structures. Every unique stored PDB was independently scored—922 distinct hashes—reusing a score only when the entire file was byte-identical. This includes the retained CUDA-affected attempt. All 922 pass the H bond, angle, planar/outward and tetrahedral-orientation screens. Every one of the 27 successful rescue calls also passes the independent paired heavy/core/stereochemistry and full bond-topology checks. Of these, 24 belong to the eight complete rescue jobs; the other three belong to the retained partially failed attempt.')
    add();add('![Paired whole-cohort geometry and rescue movements](assets/rescue_pipeline512_geometry_271026.png)')
    add();add('Raw neural predictions are byte-identical across all compared jobs. In every complete job, the 509 non-rescued prepared PDBs are byte-identical to the corresponding no-rescue route. Rescue is triggered by the exact existing frozen-heavy planar incompatibility error, followed by a confirming geometric preflight. It does not use a target whitelist and does not catch unrelated CUDA or parameterisation failures as reasons to move heavy atoms.')
    add();add('The three rejected motifs are restored without new heavy bond flags >0.10 Å, heavy angle flags >20°, severe heavy contact pairs or true Cα/ILE/THR stereocentre inversions. Their original failed core bonds finish within 0.05 Å and their core angles within 10° of the ff14SB targets. Disulfide connectivity, atom/bond identity and the selected protonation state are preserved. Final hydrogen placement uses the same R8 geometry objective after either repair method; the final file is not an untouched Amber-minimisation endpoint. [The isolated rescue report](heavy_atom_rescue_271026.md) describes the exact objectives, safeguards and earlier failed approaches.')
    add();add('| Rejected input | Local heavy bond / angle / severe-contact flags | Restrained flags | Probe H-overlap pairs, local / restrained | Largest heavy movement, local / restrained |')
    add('| --- | ---: | ---: | ---: | ---: |')
    scores={s['prepared_sha256']:s for s in data['unique_geometry']}
    for t in TARGETS:
        local=[r for r in data['rescue_calls'] if r['route']=='openmm' and r['policy']=='local' and r['target']==t]
        full=[r for r in data['rescue_calls'] if r['route']=='openmm' and r['policy']=='restrained' and r['target']==t]
        h=local[0]['independent_validation']['paired'];v=h['after_heavy']
        lp=st.median(scores[r['prepared_sha256']]['probe']['H_clash_pairs'] for r in local);fp=st.median(scores[r['prepared_sha256']]['probe']['H_clash_pairs'] for r in full)
        add(f"| {SHORT[t]} | {v['heavy_bond_errors_over_0_10_A']} / {v['heavy_angle_errors_over_20_deg']} / {h['after_heavy_contact_pairs']} | 0 / 0 / 0 | {lp:g} / {fp:g} | {st.median(r['displacement']['heavy_max_A'] for r in local):.2f} / {st.median(r['displacement']['heavy_max_A'] for r in full):.2f} Å |")
    add();add('OpenMM route; movement and Probe values are medians of two repeats. Local repair moves 28–33 original heavy atoms; restrained rescue moves almost all heavy atoms of each rejected structure. Reduce gives the same heavy-screen conclusions, with 47–73 local versus1–2 restrained Probe H-overlap pairs per rescued input. Probe is a separate contact diagnostic, **not MolProbity clashscore**. Local repair leaves substantial older strain; restrained rescue clears these heavy screens for the three repaired cases, while the other 509 retain their original heavy geometry. The full 512 cohort must not be described as fully relaxed, clash-free or chemically flawless.')
    add();add('No experimental reference structure is used here. These are geometry and numerical-regression results, not evidence of improved structural or binding-site accuracy. The 512 native pairs are a development panel with combined lengths 220–246 residues, not a probability sample of all antibody repertoires. The combined batch plan differs from the earlier 128/256/128 jobs; exact equality is established between the current matched conditions, not asserted for the older jobs.')
    add();add('## CUDA failure and explicit replacement attempt')
    add();add('The first Reduce/local job produced all 512 raw predictions and audits, and all three rescues succeeded, but four compatible targets failed OpenMM CUDA initialisation with `No compatible CUDA device is available`. It finished with 508 prepared structures in 235.42 s and a 14.20 GiB observed memory peak. This is a runtime failure, not a geometry rejection; its outputs, logs and memory traces remain intact. No CPU/backend fallback or per-target automatic retry was used.')
    add();add('After the original finite queue completed, one explicitly recorded replacement used the **same source, image, configuration and input order**. It prepared 512/512 with exactly three rescues in 237.28 s, using 8.65 GiB by the pre-job estimate. This slower but complete replacement, together with the original second repeat, supplies the two complete Reduce/local timings. The failed attempt is excluded from that summary because its coverage is incomplete, not because its time or memory is unfavourable. The CUDA-initialisation issue remains an observed runtime limitation; this study does not claim to fix it.')
    add();add('## Individual full-job measurements')
    add();add('| Job | Wall time | Prepared | Rescued | Pre-baseline RAM peak | Post-idle drift | Summary use |')
    add('| --- | ---: | ---: | ---: | ---: | ---: | --- |')
    for r in data['cases']:
        use='Retained failed attempt' if not r['complete_condition_summary'] else 'Incomplete no-rescue reference' if r['policy']=='off' else 'Complete rescue comparison'
        add(f"| `{r['label']}` | {r['command_seconds']:.2f} s | {r['prepared']} | {r['rescued']} | {r['peak_additional_unavailable_bytes']/2**30:.2f} GiB | {r['post_idle_drift_bytes']/2**30:+.2f} GiB | {use} |")
    add();add('## Implementation, evidence and decision')
    add();add('The experiment retains the original R8 `prepare_structure` implementation byte-for-byte, then adds explicit opt-in dispatch. Compatible inputs return that original result unchanged; only the exact rejected cause can reach the separately qualified rescue solver. The real worker-controller smoke checks passed for both routes and policies, including exact compatible-output equality. The 45 preparation/regression tests pass. Prediction, AOT packages, model weights, native writer, batching policy and main/dev image tags are unchanged. Each full job used 18 AOT batches, no eager fallback and no OOM splitting.')
    add();add('For speed-focused use, these results support keeping **local rescue as the first explicit candidate**, with guarded restrained rescue as a broader-cleanup option. The full-job penalty of restrained rescue is modest when only 3/512 need it, although its coordinate changes are larger. Neither policy has been enabled on main or dev.')
    add();add('[Anonymised measurements and geometry summaries](data/rescue_pipeline512_results_271026.json), [source/runtime provenance](../provenance_rescue_pipeline512_271026.json), [plot/report generator](../tools/write_rescue_pipeline_report.py), [independent validator](../tools/validate_rescue_pipeline.py), and [experimental Docker layer](../Dockerfile.rescue-pipeline). All input CSVs, raw/prepared structures, detailed per-residue diagnostics, failed attempts, source snapshots and memory traces stay local in `rescue_pipeline512_271026`; none are included here. The new image inherits the exact previously qualified rescue layer. Use the recorded immutable identities and frozen environment to reproduce the optional experiment, not a mutable default tag.')
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);args=p.parse_args()
    data=collect(args.root);(args.repo/'docs/data/rescue_pipeline512_results_271026.json').write_text(json.dumps(data,indent=2)+'\n')
    plots(data,args.repo/'docs/assets');(args.repo/'docs/rescue_pipeline512_271026.md').write_text(report(data))
    print('FULL PIPELINE REPORT READY',len(data['cases']),'attempts',data['unique_scored'],'unique geometry scores',flush=True)


if __name__=='__main__':main()
