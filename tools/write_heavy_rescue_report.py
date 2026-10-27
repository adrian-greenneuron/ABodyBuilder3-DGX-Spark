"""Package sequence-free rescue evidence and render the paired comparisons."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES={'fixed':'No rescue','local':'Local repair','restrained':'Restrained relaxation'}
COLOURS={'raw':'#75808f','local':'#087f8c','restrained':'#8545a4'}
TARGETS=['tuned128_native_000004','validation256_validation_000255','fresh128_validation_000121']
SHORT={t:f'Antibody {i+1}' for i,t in enumerate(TARGETS)}


def read(path):return json.loads(path.read_text())


def atom_arrays(path):
    values={}
    for line in path.read_text().splitlines():
        if line.startswith(('ATOM  ','HETATM')):
            key=(line[21:22],line[22:27],line[12:16]);values[key]=(line[76:78].strip(),np.array([float(line[i:i+8]) for i in [30,38,46]]))
    return values


def saved_comparison(a,b):
    x,y=atom_arrays(a),atom_arrays(b);assert x.keys()==y.keys()
    result={'byte_identical':a.read_bytes()==b.read_bytes()}
    for label,is_h in [('heavy',False),('H',True)]:
        d=[np.linalg.norm(x[k][1]-y[k][1]) for k in x if (x[k][0]=='H')==is_h]
        result[label+'_rms_A']=float(np.sqrt(np.mean(np.square(d))));result[label+'_max_A']=float(max(d))
    return result


def collect(root):
    protocol=read(root/'qualification_protocol_271026.json');measurements=read(root/'qualification_measurements_271026.json')
    geometry=read(root/'qualification_stored_geometry_271026.json');assert measurements['completed'] and geometry['completed']
    index={(r['label'],r['target']):r for r in geometry['rows']};rows=[]
    for case in protocol['cases']:
        job=read(root/case['directory']/'preparation/job_271026.json')
        for r in job['rows']:
            item={k:r[k] for k in ['target','group','policy','route','status','seconds_to_output','failing_centres_before']}
            item.update(label=case['label'],replicate=case['replicate'],raw_sha256=r['raw_sha256'])
            for key in ['heavy_before','heavy_after','heavy_attempts','repair_preflight_seconds','heavy_rescue_seconds',
                        'final_H_seconds','final_H_optimisation','rescue_displacement','prepared_sha256','error','disulfide_count']:
                if key in r:item[key]=r[key]
            if r['status']=='completed':
                item['peptide_bonds_over_1_8_A']=r['prepared_geometry']['peptide_cn_over_1_8_angstrom']
                item['max_peptide_cn_A']=r['prepared_geometry']['max_peptide_cn_angstrom']
                item['disulfide_lengths_A']=[v['distance_angstrom'] for v in r['prepared_geometry']['disulfides']]
                item['reduce_moved_heavy_count']=len(r['reduce_orientation_changes']['moved_heavy_atoms'])
                item['max_raw_heavy_displacement_A']=r['raw_changes']['max_original_heavy_displacement_angstrom']
                item['protonation_variants_sha256']=hashlib.sha256(json.dumps(r['protonation_variants']).encode()).hexdigest()
                item['validation']=index[case['label'],r['target']]
            rows.append(item)
    repeat=[]
    for route in ['openmm','reduce']:
        for policy in ['fixed','local','restrained']:
            for target in [r['target'] for r in read(root/'protocol_271026.json')['inputs']]:
                paths=[root/'qualification'/f'{route}_{policy}_run{rep}'/'preparation'/target/'rescued.prepared.pdb' for rep in [1,2]]
                if all(p.exists() for p in paths):repeat.append(dict(route=route,policy=policy,target=target,**saved_comparison(*paths)))
    controls=[]
    for route in ['openmm','reduce']:
        for target in [r['target'] for r in rows if r['route']==route and r['group']=='compatible_control' and r['policy']=='fixed' and r['replicate']==1]:
            hashes={r['prepared_sha256'] for r in rows if r['target']==target and r['route']==route}
            controls.append(dict(route=route,target=target,all_six_policy_repeat_outputs_byte_identical=len(hashes)==1))
    assert all(r['all_six_policy_repeat_outputs_byte_identical'] for r in controls)
    screens=[]
    for version in range(1,6):
        screen=read(root/f'screen_v{version}_271026.json')
        for r in screen['rows']:
            keep={k:r[k] for k in ['target','group','condition','status','preparation_and_scoring_seconds']}
            for key in ['config','error','heavy_before','heavy_after','original_motifs_after','original_motifs_restored',
                        'new_heavy_bond_errors_over_0_10_A','new_heavy_angle_errors_over_20_deg','heavy_chiral_inversions',
                        'biological_heavy_chiral_inversions','movements','heavy_repair','hydrogen_geometry_passed']:
                if key in r:keep[key]=r[key]
            screens.append(dict(screen_version=version,**keep))
    diagnosis=[]
    for r in read(root/'diagnosis/heavy_rescue_diagnosis_271026.json')['rows']:
        diagnosis.append({'target':r['target'],'group':r['group'],'heavy':r['heavy'],
            'failed_angles_deg':[b['heavy_angle_deg'] for b in r['bad']],
            'failed_motif_bond_lengths_A':[b['heavy_distances_A'] for b in r['bad']]})
    return dict(question=read(root/'protocol_271026.json')['question'],inputs=read(root/'protocol_271026.json')['inputs'],diagnosis=diagnosis,
                scope=protocol['scope'],image_id=protocol['image_id'],parent_runtime=protocol['parent_runtime'],
                policies={k:protocol[k] for k in ['local_policy','restrained_policy','gates','ram_scope','selection']},
                measurements=measurements['rows'],qualifications=rows,repeatability=repeat,controls=controls,screening=screens)


def med(rows,key):return st.median(r[key] for r in rows)


def save(fig,folder,name):
    for extension in ['png','svg']:fig.savefig(folder/(name+'.'+extension),dpi=180,bbox_inches='tight')
    plt.close(fig)


def plots(data,assets):
    rows=[r for r in data['qualifications'] if r['route']=='openmm' and r['group']=='rejected' and r['status']=='completed']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(14,4.1),layout='constrained')
    panels=[('Preparation time (s)',lambda r:r['seconds_to_output']),
            ('Largest heavy-atom movement (Å)',lambda r:r['rescue_displacement']['heavy_max_A']),
            ('Probe H-involving overlap pairs',lambda r:r['validation']['probe']['H_clash_pairs'])]
    for axis,(title,get) in zip(axes,panels):
        for i,t in enumerate(TARGETS):
            xs=[]
            for policy,offset in [('local',-.10),('restrained',.10)]:
                values=[get(r) for r in rows if r['target']==t and r['policy']==policy];middle=st.median(values);xs.append(middle)
                axis.errorbar(middle,i+offset,xerr=[[middle-min(values)],[max(values)-middle]],fmt='o',markersize=8,capsize=4,color=COLOURS[policy],label=NAMES[policy] if i==0 else None)
            axis.plot(xs,[i-.10,i+.10],color='#c0c5ca',lw=1.5,zorder=0)
        axis.set_yticks(range(3),[SHORT[t] for t in TARGETS]);axis.invert_yaxis();axis.set_xlabel(title);axis.grid(axis='x',alpha=.18);axis.set_xlim(left=0)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.09),ncol=2,frameon=False,fontsize=11)
    fig.suptitle('Local repair is faster and moves fewer atoms; relaxation removes more clashes',fontsize=15,fontweight='bold')
    fig.supxlabel('OpenMM route · three rejected antibodies · two repeats · three preparation workers\nDots: medians; bars: observed ranges. Saved predictions only; no neural inference. Probe is a diagnostic, not a MolProbity clashscore.',fontsize=10)
    save(fig,assets,'heavy_atom_rescue_tradeoffs_271026')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    panels=[('Heavy bond errors >0.10 Å','heavy_bond_errors_over_0_10_A'),('Heavy angle errors >20°','heavy_angle_errors_over_20_deg'),('Severe heavy-atom contact pairs','contacts'),('Angle at the original failed amide N (°)','motif')]
    for axis,(title,key) in zip(axes.ravel(),panels):
        for j,policy in enumerate(['raw','local','restrained']):
            values=[]
            for t in TARGETS:
                local=[r for r in rows if r['target']==t and r['policy']=='local']
                picked=[r for r in rows if r['target']==t and r['policy']==policy] if policy!='raw' else local
                if key=='motif':
                    value=st.median(r['validation']['motifs'][0]['heavy_angle_deg'] for r in picked) if policy!='raw' else next(d['failed_angles_deg'][0] for d in data['diagnosis'] if d['target']==t)
                elif key=='contacts':value=st.median(r['validation']['heavy']['after_heavy_contact_pairs' if policy!='raw' else 'before_heavy_contact_pairs'] for r in picked)
                else:value=st.median(r['validation']['heavy']['after_heavy' if policy!='raw' else 'before_heavy'][key] for r in picked)
                values.append(value)
            axis.bar(np.arange(3)+(j-1)*.23,values,.22,color=COLOURS[policy],label='Raw prediction' if policy=='raw' else NAMES[policy])
            for i,v in enumerate(values):
                if v==0:axis.text(i+(j-1)*.23,.4,'0',ha='center',va='bottom',color=COLOURS[policy],fontsize=10)
        axis.set_title(title);axis.set_xticks(range(3),[SHORT[t] for t in TARGETS]);axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
        if key=='motif':
            axis.axhspan(111.9,131.9,color='#f2d35e',alpha=.2);axis.axhline(121.9,color='#6e642b',lw=1,ls='--');axis.set_ylim(0,145)
    axes[0,0].legend(frameon=False,fontsize=10)
    fig.suptitle('Both restore the failed motif; local repair leaves unrelated heavy-atom strain',fontsize=15,fontweight='bold')
    fig.supxlabel('OpenMM route · medians of two stored-PDB checks. Severe contacts: <0.7 × vdW radius sum, excluding atoms within three bonds.\nShaded angle band: ff14SB target 121.9° ±10°. Counts are geometry screens, not structural accuracy measurements.',fontsize=10)
    save(fig,assets,'heavy_atom_rescue_geometry_271026')


def report(data):
    rows=data['qualifications'];m=data['measurements'];lines=[]
    def add(value=''):lines.append(value)
    add('# Rescuing the three rejected antibodies — 27 October 2026')
    add();add('**All three can be rescued by either targeted local repair or restrained relaxation.** Both routes passed the stored-file hydrogen, valence, contact and stereochemistry checks in two repeats, with and without Reduce orientation. These remain explicit experiments on this component branch; main and dev are unchanged.')
    add();add('Local repair is the faster, less disruptive choice for making the previously rejected hydrogen placement possible. Restrained relaxation improves the rest of the heavy geometry much more, but moves almost the whole structure. Neither result establishes better experimental structural accuracy.')
    add();add('![Speed, movement and hydrogen contacts](assets/heavy_atom_rescue_tradeoffs_271026.png)')
    add();add('## Speed and coordinate changes')
    add();add('Per-antibody preparation below includes atom repair, rescue checks, hydrogen placement and writing, while three workers share the GPU. Values are medians of two calls. Coordinate changes are **unfitted displacements from the starting prediction**, not RMSD after structural alignment. The reported local count excludes newly added terminal atoms.')
    add();add('| Rejected input | Local preparation | Restrained preparation | Original heavy atoms moved >0.01 Å, local / restrained | Largest heavy movement, local / restrained |')
    add('| --- | ---: | ---: | ---: | ---: |')
    for t in TARGETS:
        local=[r for r in rows if r['target']==t and r['policy']=='local' and r['route']=='openmm'];full=[r for r in rows if r['target']==t and r['policy']=='restrained' and r['route']=='openmm']
        counts=[r['rescue_displacement']['heavy_moved_over_0_01_A'] for r in full]
        count=str(min(counts)) if min(counts)==max(counts) else f'{min(counts)}–{max(counts)}'
        add(f"| {SHORT[t]} | {med(local,'seconds_to_output'):.2f} s | {med(full,'seconds_to_output'):.2f} s | {int(st.median(r['rescue_displacement']['heavy_moved_over_0_01_A'] for r in local))} / {count} | {st.median(r['rescue_displacement']['heavy_max_A'] for r in local):.2f} / {st.median(r['rescue_displacement']['heavy_max_A'] for r in full):.2f} Å |")
    add();add('OpenMM route. Local heavy-atom displacement RMS is 0.053–0.143 Å; restrained RMS is about 0.34–0.49 Å. Small global RMS values hide substantial local changes: the largest local movement is **1.29–3.02 Å**. “Local” describes the repair region, not a sub-ångström limit or a mathematically proven minimum-displacement solution.')
    add();add('Whole-container jobs process the same six saved predictions: the three rejected inputs and three compatible controls. Times include container/Python/worker startup and output writing; **they exclude neural inference and independent post-run scoring**. Runs reverse condition and input order in the second repeat. The fixed-heavy control is deliberately incomplete and is not a six-success throughput baseline.')
    add();add('| Route | Policy | Successful inputs per job | Container time, median (two individual runs) | Additional unified RAM, observed range |')
    add('| --- | --- | ---: | ---: | ---: |')
    for route in ['openmm','reduce']:
        for policy in ['fixed','local','restrained']:
            picked=[r for r in m if r['route']==route and r['policy']==policy];picked.sort(key=lambda r:r['replicate'])
            times=[r['command_seconds'] for r in picked];ram=[r['peak_additional_unavailable_bytes']/2**30 for r in picked]
            add(f"| {'OpenMM' if route=='openmm' else 'Reduce + OpenMM'} | {NAMES[policy]} | {'3/6; three rejected' if policy=='fixed' else '6/6'} | {st.median(times):.2f} s ({times[0]:.2f}, {times[1]:.2f}) | {min(ram):.2f}–{max(ram):.2f} GiB |")
    add();add('Thus local rescue takes roughly **half the complete preparation-job time** of the restrained option. These six deliberately selected cases are not a repertoire throughput benchmark. RAM uses host `MemAvailable` sampled every 20 ms, with a 5 s pre-job baseline and 8 s post-job window; it does not add RSS to CUDA memory. Three no-rescue jobs have >0.25 GiB post-idle drift. Across the study, sensitivity to the pre/post baseline is retained in the data; these small differences do **not** support a memory ranking. No swap growth was measured.')
    add();add('## What was wrong, and what each method fixes')
    add();add('The original failed backbone C–N–Cα angles are **56.3°, 52.8° and 59.2°**, against the ff14SB target of 121.9°. A planar outward amide H cannot satisfy the existing angle limit at those frozen heavy coordinates. One failed C–N bond is also 3.06 Å long; the other two are about 1.06–1.07 Å, versus a 1.335 Å target. This is a heavy-geometry problem, so changing only the hydrogen optimiser cannot resolve it.')
    add();add('![Paired geometry changes](assets/heavy_atom_rescue_geometry_271026.png)')
    add();add('**Local repair** moves the failed residue and its neighbours. It tries radius one first, then radius two from the original coordinates if the geometry gate fails. Antibody 1 needs radius two; antibodies 2 and 3 pass radius one. The recorded objective uses heavy-only Amber valence/torsion terms, stronger bond and angle penalties, positional restraints, a bounded steric term and orientation guards. Existing strained non-core bonds/angles retain their original targets during this small repair, so the method does not pull a local patch into fixing unrelated chain gaps. The radius-two attempt uses a stronger steric penalty. This is a geometry objective, **not full Amber minimisation**.')
    add();add('**Restrained relaxation** uses a global geometry prepass, then fixed-heavy physical H minimisation and CUDA mixed-precision Amber ff14SB/GBn2 minimisation with heavy-atom positional restraints of 1,000 kJ mol⁻¹ nm⁻². Restraints stay anchored to the original repaired prediction, not the prepass. An explicit signed-volume wall protects the original true protein stereocentres. Both physical stages must meet the recorded RMS-force tolerance of 10 kJ mol⁻¹ nm⁻¹ within 3,000 iterations. All 12 rescued target calls converged. The chirality wall means this is **Amber plus additional restraints**, not unmodified Amber.')
    add();add('All policies first use the same deterministic PDBFixer repair, which adds the missing terminal OXT atoms; no missing sequence is reconstructed. Both rescue methods then discard the temporary H coordinates and use the **same previously tested R8 GPU hydrogen protocol**, with pH 7 and the original selected protonation variants. That final H objective omits Coulomb/GB and uses exact X–H lengths, Amber angles/torsions and bounded sterics. The final file is therefore a repaired-heavy result followed by a common geometry-based H placement, not an untouched endpoint of the full physical minimisation. Reduce is an explicit orientation-first option, and its hydrogens are rebuilt against the authoritative topology; no backend or protonation fallback is used.')
    add();add('Local repair leaves **49–62 heavy bond flags**, **22–38 heavy angle flags** and **16–53 severe heavy contact pairs** per rescued structure, including 8–14 peptide C–N bonds longer than 1.8 Å. Restrained relaxation reduces all four heavy screens to zero in these cases. Probe still finds **one or two H-involving bad-overlap pairs** after relaxation, versus **47–73** after local repair across the repeated OpenMM/Reduce runs. Probe uses a different contact definition from the simple severe-distance screen; these numbers are not MolProbity clashscores. No output should be described as clash-free or fully chemically validated.')
    add();add('## Verification and boundaries')
    add();add('All 24 rescued outputs (three inputs × two policies × two routes × two repeats) pass independent checks after re-reading the PDB. Across the 60 completed outputs, the tests found no H bond errors >0.10 Å, H angle errors >30°, wrong-side α/tetrahedral H, tetrahedral signed-volume inversions, or planar/inward H violations >1°. Heavy atom/bond topology, disulfide connectivity and protonation identity are preserved relative to the atom-repaired, optionally Reduce-oriented reference. For each rescue, the original failed motif has heavy bond errors ≤0.05 Å and angle error ≤10°; no new heavy bond/angle flags, severe heavy contacts or true Cα/ILE/THR stereocentre inversions were introduced relative to its starting geometry.')
    add();add('Compatible controls take the no-heavy-repair branch. For each of the three controls and each route, **all six outputs across the three policies and two repeats are byte-identical**. Local rescued files also repeat byte-for-byte within each route. Restrained results vary across independent mixed-precision calls: the largest between-repeat heavy-atom RMS difference is 0.0083 Å, with a maximum individual heavy movement of 0.147 Å. H differences reach 1.47 Å for an individual atom; all repeated outputs still pass the same geometry screens. This is not a bitwise deterministic physical-relaxation claim.')
    add();add('The focused force/geometry tests and inherited preparation regression suite passed: **40 tests**. Previous neural/AOT/writer evidence is reused because those paths were not changed. This study prepares saved predictions and does not rerun the full 512-antibody inference job; the earlier 509 compatible results remain separate evidence. The three rescued cases were used to develop the repair, not held out to estimate general rescue rates. There is no experimental-structure reference here, and no claim about improved binding-site accuracy, Ramachandran quality or absolute stereochemical correctness of the initial predictions.')
    add();add('## Failed approaches retained')
    add();add('The first screen had an import-path setup failure before minimisation; the mounted candidate directory shadowed the installed PDBFixer resources. An initial image build also used unsupported bare-local-ID `FROM` syntax; the successful build verified the parent tag against its immutable ID first. Both logs remain local.')
    add();add('Small unguarded local repairs enabled H placement but introduced heavy valence defects. Ordinary full restrained minimisation converged but inverted **1, 3 and 1 true stereocentres** in the three rejected structures. A very stiff chirality wall prevented inversion but failed to converge. The qualified protocol uses the recorded prepass and softer wall instead. A later radius-two local attempt introduced one new severe contact; increasing its steric penalty removed that contact without lowering any geometry threshold. Screening times share a process and include scoring, so they are not pooled with the fresh-container benchmarks.')
    add();add('## Recommendation')
    add();add('Keep a geometry-gated **local rescue as an explicit experimental option** for the rejected motifs, with a clear warning that other strain remains. Offer guarded restrained relaxation separately when broader geometry improvement justifies the extra movement and runtime. Do not silently relax a structure that failed local repair. Main and dev remain unchanged pending review.')
    add();add('## Evidence and reproduction')
    add();add('[Compact sequence-free measurements and screening history](data/heavy_atom_rescue_results_271026.json), [runtime/source identities](../provenance_heavy_atom_rescue_271026.json), and [PNG/SVG figure sources](../tools/write_heavy_rescue_report.py). Input identifiers map as follows: antibody 1 = `tuned128_native_000004`, antibody 2 = `validation256_validation_000255`, antibody 3 = `fresh128_validation_000121`. Raw structures, detailed per-residue diagnostics, failed attempts and traces stay in the local `heavy_atom_rescue_271026` run directory; none are included in this repository.')
    add();add('The image is an incremental experimental layer, not a standalone upstream installation. Verify the R8 parent image ID from the provenance before building `Dockerfile.heavy-rescue`; the installed predictor and preparation CLI are not modified. Run `tools/run_heavy_rescue.py` explicitly with `--policy fixed|local|restrained`, `--route openmm|reduce`, a local study root containing the hashed inputs/protocol, and a new output directory. Use the frozen environment settings in the provenance. `tools/measure_frozen_commands.py` measures the recorded finite command list; `tools/validate_rescue_jobs.py` independently scores its stored outputs with the retained unmodified Probe binary. [The parent hydrogen study](hydrogen_geometry_271026.md) documents the final H objective and earlier 509 compatible cases.')
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);args=p.parse_args()
    data=collect(args.root);folder=args.repo/'docs/data';folder.mkdir(exist_ok=True);assets=args.repo/'docs/assets';assets.mkdir(exist_ok=True)
    (folder/'heavy_atom_rescue_results_271026.json').write_text(json.dumps(data,indent=2)+'\n')
    plots(data,assets);(args.repo/'docs/heavy_atom_rescue_271026.md').write_text(report(data))
    print('PACKAGED',len(data['qualifications']),'target attempts;',len(data['screening']),'screening calls',flush=True)


if __name__=='__main__':main()
