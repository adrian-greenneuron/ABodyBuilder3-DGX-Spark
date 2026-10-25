"""Score all stored raw predictions and their paired experimental accuracy."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from structural_metrics import compare, CDRS


def short(metrics):
    return {k:v for k,v in metrics.items() if k not in ('per_residue_aligned_ca_distance','ca_lddt')} | {
        'ca_lddt':metrics['ca_lddt']['score']}


def confidence(logits):
    if not len(logits):return None
    z=np.asarray(logits,dtype=np.float64);z-=z.max(-1,keepdims=True)
    probs=np.exp(z);probs/=probs.sum(-1,keepdims=True)
    return probs @ np.arange(1,101,2)


def bootstrap_paired(values,clusters):
    unique=sorted(set(clusters));rng=np.random.default_rng(251026)
    groups=[np.flatnonzero(np.array(clusters)==x) for x in unique]
    replicates=[np.mean(values[np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])])
                for _ in range(2000)]
    return np.percentile(replicates,[2.5,97.5]).tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args();root=args.study
    all_rows=[];summaries=[];timings=[];layer_stats=[]
    for model in ('base','plddt'):
        folders={p.name:p for p in (root/model).iterdir() if (p/'metadata_251026.json').exists()}
        base_meta=json.loads((folders['accepted']/'metadata_251026.json').read_text())
        names=list(base_meta['targets']);exp=[x for x in names if base_meta['targets'][x]['experimental']]
        metrics={}
        for variant,folder in folders.items():
            metadata=json.loads((folder/'metadata_251026.json').read_text())
            if set(names)!=set(metadata['targets']):raise ValueError('Unmatched target sets')
            pair_folder=folders['strict_reference'] if variant in ('strict_attention','strict_compiled') else folders['accepted']
            rows=[];full=[]
            for name in names:
                d=np.load(folder/(name+'.npz'));reference=np.load(pair_folder/(name+'.npz'))
                numerical=compare(d['positions'],reference['positions'],d['masks'],reference['masks'],
                                  d['heavy'],d['regions'])
                experimental=None
                if 'truth_positions' in d:
                    experimental=compare(d['positions'],d['truth_positions'],d['masks'],d['truth_masks'],
                                         d['heavy'],d['regions'],False)
                score,reference_score=confidence(d['confidence_logits']),confidence(reference['confidence_logits'])
                conf=None if score is None else {'max_absolute_plddt_difference':float(np.abs(score-reference_score).max()),
                    'mean_absolute_plddt_difference':float(np.abs(score-reference_score).mean()),
                    'mean_plddt':float(score.mean()),
                    'per_residue_plddt_difference':(score-reference_score).tolist()}
                row={'model':model,'variant':variant,'name':name,'comparison_reference':pair_folder.name,
                    'experimental':experimental is not None,
                    'coordinates_exact':bool(np.array_equal(d['positions'],reference['positions'])),
                    'numerical':short(numerical),'accuracy':short(experimental) if experimental else None,
                    'confidence':{k:v for k,v in conf.items() if k!='per_residue_plddt_difference'} if conf else None}
                rows.append(row);full.append({**row,'numerical':numerical,'accuracy':experimental,'confidence':conf})
            all_rows.extend(rows);metrics[variant]=rows
            (folder/'structural_scores_251026.json').write_text(json.dumps(full,indent=2,allow_nan=False)+'\n')
            records=metadata['measurements']
            timings.append({'model':model,'variant':variant,'targets':len(names),
                'median_model_pass_seconds':statistics.median(r['seconds'] for r in records),
                'range_seconds':[min(r['seconds'] for r in records),max(r['seconds'] for r in records)],
                'first_batch_forward_seconds':metadata['first_batch_forward_seconds'],
                'peak_cuda_allocated_gib':max(r['peak_cuda_allocated_bytes'] for r in records)/2**30,
                'peak_cuda_reserved_gib':max(r['peak_cuda_reserved_bytes'] for r in records)/2**30,
                'sampled_process_rss_gib':max(r['sampled_process_tree_rss_peak_bytes'] for r in records)/2**30})
        # The current dev shortcuts themselves must match the frozen original on
        # this expanded set, with identical batch shapes, before attributing drift.
        exact_control=all(r['coordinates_exact'] for r in metrics['reference'])
        baseline={r['name']:r for r in metrics['accepted']}
        for variant,rows in metrics.items():
            true_rows=[r for r in rows if r['experimental']]
            summary={'model':model,'variant':variant,'targets':len(rows),'experimental_targets':len(true_rows),
                'reference_accepted_control_exact':exact_control,
                'coordinates_exact_count':sum(r['coordinates_exact'] for r in rows),
                'max_raw_ca_displacement':max(r['numerical']['raw_ca_max'] for r in rows),
                'max_raw_atom_displacement':max(r['numerical']['raw_atom_max'] for r in rows),
                'max_aligned_ca_rmsd':max(r['numerical']['aligned_ca_rmsd'] for r in rows),
                'experimental_region_changes':{}}
            cdrh3=sorted(rows,key=lambda r:r['numerical']['regions'].get('cdrh3',{}).get('framework_aligned_backbone_rmsd',-1),reverse=True)
            summary['largest_cdrh3_prediction_differences']=[{'name':r['name'],
                'framework_aligned_backbone_rmsd':r['numerical']['regions']['cdrh3']['framework_aligned_backbone_rmsd']}
                for r in cdrh3[:5] if 'cdrh3' in r['numerical']['regions']]
            for region in ['fwh','fwl',*CDRS]:
                available=[r for r in true_rows if region in r['accuracy']['regions']]
                if not available:continue
                values=np.array([r['accuracy']['regions'][region]['framework_aligned_backbone_rmsd'] for r in available])
                control=np.array([baseline[r['name']]['accuracy']['regions'][region]['framework_aligned_backbone_rmsd'] for r in available])
                delta=values-control;clusters=[r['name'].split('_')[0] for r in available]
                summary['experimental_region_changes'][region]={'targets':len(available),
                    'baseline_mean_backbone_rmsd':float(control.mean()),'candidate_mean_backbone_rmsd':float(values.mean()),
                    'mean_paired_rmsd_change':float(delta.mean()),'median_paired_rmsd_change':float(np.median(delta)),
                    'pdb_cluster_bootstrap_95_ci_mean_change':bootstrap_paired(delta,clusters),
                    'max_absolute_paired_change':float(np.abs(delta).max()),
                    'worsened_by_over_0_01_A':int((delta>.01).sum()),'improved_by_over_0_01_A':int((delta<-.01).sum()),
                    'worsened_by_over_0_05_A':int((delta>.05).sum()),
                    'worst_worsening':{'name':available[int(delta.argmax())]['name'],'change':float(delta.max())},
                    'largest_improvement':{'name':available[int(delta.argmin())]['name'],'change':float(delta.min())}}
            lddt=np.array([r['accuracy']['ca_lddt'] for r in true_rows])
            control_lddt=np.array([baseline[r['name']]['accuracy']['ca_lddt'] for r in true_rows])
            summary['experimental_ca_lddt']={'baseline_mean':float(control_lddt.mean()),'candidate_mean':float(lddt.mean()),
                'mean_paired_change_percentage_points':float((lddt-control_lddt).mean()),
                'max_absolute_paired_change_percentage_points':float(np.abs(lddt-control_lddt).max())}
            if variant in ('strict_attention','strict_compiled'):
                strict_control={r['name']:r for r in metrics['strict_reference']}
                changes=np.array([r['accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd']-
                    strict_control[r['name']]['accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd'] for r in true_rows])
                summary['precision_matched_cdrh3_accuracy_change']={'reference':'strict_reference',
                    'mean':float(changes.mean()),'max_absolute':float(np.abs(changes).max())}
            candidate_clashes=np.array([r['numerical']['geometry']['severe_clashes'] for r in true_rows])
            control_clashes=np.array([baseline[r['name']]['numerical']['geometry']['severe_clashes'] for r in true_rows])
            summary['severe_clash_proxy']={'baseline_total':int(control_clashes.sum()),'candidate_total':int(candidate_clashes.sum()),
                'targets_with_increase':int((candidate_clashes>control_clashes).sum()),
                'max_increase_per_target':int((candidate_clashes-control_clashes).max())}
            summary['peptide_outliers']={'baseline_total':sum(baseline[r['name']]['numerical']['geometry']['bonds']['peptide_C_N']['outside_1_1_to_1_6'] for r in true_rows),
                'candidate_total':sum(r['numerical']['geometry']['bonds']['peptide_C_N']['outside_1_1_to_1_6'] for r in true_rows)}
            if model=='plddt':
                summary['max_absolute_plddt_change']=max(r['confidence']['max_absolute_plddt_difference'] for r in rows)
                summary['mean_target_absolute_plddt_change']=float(np.mean([r['confidence']['mean_absolute_plddt_difference'] for r in rows]))
            summaries.append(summary)
        for variant,folder in folders.items():
            f=folder/'ipa_first_batch.npz'
            if not f.exists():continue
            ref=folders['strict_reference'] if variant in ('strict_attention','strict_compiled') else folders['accepted']
            with np.load(f) as a,np.load(ref/'ipa_first_batch.npz') as b:
                layer_stats.append({'model':model,'variant':variant,'reference':ref.name,
                    'scope':'First fixed batch; IPA outputs including padded rows; diagnostic pass only.',
                    'layers':[{'layer':i,'max_absolute_difference':float(np.abs(a[str(i)]-b[str(i)]).max()),
                        'rms_difference':float(np.sqrt(np.mean((a[str(i)].astype(np.float64)-b[str(i)])**2)))} for i in range(8)]})
    compact={'scope':'Paired published upstream test set; implementation screening, not a new universal accuracy validation.',
             'summaries':summaries,'timings':timings,'ipa_diagnostics':layer_stats}
    (root/'structural_summary_251026.json').write_text(json.dumps(compact,indent=2,allow_nan=False)+'\n')
    (root/'structural_per_target_251026.json').write_text(json.dumps(all_rows,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completed':True,'conditions':len(summaries),'target_condition_records':len(all_rows)}),flush=True)


if __name__=='__main__':main()
