"""Per-antibody direction of changes, with numerical/reporting ties explicit."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

NORMAL=['accepted','attention','point_ieee','compiled']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args();root=args.study
    data=json.loads((root/'structural_per_target_251026.json').read_text())
    ids=json.loads((root/'data/test_ids_251026.json').read_text())
    summary=[];all_rows=[]
    for model in ['base','plddt']:
        conditions={variant:{r['name']:r for r in data if r['model']==model and r['variant']==variant and r['experimental']}
                    for variant in NORMAL}
        for variant in NORMAL[1:]:
            base=[conditions['accepted'][name] for name in ids]
            actual=[conditions[variant][name] for name in ids]
            result={'model':model,'variant':variant,'experimental_accuracy':{}}
            for metric in ['ca','cdrh3','lddt']:
                if metric=='ca':
                    a=np.array([r['accuracy']['aligned_ca_rmsd'] for r in actual]);b=np.array([r['accuracy']['aligned_ca_rmsd'] for r in base])
                elif metric=='cdrh3':
                    a=np.array([r['accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd'] for r in actual]);b=np.array([r['accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd'] for r in base])
                else:
                    a=-np.array([r['accuracy']['ca_lddt'] for r in actual]);b=-np.array([r['accuracy']['ca_lddt'] for r in base])
                delta=a-b
                result['experimental_accuracy'][metric]={'better_unrounded':int((delta < -1e-10).sum()),
                    'worse_unrounded':int((delta > 1e-10).sum()),'numerical_ties_1e_10':int((abs(delta)<=1e-10).sum()),
                    'ties_within_0_01_A_or_percentage_point':int((abs(delta)<=.01).sum()),
                    'largest_absolute_change':float(abs(delta).max())}
                for i,name in enumerate(ids):
                    all_rows.append({'model':model,'variant':variant,'target':name,'metric':metric,
                        'candidate_score':float(-a[i] if metric=='lddt' else a[i]),
                        'reference_score':float(-b[i] if metric=='lddt' else b[i]),
                        'change_negative_is_better':float(delta[i])})
            summary.append(result)
    pair_summary=json.loads((root/'pairwise_summary_251026.json').read_text())
    labels=pair_summary['labels'];agreement=[];accuracy_winners=[]
    for model in ['base','plddt']:
        reference=labels.index(model+'/accepted');candidate_indices=[labels.index(model+'/'+v) for v in NORMAL[1:]]
        for metric in ['ca_aligned_rmsd','cdrh3_framework_backbone_rmsd','local_ca_distance_rmsd']:
            counts={v:0 for v in NORMAL[1:]};examples={v:[] for v in NORMAL[1:]}
            for name in ids:
                p=json.loads((root/'pairwise'/(name+'.json')).read_text())
                distances=np.array(p['matrices'][metric])[reference,candidate_indices]
                for i in np.flatnonzero(distances<=distances.min()+1e-12):
                    variant=NORMAL[1:][int(i)];counts[variant]+=1;examples[variant].append(name)
            agreement.append({'model':model,'metric':metric,'closest_candidate_to_current_prediction_counts':counts,
                'targets':examples,'ties':'All candidates within 1e-12 Å of the minimum count as tied.'})
        for metric in ['ca','cdrh3']:
            scores=[]
            for name in ids:
                values=[]
                for v in NORMAL:
                    row=next(r for r in data if r['name']==name and r['model']==model and r['variant']==v)
                    values.append(row['accuracy']['aligned_ca_rmsd'] if metric=='ca' else row['accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd'])
                values=np.array(values)
                scores.append({'target':name,'best_unrounded':[NORMAL[i] for i in np.flatnonzero(values<=values.min()+1e-10)],
                    'best_within_0_01_A':[NORMAL[i] for i in np.flatnonzero(values<=values.min()+.01)],
                    'score_spread':float(values.max()-values.min())})
            accuracy_winners.append({'model':model,'metric':metric,
                'unrounded_best_counts':{v:sum(v in r['best_unrounded'] for r in scores) for v in NORMAL},
                'best_within_0_01_A_counts':{v:sum(v in r['best_within_0_01_A'] for r in scores) for v in NORMAL},
                'per_target':scores})
    report={'numerical_equality_tolerance':1e-10,'reporting_tolerance':.01,
        'warning':'Unrounded winner counts describe tiny numerical changes; they are not evidence of biological superiority.',
        'paired_experimental_directions':summary,'agreement_with_existing_prediction':agreement,
        'experimental_winners':accuracy_winners}
    (root/'antibody_rankings_251026.json').write_text(json.dumps(report,indent=2)+'\n')
    with (root/'per_antibody_accuracy_changes_251026.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    print(json.dumps({'completed':True,'paired_metric_records':len(all_rows)}),flush=True)


if __name__=='__main__':main()
