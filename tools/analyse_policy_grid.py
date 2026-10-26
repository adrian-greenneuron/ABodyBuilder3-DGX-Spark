"""Compare completed batch-cap/CPU-worker screens without changing dev policy."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np
from analyse_bulk_pipeline import load
from structural_metrics import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caps',type=Path,required=True)
    parser.add_argument('--workers',type=Path,required=True)
    parser.add_argument('--truth',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    baseline=args.caps/'batch32_workers2'
    reference=json.loads((baseline/'measurements_261026.json').read_text())
    cap16=args.caps/'batch16_workers2'
    reference16=json.loads((cap16/'measurements_261026.json').read_text())
    names=list(json.loads((args.truth/'base/compiled/metadata_251026.json').read_text())['targets'])
    truth={name:load(args.truth/'base/compiled'/f'{name}.npz') for name in names}
    summaries=[];checks=[]
    for parent in [args.caps,args.workers]:
        for study in sorted(parent.glob('batch*_workers*')):
            records=json.loads((study/'measurements_261026.json').read_text())
            grouped=defaultdict(list)
            for row in records:
                key=row['model'],row['precision'],row['replicate']
                other=next(r for r in reference if (r['model'],r['precision'],r['replicate'])==key)
                path=study/row['relative_output'];ref=baseline/other['relative_output']
                actual_fp=json.loads((path/'fingerprints.json').read_text())
                ref_fp=json.loads((ref/'fingerprints.json').read_text())
                assert len(actual_fp)==len(ref_fp)==1000
                assert [x['name'] for x in actual_fp]==[x['name'] for x in ref_fp]
                numerical=[];h3=[]
                for name in names:
                    a,b,t=load(path/'raw'/f'{name}.npz'),load(ref/'raw'/f'{name}.npz'),truth[name]
                    n=compare(a['positions'],b['positions'],a['masks'],b['masks'],a['heavy'],t['regions'],False)
                    ac=compare(a['positions'],t['truth_positions'],a['masks'],t['truth_masks'],a['heavy'],t['regions'],False)
                    bc=compare(b['positions'],t['truth_positions'],b['masks'],t['truth_masks'],b['heavy'],t['regions'],False)
                    numerical.append(n)
                    h3.append(ac['regions']['cdrh3']['framework_aligned_backbone_rmsd']-
                              bc['regions']['cdrh3']['framework_aligned_backbone_rmsd'])
                check={'policy':study.name,'model':key[0],'precision':key[1],'replicate':key[2],
                    'raw_exact_vs_batch32':sum(a==b for a,b in zip(actual_fp,ref_fp)),
                    'max_ca_displacement_vs_batch32':max(r['raw_ca_max'] for r in numerical),
                    'max_atom_displacement_vs_batch32':max(r['raw_atom_max'] for r in numerical),
                    'h3_max_absolute_accuracy_delta_vs_batch32':float(max(abs(x) for x in h3)),
                    'h3_mean_accuracy_delta_vs_batch32':float(statistics.mean(h3))}
                if parent==args.workers:
                    r16=next(r for r in reference16 if (r['model'],r['precision'],r['replicate'])==key)
                    p16=cap16/r16['relative_output']
                    assert row['batching']['batches']==r16['batching']['batches']
                    fp16=json.loads((p16/'fingerprints.json').read_text())
                    assert actual_fp==fp16
                    assert all((path/'pdbs'/f"{r['name']}.pdb").read_bytes()==
                               (p16/'pdbs'/f"{r['name']}.pdb").read_bytes() for r in actual_fp)
                    check['all_1000_raw_and_pdb_exact_vs_two_workers']=True
                checks.append(check)
                grouped[key[:2]].append(row)
            for key,rows in sorted(grouped.items()):
                summaries.append({'policy':study.name,'model':key[0],'precision':key[1],
                    'seconds_median':statistics.median(r['container_command_seconds'] for r in rows),
                    'seconds_runs':[r['container_command_seconds'] for r in rows],
                    'cuda_allocated_peak_gib':max(r['metrics']['peak_cuda_allocated_bytes'] for r in rows)/2**30,
                    'process_tree_rss_peak_gib':max(r['metrics']['sampled_process_tree_rss_peak_bytes'] for r in rows)/2**30})
            print('Scored policy',study.name,flush=True)
    args.output.write_text(json.dumps({'scope':'1,000 input rows from 100 distinct experimental antibodies; '
        'two fresh jobs per model/precision/policy. Cap comparisons use batch32/two workers; '
        'worker-only exact comparisons use batch16/two workers. Memory estimate budget24GiB for this screen.',
        'resources':summaries,'comparisons':checks},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
