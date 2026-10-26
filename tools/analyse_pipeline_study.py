"""Score completed full-pipeline jobs with the established structural metrics."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from analyse_structures import confidence, short
from structural_metrics import compare, ATOM_NAMES


def load(path):
    with np.load(path) as d:
        return {key: d[key] for key in d.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--truth-study', type=Path, required=True)
    p.add_argument('--replicate', type=int, choices=[1,2], default=1)
    args = p.parse_args(); root = args.study
    measurements = json.loads((root/'measurements_251026.json').read_text())
    assert len(measurements) == 48
    grouped = {}
    for r in measurements:
        key = r['count'], r['model'], r['precision'], r['mode']
        grouped.setdefault(key, []).append(r)
    ids = list(json.loads((args.truth_study/'base/compiled/metadata_251026.json').read_text())['targets'])
    truth = {name: load(args.truth_study/'base/compiled'/f'{name}.npz') for name in ids}
    resource = []; repeatability = []; scores = []; summaries = []
    for (count, model, precision, mode), records in sorted(grouped.items()):
        records.sort(key=lambda r:r['replicate'])
        assert [r['replicate'] for r in records] == [1,2]
        paths = [root/r['relative_output'] for r in records]
        a, b = [json.loads((path/'fingerprints.json').read_text()) for path in paths]
        assert len(a) == len(b) == count
        assert [x['name'] for x in a] == [x['name'] for x in b]
        exact = sum(x == y for x,y in zip(a,b))
        pdb_exact = sum((paths[0]/'pdbs'/f"{x['name']}.pdb").read_bytes() ==
                        (paths[1]/'pdbs'/f"{x['name']}.pdb").read_bytes() for x in a)
        repeatability.append({'count':count,'model':model,'precision':precision,'mode':mode,
                              'predictions':count,'exact_raw_repeat_predictions':exact,
                              'exact_pdb_repeat_predictions':pdb_exact})
        times = [r['container_command_seconds'] for r in records]
        resource.append({'count':count,'model':model,'precision':precision,'mode':mode,
            'complete_seconds_median':statistics.median(times), 'complete_seconds_range':[min(times),max(times)],
            'replicates_seconds':times,
            'cuda_allocated_peak_gib':max(r['metrics']['peak_cuda_allocated_bytes'] for r in records)/2**30,
            'cuda_reserved_peak_gib':max(r['metrics']['peak_cuda_reserved_bytes'] for r in records)/2**30,
            'rss_peak_gib':max(r['metrics']['sampled_process_tree_rss_peak_bytes'] for r in records)/2**30,
            'container_peak_gib':max(r['metrics']['container_lifetime_memory_peak_bytes'] for r in records)/2**30,
            'compiler_graphs':[r['metrics']['dynamo_counters'].get('stats',{}).get('unique_graphs',0) for r in records],
            'evidence_writing_seconds':[r['evidence_writing_seconds'] for r in records]})
        first = paths[args.replicate-1]
        reference = root/f'n{count}'/model/f'eager_{precision}'/f'run{args.replicate}'
        compiler = root/f'n{count}'/model/f'compiled_{precision}'/f'run{args.replicate}'
        reference_batches = json.loads((reference/'batching.json').read_text())['batches']
        for path in paths:
            assert json.loads((path/'batching.json').read_text())['batches'] == reference_batches
        rows = []
        for name in ids:
            actual = load(first/'raw'/f'{name}.npz')
            control = load(reference/'raw'/f'{name}.npz')
            target = truth[name]
            assert np.array_equal(actual['heavy'], target['heavy'])
            assert np.array_equal(actual['masks'], control['masks'])
            assert np.isfinite(actual['positions']).all()
            numerical = compare(actual['positions'], control['positions'], actual['masks'], control['masks'],
                                actual['heavy'], target['regions'])
            accuracy = compare(actual['positions'], target['truth_positions'], actual['masks'], target['truth_masks'],
                               actual['heavy'], target['regions'], False)
            control_accuracy = compare(control['positions'], target['truth_positions'], control['masks'], target['truth_masks'],
                                       actual['heavy'], target['regions'], False)
            row = {'count':count,'model':model,'precision':precision,'mode':mode,'target':name,
                   'replicate':args.replicate,
                   'comparison_reference':f'eager_{precision}', 'numerical':short(numerical),
                   'accuracy':short(accuracy), 'eager_accuracy':short(control_accuracy)}
            valid = actual['masks'].astype(bool) & control['masks'].astype(bool)
            displacement = np.linalg.norm(actual['positions'].astype(float)-control['positions'].astype(float),axis=-1)
            displacement[~valid] = -1
            i,atom = np.unravel_index(displacement.argmax(),displacement.shape)
            row['largest_atom'] = {'target':name,'input_chain':'H' if actual['heavy'][i] else 'L',
                'input_chain_residue':int(i+1 if actual['heavy'][i] else i-int(actual['heavy'].sum())+1),
                'atom':ATOM_NAMES[atom],'displacement':float(displacement[i,atom])}
            if len(actual['confidence_logits']):
                delta = confidence(actual['confidence_logits'])-confidence(control['confidence_logits'])
                row['confidence_max_absolute_change'] = float(abs(delta).max())
            if mode == 'combined':
                c = load(compiler/'raw'/f'{name}.npz')
                row['compiler_comparison'] = short(compare(actual['positions'], c['positions'], actual['masks'], c['masks'],
                                                          actual['heavy'],target['regions'],False))
                ca = compare(c['positions'],target['truth_positions'],c['masks'],target['truth_masks'],
                             actual['heavy'],target['regions'],False)
                row['compiler_accuracy'] = short(ca)
                diff = np.linalg.norm(actual['positions'].astype(float)-c['positions'].astype(float),axis=-1)
                diff[~valid] = -1
                ri,ai = np.unravel_index(diff.argmax(),diff.shape)
                row['largest_atom_vs_compiler'] = {'target':name,
                    'chain':'H' if actual['heavy'][ri] else 'L',
                    'input_chain_residue':int(ri+1 if actual['heavy'][ri] else ri-int(actual['heavy'].sum())+1),
                    'atom':ATOM_NAMES[ai],'displacement':float(diff[ri,ai])}
            if args.replicate == 2:
                previous = load(paths[0]/'raw'/f'{name}.npz')
                row['fresh_process_repeat_comparison'] = short(compare(actual['positions'],previous['positions'],
                    actual['masks'],previous['masks'],actual['heavy'],target['regions'],False))
            rows.append(row)
        scores.extend(rows)
        h3 = np.array([r['accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd'] for r in rows])
        eager_h3 = np.array([r['eager_accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd'] for r in rows])
        value = {'count':count,'model':model,'precision':precision,'mode':mode,'experimental_targets':len(rows),
            'replicate':args.replicate,
            'h3_mean_backbone_rmsd':float(h3.mean()),'eager_h3_mean_backbone_rmsd':float(eager_h3.mean()),
            'h3_max_accuracy_change_vs_eager':float(abs(h3-eager_h3).max()),
            'max_raw_ca_displacement_vs_eager':max(r['numerical']['raw_ca_max'] for r in rows),
            'largest_atom_vs_eager':max((r['largest_atom'] for r in rows),key=lambda x:x['displacement']),
            'exact_raw_first_occurrences_vs_eager':sum(np.array_equal(load(first/'raw'/f'{n}.npz')['positions'],
                load(reference/'raw'/f'{n}.npz')['positions']) for n in ids),
            'max_ca_lddt_accuracy_change_vs_eager':max(abs(r['accuracy']['ca_lddt']-r['eager_accuracy']['ca_lddt']) for r in rows)}
        if mode == 'combined':
            c = np.array([r['compiler_accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd'] for r in rows])
            value.update(h3_max_accuracy_change_vs_compiler=float(abs(h3-c).max()),
                         compiler_h3_mean_backbone_rmsd=float(c.mean()),
                         max_raw_atom_displacement_vs_compiler=max(r['compiler_comparison']['raw_atom_max'] for r in rows),
                         max_raw_ca_displacement_vs_compiler=max(r['compiler_comparison']['raw_ca_max'] for r in rows),
                         largest_atom_vs_compiler=max((r['largest_atom_vs_compiler'] for r in rows),key=lambda x:x['displacement']),
                         h3_closer_than_compiler=int(((h3-c)<-1e-10).sum()),
                         h3_further_than_compiler=int(((h3-c)>1e-10).sum()))
        if args.replicate == 2:
            value['fresh_repeat_max_raw_atom_displacement'] = max(r['fresh_process_repeat_comparison']['raw_atom_max'] for r in rows)
            value['fresh_repeat_max_raw_ca_displacement'] = max(r['fresh_process_repeat_comparison']['raw_ca_max'] for r in rows)
        summaries.append(value)
        print('Scored',count,model,precision,mode,flush=True)
    report = {'resources':resource, 'accuracy':summaries,'repeatability':repeatability,
              'scope':'100 distinct experimental targets per job; score the first occurrence of each. 1000 rows are ten repeats, not 1000 independent antibodies.',
              'validation':{'completed_jobs':48,'written_pdbs':sum(r['count'] for r in measurements),
                            'matched_batch_shapes':True,'raw_saved_predictions':48*100}}
    suffix = '_rep2' if args.replicate == 2 else ''
    (root/f'analysis{suffix}_261026.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (root/f'per_target_scores{suffix}_261026.json').write_text(json.dumps(scores,indent=2,allow_nan=False)+'\n')
    print('Complete-pipeline structural scoring completed',flush=True)


if __name__ == '__main__':
    main()
