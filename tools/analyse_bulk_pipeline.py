"""Score cached bulk jobs, including exact writer checks and paired precision.

Uses the same published experimental targets, masks and structural metrics as
the earlier attention/compiler comparison. Repeated input rows are throughput
workloads, never additional independent experimental targets.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np

from analyse_structures import confidence, short
from structural_metrics import ATOM_NAMES, compare


def load(path):
    with np.load(path) as arrays:
        return {key: arrays[key] for key in arrays.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--truth-study', type=Path, required=True)
    parser.add_argument('--date-code', default='261026')
    parser.add_argument('--count',type=int,default=10000)
    parser.add_argument('--expected-jobs',type=int,default=32)
    parser.add_argument('--reference-label',default='current_dev')
    parser.add_argument('--output-prefix',default='bulk')
    args = parser.parse_args()
    root = args.study
    records = json.loads((root / f'measurements_{args.date_code}.json').read_text())
    assert json.loads((root / f'completed_{args.date_code}.json').read_text())['completed']
    count=args.count
    bulk = [row for row in records if row['count'] == count]
    assert len(bulk) == args.expected_jobs
    names = list(json.loads((args.truth_study / 'base/compiled/metadata_251026.json').read_text())['targets'])
    assert len(names) == 100
    truth = {name: load(args.truth_study / 'base/compiled' / f'{name}.npz') for name in names}
    groups = defaultdict(list)
    by_key = {}
    data = {}
    fingerprints = {}
    validations = []
    resources = []
    for row in bulk:
        key = row['model'], row['precision'], row['label'], row['replicate']
        assert key not in by_key
        by_key[key] = row
        groups[key[:3]].append(row)
        path = root / row['relative_output']
        metrics = row['metrics']
        assert metrics['completed'] and row['batching']['saved'] == count
        assert not row['batching']['oom_retries']
        assert metrics['allow_tf32'] == (row['precision'] == 'normal')
        assert metrics['float32_matmul_precision'] == ('highest' if row['precision'] == 'strict' else 'high')
        fp = json.loads((path / 'fingerprints.json').read_text())
        assert len(fp) == count and len({entry['name'] for entry in fp}) == count
        assert len(list((path / 'pdbs').glob('*.pdb'))) == count
        fingerprints[key] = fp
        data[key] = {name: load(path / 'raw' / f'{name}.npz') for name in names}
        for name, actual in data[key].items():
            assert np.isfinite(actual['positions']).all()
            assert np.array_equal(actual['heavy'], truth[name]['heavy'])
        if row['native_formatter']:
            assert json.loads((path / 'formatter_usage.json').read_text())['native_calls'] == count
        if row['mode'] != 'eager':
            assert metrics['packaged_cache_manifest_sha256']
        if row['mode'] == 'aot':
            assert metrics['aot_root']
            assert metrics['dynamo_counters'].get('stats',{}).get('unique_graphs',0)==0
            assert len(metrics['checkpoint_loading'][0]['aot_package_loading'])==2
        validations.append({'model':key[0], 'precision':key[1], 'label':key[2],
                            'replicate':key[3], 'written_pdbs':len(fp), 'finite_saved_raw_targets':100,
                            'native_calls':count if row['native_formatter'] else None,
                            'dynamo_counters':metrics['dynamo_counters'],
                            'execution_mode':row['mode'],
                            'packaged_cache_manifest_sha256':metrics['packaged_cache_manifest_sha256']})
    for (model, precision, label), rows in sorted(groups.items()):
        rows.sort(key=lambda row:row['replicate'])
        assert [row['replicate'] for row in rows] == [1,2]
        times = [row['container_seconds'] for row in rows]
        resources.append({'model':model, 'precision':precision, 'label':label,
            'seconds_median':statistics.median(times), 'seconds_runs':times,
            'cuda_allocated_peak_gib':max(row['metrics']['peak_cuda_allocated_bytes'] for row in rows)/2**30,
            'cuda_reserved_peak_gib':max(row['metrics']['peak_cuda_reserved_bytes'] for row in rows)/2**30,
            'process_tree_rss_peak_gib':max(row['metrics']['sampled_process_tree_rss_peak_bytes'] for row in rows)/2**30,
            'container_memory_peak_gib':max(row['metrics']['container_lifetime_memory_peak_bytes'] for row in rows)/2**30,
            'evidence_writing_seconds':[row['metrics']['raw_evidence_writing_seconds'] for row in rows],
            'images':[row['image'] for row in rows]})
    exact_writer = []
    for key, row in by_key.items():
        if key[2] != 'native_writer':
            continue
        ref_key = key[:2] + ('current_dev', key[3])
        if ref_key not in by_key:
            continue
        control = by_key[ref_key]
        assert row['batching']['batches'] == control['batching']['batches']
        a, b = fingerprints[key], fingerprints[ref_key]
        assert [x['name'] for x in a] == [x['name'] for x in b]
        raw_equal = sum(x == y for x,y in zip(a,b))
        actual_path = root / row['relative_output'] / 'pdbs'
        reference_path = root / control['relative_output'] / 'pdbs'
        pdb_equal = sum((actual_path / f"{entry['name']}.pdb").read_bytes() ==
                        (reference_path / f"{entry['name']}.pdb").read_bytes() for entry in a)
        exact_writer.append({'model':key[0], 'precision':key[1], 'replicate':key[3],
                             'count':count, 'raw_exact':raw_equal, 'pdb_exact':pdb_equal})
        assert raw_equal == pdb_equal == count
        print('Writer exact', key, flush=True)
    repeatability = []
    for key in groups:
        a, b = fingerprints[key + (1,)], fingerprints[key + (2,)]
        assert [x['name'] for x in a] == [x['name'] for x in b]
        repeatability.append({'model':key[0], 'precision':key[1], 'label':key[2],
                              'raw_exact':sum(x == y for x,y in zip(a,b)), 'count':count})
    accuracy = {}
    target_accuracy = []
    for key, actuals in data.items():
        scores = {}
        for name, actual in actuals.items():
            t = truth[name]
            result = short(compare(actual['positions'], t['truth_positions'], actual['masks'],
                t['truth_masks'], actual['heavy'], t['regions'], False))
            scores[name] = result
            target_accuracy.append({'model':key[0], 'precision':key[1], 'label':key[2],
                                     'replicate':key[3], 'target':name, 'accuracy':result})
        accuracy[key] = scores
        print('Experimental accuracy', key, flush=True)
    comparisons = []
    per_target = []
    pairs = []
    for key in by_key:
        if key[2] != args.reference_label:
            pairs.append((key, key[:2] + (args.reference_label,key[3]),
                          'vs_matched_dev' if args.reference_label=='current_dev' else 'vs_matched_reference'))
        if key[2] == 'cached_combined_native':
            ref_key=key[:2]+('cached_compiled_native',key[3])
            if ref_key in by_key:
                pairs.append((key,ref_key,'vs_compilation_alone'))
        if key[1] == 'normal':
            ref_key=(key[0],'strict',key[2],key[3])
            if ref_key in by_key:
                pairs.append((key,ref_key,'tf32_vs_strict_fp32'))
            elif key[2]=='cached_combined_native':
                ref_key=(key[0],'strict','cached_compiled_native',key[3])
                if ref_key in by_key:
                    pairs.append((key,ref_key,'selected_profiles_precision_and_attention_differ'))
    for key, ref_key, reason in pairs:
        assert by_key[key]['batching']['batches'] == by_key[ref_key]['batching']['batches']
        rows = []
        for name in names:
            actual, control, t = data[key][name], data[ref_key][name], truth[name]
            assert np.array_equal(actual['masks'],control['masks'])
            numerical = short(compare(actual['positions'],control['positions'],actual['masks'],
                control['masks'],actual['heavy'],t['regions'],False))
            valid = actual['masks'].astype(bool) & control['masks'].astype(bool)
            displacement = np.linalg.norm(actual['positions'].astype(float)-control['positions'].astype(float),axis=-1)
            displacement[~valid] = -1
            ri, ai = np.unravel_index(displacement.argmax(),displacement.shape)
            ac, rc = accuracy[key][name], accuracy[ref_key][name]
            item = {'model':key[0], 'precision':key[1], 'label':key[2], 'replicate':key[3],
                'reference_label':ref_key[2], 'reference_precision':ref_key[1], 'comparison':reason,
                'target':name, 'numerical':numerical, 'accuracy':ac, 'reference_accuracy':rc,
                'coordinates_exact':bool(np.array_equal(actual['positions'],control['positions'])),
                'largest_atom':{'target':name,'chain':'H' if actual['heavy'][ri] else 'L',
                    'chain_residue':int(ri+1 if actual['heavy'][ri] else ri-int(actual['heavy'].sum())+1),
                    'atom':ATOM_NAMES[ai], 'displacement':float(displacement[ri,ai])}}
            if len(actual['confidence_logits']):
                item['max_confidence_difference'] = float(abs(confidence(actual['confidence_logits'])-
                                                            confidence(control['confidence_logits'])).max())
            rows.append(item)
        per_target.extend(rows)
        ah3=np.array([x['accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd'] for x in rows])
        rh3=np.array([x['reference_accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd'] for x in rows])
        lddt=np.array([x['accuracy']['ca_lddt']-x['reference_accuracy']['ca_lddt'] for x in rows])
        comparisons.append({'model':key[0], 'precision':key[1], 'label':key[2], 'replicate':key[3],
            'reference_label':ref_key[2], 'reference_precision':ref_key[1], 'comparison':reason,
            'targets':100, 'h3_mean_rmsd':float(ah3.mean()), 'reference_h3_mean_rmsd':float(rh3.mean()),
            'h3_mean_delta':float((ah3-rh3).mean()), 'h3_max_absolute_delta':float(abs(ah3-rh3).max()),
            'h3_improved_over_0_01_angstrom':int(((ah3-rh3)<-.01).sum()),
            'h3_worsened_over_0_01_angstrom':int(((ah3-rh3)>.01).sum()),
            'ca_lddt_mean_delta_percentage_points':float(lddt.mean()),
            'ca_lddt_max_absolute_delta_percentage_points':float(abs(lddt).max()),
            'max_raw_ca_displacement':max(x['numerical']['raw_ca_max'] for x in rows),
            'max_aligned_backbone_rmsd':max(x['numerical']['aligned_backbone_rmsd'] for x in rows),
            'largest_atom':max((x['largest_atom'] for x in rows),key=lambda x:x['displacement']),
            'coordinates_exact':sum(x['coordinates_exact'] for x in rows),
            'max_confidence_difference':max((x.get('max_confidence_difference',0) for x in rows))})
        print('Compared',key,reason,flush=True)
    result={'scope':f'{len(bulk)} fresh {count:,}-input jobs; 100 distinct experimental targets repeated {count//100} times per job. '
                    'Two timing replicates are not independent biological samples. Experimental accuracy scores '
                    'use the first occurrence of each distinct target, in both fresh processes.',
            'metrics':'CA lDDT is in percent, deltas in percentage points; distances in Angstrom. '
                      'CDR-H3 backbone RMSD follows heavy-chain framework CA alignment. '
                      '0.01 Angstrom is a descriptive reporting band, not an accuracy acceptance threshold.',
            'resources':resources,'comparisons':comparisons,'exact_writer':exact_writer,
            'repeatability':repeatability,'validation':validations}
    (root/f'{args.output_prefix}_analysis_{args.date_code}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (root/f'{args.output_prefix}_per_target_{args.date_code}.json').write_text(json.dumps(per_target,indent=2,allow_nan=False)+'\n')
    (root/f'{args.output_prefix}_experimental_accuracy_{args.date_code}.json').write_text(json.dumps(target_accuracy,indent=2,allow_nan=False)+'\n')
    print('Bulk scoring complete',flush=True)


if __name__ == '__main__':
    main()
