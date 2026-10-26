"""Score paired installed-CLI outputs without changing or rerunning predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from abodybuilder3.openfold.np import residue_constants as rc
from structural_metrics import compare, fit, ATOM_NAMES


def confidence(logits):
    if not len(logits):
        return None
    values = np.asarray(logits, dtype=np.float64)
    probabilities = np.exp(values - values.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    return probabilities @ np.arange(1, 101, 2)


def dihedral(points):
    a, b, c, d = np.asarray(points, dtype=np.float64)
    axis = c - b
    axis /= np.linalg.norm(axis)
    v = a - b - np.dot(a - b, axis) * axis
    w = d - c - np.dot(d - c, axis) * axis
    return np.degrees(np.arctan2(np.dot(np.cross(axis, v), w), np.dot(v, w)))


def sidechains(prediction, truth, data):
    positions = prediction['positions'].astype(np.float64)
    pred14 = positions[np.arange(len(positions))[:, None], data['residx_atom14_to_atom37'].numpy()]
    heavy, regions = truth['heavy'], truth['regions']
    for chain in (True, False):
        framework = (heavy == chain) & np.char.startswith(regions, 'fw') & truth['masks'][:, 1].astype(bool)
        rotation, translation = fit(positions[framework, 1], truth['positions'][framework, 1])
        pred14[heavy == chain] = pred14[heavy == chain] @ rotation + translation
    observed = data['atom14_gt_exists'].numpy().copy()
    observed[:, :4] = 0
    gt, alt = data['atom14_gt_positions'].numpy(), data['atom14_alt_gt_positions'].numpy()
    first = (((pred14 - gt) ** 2).sum(-1) * observed).sum(-1)
    second = (((pred14 - alt) ** 2).sum(-1) * observed).sum(-1)
    error = np.minimum(first, second)
    h3 = regions == 'cdrh3'
    return {'sidechain_rmsd': float(np.sqrt(error.sum() / observed.sum())),
            'h3_sidechain_rmsd': float(np.sqrt(error[h3].sum() / observed[h3].sum()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    root = args.study
    records = json.loads((root / 'accuracy_measurements_261026.json').read_text())
    results = []
    for model in ('base', 'plddt'):
        for rep in (1, 2):
            groups = {row['execution']: root / row['output'] for row in records
                      if row['model'] == model and row['replicate'] == rep}
            left, right = groups['eager'] / 'raw', groups['aot'] / 'raw'
            if {p.name for p in left.glob('*.npz')} != {p.name for p in right.glob('*.npz')}:
                raise ValueError('Paired raw identities differ.')
            for path in sorted(left.glob('*.npz')):
                name = path.stem
                with np.load(path) as data:
                    reference = {key: value.copy() for key, value in data.items()}
                with np.load(right / path.name) as data:
                    actual = {key: value.copy() for key, value in data.items()}
                with np.load(root / 'independent_validation/truth' / path.name) as data:
                    truth = {key: value.copy() for key, value in data.items()}
                if not np.array_equal(reference['masks'], actual['masks']):
                    raise ValueError(f'Atom masks changed: {name}')
                heavy, regions = truth['heavy'], truth['regions']
                numerical = compare(actual['positions'], reference['positions'], actual['masks'], reference['masks'], heavy, regions)
                expected_accuracy = compare(reference['positions'], truth['positions'], reference['masks'], truth['masks'], heavy, regions, False)
                actual_accuracy = compare(actual['positions'], truth['positions'], actual['masks'], truth['masks'], heavy, regions, False)
                old_conf, new_conf = confidence(reference['confidence_logits']), confidence(actual['confidence_logits'])
                max_chi = 0.
                for i, amino_acid in enumerate(truth['aatype']):
                    one = (rc.restypes + ['X'])[int(amino_acid)]
                    for atoms in rc.chi_angles_atoms.get(rc.restype_1to3.get(one, 'UNK'), []):
                        indices = [ATOM_NAMES.index(atom) for atom in atoms]
                        if actual['masks'][i, indices].all():
                            first = dihedral(reference['positions'][i, indices])
                            second = dihedral(actual['positions'][i, indices])
                            max_chi = max(max_chi, abs((first - second + 180) % 360 - 180))
                tensor = torch.load(root / 'independent_validation' / (name + '.pt'), map_location='cpu', weights_only=False)
                old_side, new_side = sidechains(reference, truth, tensor), sidechains(actual, truth, tensor)
                row = {'model': model, 'replicate': rep, 'name': name, 'pdb_entry': name.split('_')[0],
                       'length': len(heavy), 'h3_length': int((regions == 'cdrh3').sum()),
                       'aligned_CA_RMSD_A': numerical['aligned_ca_rmsd'],
                       'max_CA_displacement_A': numerical['raw_ca_max'],
                       'max_atom_displacement_A': numerical['raw_atom_max'],
                       'max_chi_change_degrees': max_chi,
                       'reference_CA_RMSD_to_experiment_A': expected_accuracy['aligned_ca_rmsd'],
                       'aot_CA_RMSD_to_experiment_A': actual_accuracy['aligned_ca_rmsd'],
                       'reference_H3_RMSD_to_experiment_A': expected_accuracy['regions']['cdrh3']['framework_aligned_backbone_rmsd'],
                       'aot_H3_RMSD_to_experiment_A': actual_accuracy['regions']['cdrh3']['framework_aligned_backbone_rmsd'],
                       'reference_CA_lddt_percent': expected_accuracy['ca_lddt']['score'],
                       'aot_CA_lddt_percent': actual_accuracy['ca_lddt']['score'],
                       'reference_sidechain_RMSD_A': old_side['sidechain_rmsd'],
                       'aot_sidechain_RMSD_A': new_side['sidechain_rmsd'],
                       'reference_H3_sidechain_RMSD_A': old_side['h3_sidechain_rmsd'],
                       'aot_H3_sidechain_RMSD_A': new_side['h3_sidechain_rmsd'],
                       'reference_clash_proxy': compare(reference['positions'], reference['positions'], reference['masks'], reference['masks'], heavy, regions)['geometry']['severe_clashes'],
                       'aot_clash_proxy': numerical['geometry']['severe_clashes'],
                       'max_confidence_change_points': None if old_conf is None else float(np.abs(new_conf - old_conf).max())}
                if row['max_CA_displacement_A'] > .001 or row['aligned_CA_RMSD_A'] > .0001 or row['max_atom_displacement_A'] > .01:
                    raise ValueError(f'Predeclared numerical gate failed: {row}')
                results.append(row)
    summaries = []
    for model in ('base', 'plddt'):
        # A target contributes once; fresh repetitions do not double its weight.
        rows = [row for row in results if row['model'] == model and row['replicate'] == 1]
        summary = {'model': model, 'targets': len(rows), 'pdb_entries': len({row['pdb_entry'] for row in rows}),
                   'mean_aligned_CA_RMSD_A': float(np.mean([row['aligned_CA_RMSD_A'] for row in rows])),
                   'max_CA_displacement_A': max(row['max_CA_displacement_A'] for row in rows),
                   'max_atom_displacement_A': max(row['max_atom_displacement_A'] for row in rows),
                   'max_chi_change_degrees': max(row['max_chi_change_degrees'] for row in rows)}
        for metric in ('CA_RMSD_to_experiment_A', 'H3_RMSD_to_experiment_A', 'CA_lddt_percent',
                       'sidechain_RMSD_A', 'H3_sidechain_RMSD_A', 'clash_proxy'):
            baseline = np.array([row['reference_' + metric] for row in rows])
            actual = np.array([row['aot_' + metric] for row in rows])
            summary[metric] = {'reference_mean': float(baseline.mean()), 'aot_mean': float(actual.mean()),
                               'mean_paired_change': float((actual - baseline).mean()),
                               'max_absolute_paired_change': float(np.abs(actual - baseline).max())}
        summary['max_confidence_change_points'] = max((row['max_confidence_change_points'] for row in rows
                                                       if row['max_confidence_change_points'] is not None), default=None)
        for execution in ('eager', 'aot'):
            paths = [root / row['output'] for row in records if row['model'] == model and row['execution'] == execution]
            fp = [json.loads((path / 'fingerprints_261026.json').read_text()) for path in paths]
            summary[execution + '_raw_repeats_exact'] = fp[0] == fp[1]
        summaries.append(summary)
    (root / 'paired_accuracy_261026.json').write_text(json.dumps({'scope': 'All upstream validation targets; independent of implementation tuning, not an unseen model test.',
        'summaries': summaries, 'rows': results}, indent=2, allow_nan=False) + '\n')
    with (root / 'paired_accuracy_261026.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(results)
    print(json.dumps({'completed': True, 'summaries': summaries}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
