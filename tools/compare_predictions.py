"""Compare raw antibody predictions, with explicit identity and atom-mask checks."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference, candidate):
    with np.load(reference) as a, np.load(candidate) as b:
        for key in ['aatype', 'is_heavy', 'masks']:
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f'{key} mismatch: {reference}, {candidate}')
        mask = a['masks'].astype(bool)
        x, y = a['positions'].astype(float), b['positions'].astype(float)
        if x.shape != y.shape or not np.isfinite(x[mask]).all() or not np.isfinite(y[mask]).all():
            raise ValueError('Invalid coordinate shape or values')
        delta = np.abs(x[mask] - y[mask])
        ca_mask = mask[:, 1]
        ca_x, ca_y = x[ca_mask, 1], y[ca_mask, 1]
        x0, y0 = ca_x - ca_x.mean(0), ca_y - ca_y.mean(0)
        u, _, v = np.linalg.svd(x0.T @ y0)
        d = np.eye(3)
        d[-1, -1] = np.linalg.det(u @ v)
        rmsd = np.sqrt(np.mean(np.sum((x0 @ u @ d @ v - y0) ** 2, axis=1)))
        dx = np.linalg.norm(ca_x[:, None] - ca_x[None, :], axis=-1)
        dy = np.linalg.norm(ca_y[:, None] - ca_y[None, :], axis=-1)
        neighbours = (dx < 15) & ~np.eye(len(dx), dtype=bool)
        error = np.abs(dx - dy)
        scores = sum((error < t).astype(float) for t in [.5, 1., 2., 4.]) / 4
        counts = neighbours.sum(axis=1)
        valid = counts > 0
        # Equal weight per residue with neighbours, rather than per contact pair.
        lddt = ((scores * neighbours).sum(axis=1)[valid] / counts[valid]).mean()
        pa, pb = a['confidence'].reshape(-1), b['confidence'].reshape(-1)
        if pa.shape != pb.shape:
            raise ValueError('Confidence shape mismatch')
        if not np.isfinite(pa).all() or not np.isfinite(pb).all():
            raise ValueError('Nonfinite confidence')
        # StructureModule's "plddt" output contains 50-bin logits, not scores.
        score_difference = None
        if len(pa):
            def plddt(logits):
                logits = logits.reshape(-1, 50).astype(float)
                probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
                probabilities /= probabilities.sum(axis=-1, keepdims=True)
                return probabilities @ np.arange(1, 101, 2)
            score_difference = float(np.max(np.abs(plddt(pa) - plddt(pb))))
        return {
            'raw_coordinates_exact': bool(np.array_equal(a['positions'], b['positions'])),
            'confidence_exact': bool(np.array_equal(pa, pb)),
            'max_atom_coordinate_difference_angstrom': float(delta.max()),
            'max_ca_coordinate_difference_angstrom': float(np.abs(ca_x - ca_y).max()),
            'ca_aligned_rmsd_angstrom': float(rmsd), 'ca_lddt_agreement': float(lddt),
            'max_confidence_difference': float(np.max(np.abs(pa - pb))) if len(pa) else None,
            'max_plddt_score_difference': score_difference,
            'pdb_bytes_exact': reference.with_suffix('.pdb').read_bytes() == candidate.with_suffix('.pdb').read_bytes(),
            'residues': len(x), 'atoms': int(mask.sum()),
        }


def compare_folders(reference, candidate):
    rows = []
    expected = sorted(p.name for p in (reference / 'run1').glob('*.npz'))
    for rep in [1, 2]:
        if sorted(p.name for p in (candidate / f'run{rep}').glob('*.npz')) != expected:
            raise ValueError('Missing/extra candidate outputs')
        for name in expected:
            rows.append({'replicate': rep, 'antibody': name, **compare(reference / f'run{rep}' / name,
                                                                      candidate / f'run{rep}' / name)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = compare_folders(args.reference, args.candidate)
    args.output.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps({'comparisons': len(rows), 'raw_exact': sum(r['raw_coordinates_exact'] for r in rows),
                      'max_ca_coordinate_difference_angstrom': max(r['max_ca_coordinate_difference_angstrom'] for r in rows),
                      'minimum_ca_lddt': min(r['ca_lddt_agreement'] for r in rows)}, indent=2))


if __name__ == '__main__':
    main()
