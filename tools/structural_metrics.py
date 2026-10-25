"""Explicit, paired structure metrics; distances in Angstrom, lDDT in percent.

CDR regions come from the upstream experimental dataset, not substring matching.
Experimental accuracy uses backbone atoms, avoiding side-chain naming symmetry.
The clash screen is a stated heavy-atom geometric proxy, not MolProbity.
"""
import numpy as np
from scipy.spatial import cKDTree

BACKBONE = [0, 1, 2, 4]  # atom37 N, CA, C, O
CDRS = ['cdrh1', 'cdrh2', 'cdrh3', 'cdrl1', 'cdrl2', 'cdrl3']
ATOM_NAMES = ['N','CA','C','CB','O','CG','CG1','CG2','OG','OG1','SG','CD','CD1',
              'CD2','ND1','ND2','OD1','OD2','SD','CE','CE1','CE2','CE3','NE',
              'NE1','NE2','OE1','OE2','CH2','NH1','NH2','OH','CZ','CZ2','CZ3',
              'NZ','OXT']


def fit(moving, target):
    moving, target = np.asarray(moving, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if len(moving) < 3:
        raise ValueError('Need at least three atoms to superpose.')
    cm, ct = moving.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((moving-cm).T @ (target-ct))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    return rotation, ct-cm @ rotation


def rmsd(a, b):
    return float(np.sqrt(np.mean(np.sum((a-b)**2, axis=-1)))) if len(a) else None


def ca_lddt(moving, target, valid, cutoff=15.):
    moving, target = np.asarray(moving, dtype=np.float64), np.asarray(target, dtype=np.float64)
    dt = np.linalg.norm(target[:, None]-target[None, :], axis=-1)
    dm = np.linalg.norm(moving[:, None]-moving[None, :], axis=-1)
    pairs = valid[:, None] & valid[None, :] & (dt < cutoff)
    np.fill_diagonal(pairs, False)
    scores = sum(np.abs(dm-dt) < threshold for threshold in (.5, 1., 2., 4.))/4.
    counts = pairs.sum(1)
    per_residue = np.divide((scores*pairs).sum(1), counts,
                           out=np.full(len(valid), np.nan), where=counts > 0)*100
    return {'score': float(scores[pairs].mean()*100) if pairs.any() else None,
            'per_residue': [float(x) if np.isfinite(x) else None for x in per_residue],
            'directed_pair_count': int(pairs.sum())}


def geometry(positions, masks, heavy):
    p = np.asarray(positions, dtype=np.float64)
    bonds = {}
    for name, a, b in [('N_CA',0,1), ('CA_C',1,2), ('C_O',2,4)]:
        valid = masks[:, a] & masks[:, b]
        d = np.linalg.norm(p[valid, a]-p[valid, b], axis=-1)
        bonds[name] = {'count':len(d), 'mean':float(d.mean()), 'min':float(d.min()), 'max':float(d.max())}
    valid = (heavy[:-1] == heavy[1:]) & masks[:-1,2] & masks[1:,0]
    peptide = np.linalg.norm(p[:-1,2]-p[1:,0], axis=-1)[valid]
    bonds['peptide_C_N'] = {'count':len(peptide), 'mean':float(peptide.mean()),
        'min':float(peptide.min()), 'max':float(peptide.max()),
        'outside_1_1_to_1_6':int(((peptide < 1.1) | (peptide > 1.6)).sum())}
    ri, ai = np.where(masks)
    coordinates = p[ri, ai]
    pairs = cKDTree(coordinates).query_pairs(3.1, output_type='ndarray')
    if not len(pairs):
        return {'bonds':bonds, 'severe_clashes':0, 'clashing_residues':[]}
    i, j = pairs.T
    distance = np.linalg.norm(coordinates[i]-coordinates[j], axis=-1)
    elements = np.array([ATOM_NAMES[a][0] for a in ai])
    radii = np.array([{'C':1.7,'N':1.55,'O':1.52,'S':1.8}[e] for e in elements])
    # Exclude intra-residue and immediate same-chain neighbours, and S-S bonds.
    excluded = (ri[i] == ri[j]) | ((np.abs(ri[i]-ri[j]) == 1) & (heavy[ri[i]] == heavy[ri[j]]))
    excluded |= (ai[i] == 10) & (ai[j] == 10) & (distance < 2.3)
    clash = ~excluded & (distance < radii[i]+radii[j]-.6)
    return {'bonds':bonds, 'severe_clashes':int(clash.sum()),
            'clashing_residues':sorted(set(ri[i[clash]].tolist()+ri[j[clash]].tolist()))}


def compare(moving, target, moving_mask, target_mask, heavy, regions, include_geometry=True):
    x, y = np.asarray(moving, dtype=np.float64), np.asarray(target, dtype=np.float64)
    mask = np.asarray(moving_mask, bool) & np.asarray(target_mask, bool)
    heavy, regions = np.asarray(heavy, bool), np.asarray(regions)
    ca = mask[:,1]
    delta = np.linalg.norm(x-y, axis=-1)
    rotation, translation = fit(x[ca,1], y[ca,1])
    aligned = x @ rotation+translation
    bbmask = mask[:, BACKBONE]
    bbx, bby = aligned[:, BACKBONE], y[:, BACKBONE]
    result = {'matched_ca_count':int(ca.sum()), 'matched_atom_count':int(mask.sum()),
        'raw_ca_max':float(delta[ca,1].max()), 'raw_ca_rmsd':rmsd(x[ca,1], y[ca,1]),
        'raw_atom_max':float(delta[mask].max()),
        'max_ca_coordinate_component':float(np.abs(x[ca,1]-y[ca,1]).max()),
        'aligned_ca_rmsd':rmsd(aligned[ca,1], y[ca,1]),
        'aligned_backbone_rmsd':rmsd(bbx[bbmask], bby[bbmask]),
        'per_residue_aligned_ca_distance':np.linalg.norm(aligned[:,1]-y[:,1],axis=-1).tolist(),
        'ca_lddt':ca_lddt(x[:,1],y[:,1],ca), 'regions':{}}
    for chain, prefix in [(True,'h'), (False,'l')]:
        framework = (heavy == chain) & np.char.startswith(regions, 'fw') & ca
        rot, shift = fit(x[framework,1], y[framework,1])
        chain_aligned = x @ rot+shift
        entries = [('fw'+prefix, (heavy == chain) & np.char.startswith(regions,'fw'))]
        entries += [(cdr, regions == cdr) for cdr in CDRS if cdr[3] == prefix]
        for name, residues in entries:
            valid_ca = residues & ca
            selected = residues[:,None] & bbmask
            a, b = x[:,BACKBONE][selected], y[:,BACKBONE][selected]
            if len(a) < 3:
                continue
            own_rot, own_shift = fit(a,b)
            local_lddt = [v for v,valid in zip(result['ca_lddt']['per_residue'],valid_ca) if valid and v is not None]
            result['regions'][name] = {'residue_count':int(residues.sum()),
                'matched_ca_count':int(valid_ca.sum()),'matched_backbone_atoms':int(selected.sum()),
                'framework_aligned_ca_rmsd':rmsd(chain_aligned[valid_ca,1],y[valid_ca,1]),
                'framework_aligned_backbone_rmsd':rmsd(chain_aligned[:,BACKBONE][selected],b),
                'self_aligned_backbone_rmsd':rmsd(a @ own_rot+own_shift,b),
                'mean_local_ca_lddt':float(np.mean(local_lddt)) if local_lddt else None}
    if include_geometry:
        result['geometry'] = geometry(x,np.asarray(moving_mask,bool),heavy)
    return result
