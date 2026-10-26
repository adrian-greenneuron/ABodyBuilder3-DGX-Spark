"""Native PDB formatting with the original writer as an explicit fallback."""
import numpy as np
from abodybuilder3.openfold.np import residue_constants
from reference.format_pdb import to_pdb_fast as original_to_pdb
from . import _pdb_native

RESIDUES=[residue_constants.restype_1to3.get(a,'UNK') for a in residue_constants.restypes+['X']]
ATOMS=residue_constants.atom_types


def to_pdb_fast(prot):
    arrays=[prot.aatype,prot.atom_positions,prot.atom_mask,prot.residue_index,prot.b_factors]
    if not all(isinstance(a,np.ndarray) for a in arrays):
        return original_to_pdb(prot)
    n=len(prot.aatype)
    if (prot.aatype.dtype.kind not in 'iu' or prot.residue_index.dtype.kind!='i' or
            prot.atom_positions.dtype.kind not in 'fi' or prot.b_factors.dtype.kind not in 'fi' or
            prot.atom_mask.dtype.kind not in 'fib' or
            prot.aatype.shape!=(n,) or prot.residue_index.shape!=(n,) or
            prot.atom_positions.shape!=(n,37,3) or prot.atom_mask.shape!=(n,37) or prot.b_factors.shape!=(n,37)):
        return original_to_pdb(prot)
    if n and (prot.aatype.min() < -21 or prot.aatype.max()>20):
        return original_to_pdb(prot)
    if not (np.isfinite(prot.atom_positions).all() and np.isfinite(prot.b_factors).all()):
        return original_to_pdb(prot)
    if prot.chain_index is None:
        chains=np.empty(0,dtype=np.int64)
    else:
        if (not isinstance(prot.chain_index,np.ndarray) or prot.chain_index.dtype.kind not in 'iu' or
                prot.chain_index.shape!=(n,) or (n and (prot.chain_index.min() < -2 or prot.chain_index.max()>1))):
            return original_to_pdb(prot)
        chains=prot.chain_index
    return _pdb_native.format(prot.aatype,prot.atom_positions,prot.atom_mask,
                              prot.residue_index+1,prot.b_factors,chains,
                              str(prot.remark) if prot.remark else '',RESIDUES,ATOMS)
