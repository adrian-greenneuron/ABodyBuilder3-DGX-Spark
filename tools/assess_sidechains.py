"""Resolve larger local atom changes and symmetry-aware experimental scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from structural_metrics import fit, ATOM_NAMES
from abodybuilder3.openfold.np import residue_constants as rc


def dihedral(points):
    a,b,c,d=np.asarray(points,dtype=np.float64)
    axis=c-b;axis/=np.linalg.norm(axis)
    v=(a-b)-np.dot(a-b,axis)*axis
    w=(d-c)-np.dot(d-c,axis)*axis
    return np.degrees(np.arctan2(np.dot(np.cross(axis,v),w),np.dot(v,w)))


def angle_difference(a,b):
    return abs((a-b+180)%360-180)


def symmetry_error(pred,original,alternate,mask):
    a=np.sum((pred-original)**2,axis=-1)*mask
    b=np.sum((pred-alternate)**2,axis=-1)*mask
    return np.minimum(a.sum(-1),b.sum(-1))


def clash_map(p,mask,heavy):
    ri,ai=np.where(mask);coords=p[ri,ai].astype(np.float64)
    pairs=cKDTree(coords).query_pairs(3.1,output_type='ndarray')
    if not len(pairs):return {}
    i,j=pairs.T;distance=np.linalg.norm(coords[i]-coords[j],axis=-1)
    radii=np.array([{'C':1.7,'N':1.55,'O':1.52,'S':1.8}[ATOM_NAMES[a][0]] for a in ai])
    excluded=(ri[i]==ri[j]) | ((abs(ri[i]-ri[j])==1)&(heavy[ri[i]]==heavy[ri[j]]))
    excluded|=(ai[i]==10)&(ai[j]==10)&(distance<2.3)
    overlap=radii[i]+radii[j]-distance
    selected=np.flatnonzero(~excluded&(overlap>.6))
    return {(int(ri[i[k]]),int(ai[i[k]]),int(ri[j[k]]),int(ai[j[k]])):float(overlap[k]) for k in selected}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--reference',default='accepted')
    p.add_argument('--strict-reference',default='strict_reference')
    p.add_argument('--matched-precision',action='store_true')
    p.add_argument('--variants',nargs='+',default=['accepted','attention','point_ieee','compiled','strict_reference','strict_attention','strict_compiled'])
    args=p.parse_args();root=args.study;results=[];target_rows=[]
    data=args.data or root/'data'
    ids=json.loads((data/'test_ids_251026.json').read_text())
    truth={name:torch.load(data/'experimental'/f'{name}.pt',weights_only=False,map_location='cpu') for name in ids}
    baseline_residue_errors={}
    for model in ('base','plddt'):
        for variant in args.variants:
            reference=args.strict_reference if args.matched_precision and variant.startswith('strict_') else args.reference
            folder=root/model/variant
            if not (folder/'metadata_251026.json').exists():continue
            largest=[];chi_changes=[];new_clashes=[];rows=[];residue_accuracy_changes=[]
            for name in ids:
                d=np.load(folder/f'{name}.npz');base=np.load(root/model/reference/f'{name}.npz');t=truth[name]
                pos=d['positions'];bp=base['positions'];mask=d['masks'];heavy=d['heavy'];regions=d['regions']
                valid=mask&base['masks'];difference=np.linalg.norm(pos.astype(float)-bp,axis=-1)
                difference[~valid]=0;i,a=np.unravel_index(difference.argmax(),difference.shape)
                h=int(heavy.sum());sequence=t['sequence']
                largest.append({'target':name,'chain':'H' if heavy[i] else 'L','input_chain_residue':int(i+1 if heavy[i] else i-h+1),
                    'amino_acid':sequence[i],'atom':ATOM_NAMES[a],'region':str(regions[i]),
                    'raw_displacement':float(difference[i,a])})
                # Match original/alternative names jointly within each residue.
                atom14_index=t['residx_atom14_to_atom37'].numpy()
                predicted14=pos[np.arange(len(pos))[:,None],atom14_index].astype(float)
                gt=t['atom14_gt_positions'].numpy();alt=t['atom14_alt_gt_positions'].numpy()
                observed=(t['atom14_gt_exists'].numpy()>0)&(t['atom14_alt_gt_exists'].numpy()>0)
                observed[:,:4]=False  # N/CA/C/O; CB and beyond are side chain.
                for chain in (True,False):
                    framework=(heavy==chain)&np.char.startswith(regions,'fw')&(t['all_atom_mask'].numpy()[:,1]>0)
                    r,shift=fit(pos[framework,1],t['all_atom_positions'].numpy()[framework,1])
                    selected=heavy==chain
                    predicted14[selected]=predicted14[selected] @ r+shift
                squared=symmetry_error(predicted14,gt,alt,observed)
                counts=observed.sum(-1)
                residue_error=np.sqrt(np.divide(squared,counts,out=np.zeros_like(squared),where=counts>0))
                if variant==reference:baseline_residue_errors[(model,reference,name)]=residue_error
                delta=residue_error-baseline_residue_errors[(model,reference,name)]
                for residue in np.flatnonzero(counts>0):
                    residue_accuracy_changes.append({'target':name,'chain':'H' if heavy[residue] else 'L',
                        'input_chain_residue':int(residue+1 if heavy[residue] else residue-h+1),
                        'amino_acid':sequence[residue],'region':str(regions[residue]),
                        'change_negative_is_better':float(delta[residue]),
                        'baseline_sidechain_rmsd':float(baseline_residue_errors[(model,reference,name)][residue]),
                        'candidate_sidechain_rmsd':float(residue_error[residue])})
                all_side=float(np.sqrt(squared.sum()/counts.sum()))
                h3=regions=='cdrh3'
                h3_side=float(np.sqrt(squared[h3].sum()/counts[h3].sum()))
                for residue,res in enumerate(sequence):
                    for number,atoms in enumerate(rc.chi_angles_atoms.get(rc.restype_1to3.get(res,'UNK'),[])):
                        indices=[ATOM_NAMES.index(x) for x in atoms]
                        if valid[residue,indices].all():
                            change=angle_difference(dihedral(pos[residue,indices]),dihedral(bp[residue,indices]))
                            chi_changes.append({'target':name,'residue':residue+1,'region':str(regions[residue]),
                                                'chi':number+1,'degrees':float(change)})
                old_clashes=clash_map(bp,base['masks'],heavy);current=clash_map(pos,mask,heavy)
                for pair in current.keys()-old_clashes.keys():
                    r1,a1,r2,a2=pair
                    distance=float(np.linalg.norm(bp[r1,a1].astype(float)-bp[r2,a2]))
                    threshold=sum({'C':1.7,'N':1.55,'O':1.52,'S':1.8}[ATOM_NAMES[x][0]] for x in [a1,a2])
                    new_clashes.append({'target':name,'pair':[r1+1,ATOM_NAMES[a1],r2+1,ATOM_NAMES[a2]],
                        'old_overlap':threshold-distance,'new_overlap':current[pair]})
                row={'model':model,'variant':variant,'target':name,'comparison_reference':reference,
                    'experimental_symmetry_aware_sidechain_rmsd':all_side,
                    'experimental_cdrh3_sidechain_rmsd':h3_side,'observed_sidechain_atoms':int(counts.sum()),
                    'observed_h3_sidechain_atoms':int(counts[h3].sum())}
                rows.append(row);target_rows.append(row)
            results.append({'model':model,'variant':variant,'targets':len(rows),'comparison_reference':reference,
                'mean_experimental_symmetry_aware_sidechain_rmsd':float(np.mean([x['experimental_symmetry_aware_sidechain_rmsd'] for x in rows])),
                'mean_experimental_cdrh3_sidechain_rmsd':float(np.mean([x['experimental_cdrh3_sidechain_rmsd'] for x in rows])),
                'largest_atom_displacements':sorted(largest,key=lambda x:x['raw_displacement'],reverse=True)[:8],
                'largest_chi_changes':sorted(chi_changes,key=lambda x:x['degrees'],reverse=True)[:8],
                'largest_residue_sidechain_accuracy_worsenings':sorted(residue_accuracy_changes,key=lambda x:x['change_negative_is_better'],reverse=True)[:8],
                'largest_residue_sidechain_accuracy_improvements':sorted(residue_accuracy_changes,key=lambda x:x['change_negative_is_better'])[:8],
                'new_clash_cutoff_crossings':new_clashes})
    for r in results:
        controls={x['target']:x for x in target_rows if x['model']==r['model'] and x['variant']==r['comparison_reference']}
        rows=[x for x in target_rows if x['model']==r['model'] and x['variant']==r['variant']]
        for field in ['experimental_symmetry_aware_sidechain_rmsd','experimental_cdrh3_sidechain_rmsd']:
            changes=np.array([x[field]-controls[x['target']][field] for x in rows])
            r[field+'_paired_change']={'mean':float(changes.mean()),'max_absolute':float(np.abs(changes).max()),
                                      'worsened_by_over_0_01_A':int((changes>.01).sum())}
    (root/'sidechain_assessment_251026.json').write_text(json.dumps({'summaries':results,'per_target':target_rows},indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completed':True,'conditions':len(results)}),flush=True)


if __name__=='__main__':main()
