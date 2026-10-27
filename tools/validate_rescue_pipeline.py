"""Validate full-pipeline outputs, reusing scores only for identical PDB bytes.

All raw and prepared hashes, audits and rescue applications are accounted for.
Geometry is scored on every unique stored output, including outputs from the
retained CUDA-affected attempt. No coordinates are regenerated or modified.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

from openmm import app
sys.path.insert(0,'/opt/abodybuilder3/scripts')
import structure_preparation as sp
from measure_chemical_geometry import score
from score_hydrogen_planes import score_planes
from score_probe_hydrogens import run as probe_run
from diagnose_heavy_rescue import heavy_metrics,failing_centres,assess_original_motifs
from validate_heavy_rescue import compare


def bond_keys(top):
    keys=sp._keys(top)
    return {tuple(sorted((keys[a.index],keys[b.index]))) for a,b in top.bonds()}


def score_unique(job):
    path,sha,probe=job;path=Path(path);assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    valence=score(path);planes=score_planes(path);summary={**valence['summary'],**planes['summary']}
    pdb=app.PDBFile(str(path));system=sp._forcefield().createSystem(pdb.topology,nonbondedMethod=app.NoCutoff,constraints=None)
    heavy=heavy_metrics(system,pdb.topology,pdb.positions);contacts=probe_run((probe,str(path),{}))
    errors=['H_bond_errors_over_0_10_A','H_angle_errors_over_30_deg','alpha_H_wrong_side','tetrahedral_H_wrong_side',
            'tetrahedral_H_signed_volume_inversions','planar_H_over_1_deg','inward_planar_H_over_1_deg']
    result=dict(prepared_sha256=sha,hydrogen=summary,heavy=heavy,probe=contacts,hydrogen_passed=not any(summary[k] for k in errors))
    path.with_suffix('.independent_geometry_271026.json').write_text(json.dumps(dict(valence=valence,planes=planes),indent=2)+'\n')
    return result


def validate_rescue(job):
    path,audit=job;path=Path(path);ref=path.with_suffix('.rescue_reference_271026.pdb')
    ref.write_text(audit['rescue']['seeded_reference_pdb'])
    a,b=app.PDBFile(str(ref)),app.PDBFile(str(path));assert sp._keys(a.topology)==sp._keys(b.topology)
    assert bond_keys(a.topology)==bond_keys(b.topology)
    paired=compare(ref,path);system=sp._forcefield().createSystem(a.topology,nonbondedMethod=app.NoCutoff,constraints=None)
    bad=failing_centres(system,a.topology,a.positions);motifs=assess_original_motifs(bad,b.positions)
    passed=all(m['restored'] for m in motifs) and not any(paired[k] for k in [
        'new_heavy_bond_flags','new_heavy_angle_flags','new_heavy_contact_pairs','biological_stereocentre_inversions'])
    return dict(paired=paired,motifs=motifs,passed=passed,topology_identity_passed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    protocols=[json.loads((args.root/'protocol_271026.json').read_text()),json.loads((args.root/'retry1_protocol_271026.json').read_text())]
    cases=[c for p in protocols for c in p['cases']];index={};records=[];raw_index={};unique={};rescues=[]
    for case in cases:
        folder=args.root/case['directory'];d=json.loads((folder/'completion_271026.json').read_text());assert d['completed'] and d['targets']==512
        for row in d['rows']:
            path=folder/'pdbs'/(row['target']+'.prepared.pdb');audit=json.loads(path.with_name(row['target']+'.preparation.json').read_text())
            record={k:case[k] for k in ['label','policy','route','replicate']};record.update(row)
            original=raw_index.setdefault(row['target'],row['raw_sha256']);assert original==row['raw_sha256'],'Raw inference changed across compared jobs.'
            if row['prepared_sha256']:
                assert hashlib.sha256(path.read_bytes()).hexdigest()==row['prepared_sha256']
                unique.setdefault(row['prepared_sha256'],str(path))
                if row['rescue_applied']:rescues.append((str(path),audit));record['rescue_validation_index']=len(rescues)-1
            index[case['label'],row['target']]=record;records.append(record)
    print('HASH ACCOUNTING',len(records),'attempts;',len(unique),'unique prepared files;',len(rescues),'rescue calls',flush=True)
    comparisons=[]
    for case in cases:
        for target in raw_index:
            row=index[case['label'],target];baseline=index[case['route']+'_off_r1',target]
            if row['prepared_sha256'] and baseline['prepared_sha256'] and not row['rescue_applied']:
                assert row['prepared_sha256']==baseline['prepared_sha256'],'Non-rescued prepared output changed.'
        comparisons.append(dict(label=case['label'],raw_identical=True,nonrescued_prepared_identical=True))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        scores=[]
        for i,result in enumerate(pool.map(score_unique,[(path,sha,str(args.probe)) for sha,path in unique.items()]),1):
            scores.append(result)
            if i%100==0:print('GEOMETRY',i,'/',len(unique),flush=True)
        rescue_scores=list(pool.map(validate_rescue,rescues))
    result=dict(completed=True,unique_scored=len(scores),accounted_attempts=len(records),
        scores=scores,rescues=rescue_scores,records=records,comparisons=comparisons,
        scope='All unique stored PDB hashes scored; geometry results reused only for byte-identical PDB content. Detailed residue/atom diagnostics and seeded rescue references stay local. Probe diagnostic is not MolProbity clashscore.')
    (args.root/'geometry_validation_271026.json').write_text(json.dumps(result,indent=2)+'\n')
    bad_H=[r for r in scores if not r['hydrogen_passed']];bad_rescue=[i for i,r in enumerate(rescue_scores) if not r['passed']]
    print('FINAL GEOMETRY',len(bad_H),'H failures;',len(bad_rescue),'rescue failures;',len(scores),'unique stored PDBs',flush=True)
    assert not bad_H and not bad_rescue


if __name__=='__main__':main()
