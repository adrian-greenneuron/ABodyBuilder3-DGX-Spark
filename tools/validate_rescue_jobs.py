"""Independently score stored rescue jobs, without regenerating any coordinates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

from openmm import app
sys.path.insert(0,'/opt/abodybuilder3/scripts')
import structure_preparation as sp
from diagnose_heavy_rescue import failing_centres,assess_original_motifs
from validate_heavy_rescue import compare
from measure_chemical_geometry import score
from score_hydrogen_planes import score_planes
from score_probe_hydrogens import run as probe_run


def bonds(top):
    keys=sp._keys(top)
    return {tuple(sorted((keys[a.index],keys[b.index]))) for a,b in top.bonds()}


def check(job):
    root,case,row,probe=job;folder=Path(root)/case['directory']/'preparation'/row['target']
    meta={k:case[k] for k in ['label','route','policy','replicate']};meta.update(target=row['target'],group=row['group'],preparation_status=row['status'])
    if row['status']!='completed':return {**meta,'error':row.get('error'),'output_exists':(folder/'rescued.prepared.pdb').exists()}
    source=folder/'seeded_reference.pdb';path=folder/'rescued.prepared.pdb'
    a,b=app.PDBFile(str(source)),app.PDBFile(str(path));assert sp._keys(a.topology)==sp._keys(b.topology)
    assert bonds(a.topology)==bonds(b.topology),'Bond topology changed in stored PDB.'
    heavy=compare(source,path)
    system=sp._forcefield().createSystem(a.topology,nonbondedMethod=app.NoCutoff,constraints=None)
    bad=failing_centres(system,a.topology,a.positions);motifs=assess_original_motifs(bad,b.positions)
    valence=score(path);planes=score_planes(path);hydrogen={**valence['summary'],**planes['summary']}
    detail={'valence':valence,'planarity':planes,'heavy':heavy,'motifs':motifs}
    (folder/'independent_geometry_271026.json').write_text(json.dumps(detail,indent=2)+'\n')
    contacts=probe_run((probe,str(path),{}))
    errors=['H_bond_errors_over_0_10_A','H_angle_errors_over_30_deg','alpha_H_wrong_side','tetrahedral_H_wrong_side',
            'tetrahedral_H_signed_volume_inversions','planar_H_over_1_deg','inward_planar_H_over_1_deg']
    passed=not any(hydrogen[k] for k in errors) and all(m['restored'] for m in motifs) and not any(heavy[k] for k in (
        'new_heavy_bond_flags','new_heavy_angle_flags','new_heavy_contact_pairs','biological_stereocentre_inversions'))
    result={**meta,'stored_geometry_passed':passed,'heavy':heavy,'motifs':motifs,'hydrogen':hydrogen,'probe':contacts,
            'prepared_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'topology_identity_passed':True}
    print(meta['label'],row['target'],'PASS' if passed else 'FAIL',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    protocol=json.loads((args.root/'qualification_protocol_271026.json').read_text());jobs=[]
    for case in protocol['cases']:
        d=json.loads((args.root/case['directory']/'preparation/job_271026.json').read_text())
        for row in d['rows']:jobs.append((str(args.root),case,row,str(args.probe)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(check,jobs))
    (args.root/'qualification_stored_geometry_271026.json').write_text(json.dumps({'completed':True,'rows':rows,
        'scope':'Independent stored-PDB H valence, plane/orientation, paired heavy valence, severe contacts, true stereochemistry and complete bond identity. Unmodified Probe bad-overlap pairs are a separate diagnostic, not a pass criterion or MolProbity clashscore.'},indent=2)+'\n')
    assert all(r.get('stored_geometry_passed',False) if r['preparation_status']=='completed' else r['policy']=='fixed' and r['group']=='rejected' and not r['output_exists'] for r in rows)
    print('ALL EXPECTED QUALIFICATIONS PASSED',len(rows),flush=True)


if __name__=='__main__':main()
