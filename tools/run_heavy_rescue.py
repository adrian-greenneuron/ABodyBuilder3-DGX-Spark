"""Explicit experimental rescue of saved predictions; never a default fallback.

Local and restrained policies run only when the planar-H preflight rejects a
heavy motif. The fixed policy is the no-rescue control. No neural inference is
included. This CLI writes raw, intermediate, prepared and failed-case records.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import random
import sys
import time

import numpy as np
import openmm as mm
from openmm import app,unit

sys.path.insert(0,'/opt/abodybuilder3/scripts')
import structure_preparation as sp
import hydrogen_geometry as hg
from diagnose_heavy_rescue import failing_centres,heavy_error_sets,heavy_metrics,assess_original_motifs
from validate_heavy_rescue import heavy_contacts,biological_volumes

spec=importlib.util.spec_from_file_location('heavy_atom_rescue',Path(__file__).resolve().parents[1]/'candidate/heavy_atom_rescue.py')
rescue=importlib.util.module_from_spec(spec);spec.loader.exec_module(rescue)


def disulfides(top):
    keys=sp._keys(top)
    return sorted(tuple(sorted((keys[a.index],keys[b.index]))) for a,b in top.bonds() if a.name==b.name=='SG')


def heavy_gate(system,top,before,after,bad):
    bb,ba=heavy_error_sets(system,top,before);ab,aa=heavy_error_sets(system,top,after)
    bc=heavy_contacts(top,before);ac=heavy_contacts(top,after)
    bv=biological_volumes(top,before);av=biological_volumes(top,after)
    motifs=assess_original_motifs(bad,after)
    result={'new_heavy_bond_flags':len(ab-bb),'new_heavy_angle_flags':len(aa-ba),
            'new_heavy_contact_pairs':len(ac-bc),'resolved_heavy_contact_pairs':len(bc-ac),
            'biological_stereocentre_inversions':sum(bv[k]*av[k]<=0 for k in bv),
            'original_motifs':motifs,'original_motifs_restored':all(m['restored'] for m in motifs)}
    result['passed']=result['original_motifs_restored'] and not any(result[k] for k in (
        'new_heavy_bond_flags','new_heavy_angle_flags','new_heavy_contact_pairs','biological_stereocentre_inversions'))
    return result


def displacement(top,before,after,keys):
    atoms=list(top.atoms());tk=sp._keys(top)
    heavy=[a.index for a in atoms if a.element.symbol!='H' and tk[a.index] in keys]
    ca=[i for i in heavy if atoms[i].name=='CA']
    delta=np.linalg.norm((sp._positions(after)-sp._positions(before))*10,axis=1)
    return {'original_heavy_count':len(heavy),'heavy_rms_A':float(np.sqrt(np.mean(delta[heavy]**2))),
            'heavy_max_A':float(max(delta[heavy])),'heavy_moved_over_0_01_A':int(np.sum(delta[heavy]>.01)),
            'CA_rms_A':float(np.sqrt(np.mean(delta[ca]**2))),'CA_max_A':float(max(delta[ca]))}


def prepare_one(job):
    source,output,case,policy,route=job;started=time.perf_counter()
    folder=Path(output)/case['target'];folder.mkdir()
    raw=Path(source).read_bytes();assert hashlib.sha256(raw).hexdigest()==case['raw_sha256']
    (folder/'raw.pdb').write_bytes(raw)
    report={**case,'policy':policy,'route':route,'status':'started','heavy_attempts':[],'seed':42,'ph':7,
            'fallback':'No automatic relaxation, topology substitution or alternate hydrogen backend.'}
    try:
        random.seed(42);np.random.seed(42)
        original,fixer=sp._repair(raw.decode(),sp.Options(seed=42),report)
        top,positions=fixer.topology,fixer.positions
        repaired_positions=positions
        if route=='reduce':
            top,positions=sp._reduce(fixer,sp.Options(seed=42,hydrogen_method='reduce',sidechain_flips=True),report)
        report['reduce_orientation_changes']=sp._changes(fixer.topology,repaired_positions,top,positions)
        modeller=app.Modeller(top,positions)
        variants=modeller.addHydrogens(sp._forcefield(),pH=7,platform=mm.Platform.getPlatformByName('CPU'),
            platformProperties={'Threads':'1','DeterministicForces':'true'},hydrogenOptimizer=lambda s,t,p,m,pl,pr:p)
        top,before=modeller.topology,modeller.positions;positions=before
        system=sp._forcefield().createSystem(top,nonbondedMethod=app.NoCutoff,constraints=None)
        bad=failing_centres(system,top,before)
        report.update(failing_centres_before=len(bad),heavy_before=heavy_metrics(system,top,before),
                      protonation_variants=list(variants),disulfide_count=len(disulfides(top)))
        (folder/'seeded_reference.pdb').write_text(sp._pdb(top,before))
        report['repair_preflight_seconds']=time.perf_counter()-started
        if bad and policy=='fixed':
            raise ValueError('Explicit no-rescue control: frozen heavy motif is incompatible with planar outward H.')
        elif bad and policy=='local':
            accepted=False
            for radius,steric in [(1,100000),(2,500000)]:
                positions,detail=rescue.repair(system,top,before,[m['parent_index'] for m in bad],radius=radius,
                    restraint_k=10000,bond_scale=10,preserve_unrelated=True,steric_k=steric)
                gate=heavy_gate(system,top,before,positions,bad)
                report['heavy_attempts'].append({'radius':radius,'steric_k':steric,'minimisation':detail,'geometry_gate':gate})
                (folder/f'local_radius{radius}.pdb').write_text(sp._pdb(top,positions))
                if gate['passed']:accepted=True;break
            if not accepted:raise ValueError('Local search exhausted; geometry checks failed. No full relaxation was substituted.')
        elif bad and policy=='restrained':
            positions,prepass=rescue.repair(system,top,before,[],selection='all',restraint_k=10000,bond_scale=10,max_iterations=2000)
            positions=hg.initial_positions(system,top,positions,[a.index for a in top.atoms() if a.element.symbol=='H'])
            opts=sp.Options(seed=42,relax='restrained',restraint_k=1000,max_iterations=3000,relax_platform='CUDA')
            positions,first=sp._minimise(top,positions,opts,freeze_heavy=True)
            positions,second=rescue.restrained_with_guard(system,top,positions,restraint_k=1000,max_iterations=3000,
                reference=before,chirality_strength=1e8)
            gate=heavy_gate(system,top,before,positions,bad)
            report['heavy_attempts'].append({'geometry_prepass':prepass,'physical_H_stage':first,
                'restrained_stage':second,'geometry_gate':gate})
            if not (gate['passed'] and first['converged'] and second['converged']):
                raise ValueError('Restrained relaxation did not converge or did not pass heavy geometry checks.')
        elif policy not in ('fixed','local','restrained'):raise ValueError('Unknown rescue policy.')
        report['heavy_rescue_seconds']=time.perf_counter()-started-report['repair_preflight_seconds']
        final=app.Modeller(top,positions);final.delete([a for a in top.atoms() if a.element.symbol=='H'])
        h_started=time.perf_counter()
        final.addHydrogens(sp._forcefield(),pH=7,variants=variants,platform=mm.Platform.getPlatformByName('CUDA'),
            platformProperties={'Precision':'mixed','DeterministicForces':'true'},hydrogenOptimizer=hg.optimise)
        report.update(final_H_seconds=time.perf_counter()-h_started,final_H_optimisation=dict(hg.optimise.last_report))
        assert sp._keys(top)==sp._keys(final.topology) and disulfides(top)==disulfides(final.topology)
        # Final H placement must preserve the selected heavy result exactly.
        xyz=sp._positions(positions);new=sp._positions(final.positions)
        heavy=[a.index for a in top.atoms() if a.element.symbol!='H']
        assert np.max(abs(xyz[heavy]-new[heavy]))<1e-12
        report.update(heavy_after=heavy_metrics(system,top,final.positions),
                      rescue_displacement=displacement(top,before,final.positions,set(sp._keys(original.topology))),
                      raw_changes=sp._changes(original.topology,original.positions,final.topology,final.positions),
                      prepared_geometry=sp._geometry(final.topology,final.positions))
        path=folder/'rescued.prepared.pdb';path.write_text(sp._pdb(final.topology,final.positions))
        stored=app.PDBFile(str(path));assert sp._keys(stored.topology)==sp._keys(top) and disulfides(stored.topology)==disulfides(top)
        report.update(status='completed',prepared_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except Exception as exc:report.update(status='rejected',error=type(exc).__name__+': '+str(exc))
    report['seconds_to_output']=time.perf_counter()-started
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==case['raw_sha256']
    (folder/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(case['target'],route,policy,report['status'],round(report['seconds_to_output'],3),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',choices=['fixed','local','restrained'],required=True)
    p.add_argument('--route',choices=['openmm','reduce'],required=True);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--reverse',action='store_true');args=p.parse_args()
    cases=json.loads((args.root/'protocol_271026.json').read_text())['inputs']
    if args.reverse:cases=list(reversed(cases))
    args.output.mkdir();started=time.perf_counter()
    jobs=[(str(args.root/'inputs'/(c['target']+'.pdb')),str(args.output),c,args.policy,args.route) for c in cases]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(prepare_one,jobs))
    result={'completed':True,'seconds_with_worker_startup_and_writing':time.perf_counter()-started,
            'policy':args.policy,'route':args.route,'workers':args.workers,'rows':rows,
            'successful_targets':sum(r['status']=='completed' for r in rows),
            'scope':'Complete optional preparation of saved raw predictions; no neural inference; independent scoring excluded.'}
    (args.output/'job_271026.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
