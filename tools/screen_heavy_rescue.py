"""Screen local valence repair against full restrained Amber relaxation.

Uses saved predictions and reports preparation time, not neural inference.
Screen calls share one process; fresh-container confirmation follows separately.
"""
import argparse
import hashlib
import importlib.util
import json
import os
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
spec=importlib.util.spec_from_file_location('heavy_atom_rescue',Path(__file__).resolve().parents[1]/'candidate/heavy_atom_rescue.py')
rescue=importlib.util.module_from_spec(spec);spec.loader.exec_module(rescue)
from diagnose_heavy_rescue import heavy_metrics, failing_centres, heavy_error_sets, assess_original_motifs
from measure_chemical_geometry import score
from score_hydrogen_planes import score_planes


def ss_keys(topology):
    keys=sp._keys(topology)
    return sorted(tuple(sorted((keys[a.index],keys[b.index]))) for a,b in topology.bonds() if a.name==b.name=='SG')


def movements(topology,before,after,original_keys):
    atoms=list(topology.atoms());keys=sp._keys(topology)
    xyz=sp._positions(before)*10;new=sp._positions(after)*10
    heavy=[a.index for a in atoms if a.element.symbol!='H' and keys[a.index] in original_keys]
    ca=[i for i in heavy if atoms[i].name=='CA'];delta=np.linalg.norm(new-xyz,axis=1)
    return {'original_heavy_count':len(heavy),'original_heavy_rms_A':float(np.sqrt(np.mean(delta[heavy]**2))),
            'original_heavy_max_A':float(max(delta[heavy])),'original_heavy_moved_over_0_01_A':int(np.sum(delta[heavy]>.01)),
            'CA_rms_A':float(np.sqrt(np.mean(delta[ca]**2))),'CA_max_A':float(max(delta[ca]))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol',default='screen_protocol_271026.json');args=parser.parse_args()
    root=args.root;protocol=json.loads((root/args.protocol).read_text());out=root/protocol['directory'];out.mkdir();rows=[]
    for case in protocol['inputs']:
        raw_path=root/'inputs'/(case['target']+'.pdb');raw=raw_path.read_text()
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==case['raw_sha256']
        for condition in protocol['conditions']:
            name=case['target'];label=condition['name'];folder=out/name/label;folder.mkdir(parents=True)
            started=time.perf_counter();report={**case,'condition':label,'config':condition,'status':'started'}
            try:
                random.seed(42);np.random.seed(42);repair_audit={}
                original,fixer=sp._repair(raw,sp.Options(seed=42),repair_audit)
                modeller=app.Modeller(fixer.topology,fixer.positions)
                variants=modeller.addHydrogens(sp._forcefield(),pH=7,platform=mm.Platform.getPlatformByName('CPU'),platformProperties={'Threads':'1','DeterministicForces':'true'},hydrogenOptimizer=lambda s,t,p,m,pl,pr:p)
                top=modeller.topology;positions=modeller.positions
                system=sp._forcefield().createSystem(top,nonbondedMethod=app.NoCutoff,constraints=None)
                before=positions;bad=failing_centres(system,top,positions)
                report.update(heavy_before=heavy_metrics(system,top,positions),failing_centres_before=len(bad),repair_audit=repair_audit)
                if condition['kind']=='local' and bad:
                    positions,detail=rescue.repair(system,top,positions,[x['parent_index'] for x in bad],radius=condition['radius'],restraint_k=condition['restraint_k'],selection=condition.get('selection','residue'),preserve_boundary=condition.get('preserve_boundary',False),preserve_unrelated=condition.get('preserve_unrelated',False),bond_scale=condition.get('bond_scale',1),steric_k=condition.get('steric_k',100000))
                    report['heavy_repair']=detail
                elif condition['kind']=='restrained':
                    opts=sp.Options(seed=42,relax='restrained',restraint_k=condition['restraint_k'],max_iterations=condition['max_iterations'],relax_platform=condition['platform'])
                    positions,first=sp._minimise(top,positions,opts,freeze_heavy=True)
                    positions,second=sp._minimise(top,positions,opts,freeze_heavy=False)
                    report['heavy_repair']={'kind':'full restrained Amber ff14SB/GBn2','hydrogen_stage':first,'restrained_stage':second,'seconds':first['seconds']+second['seconds']}
                elif condition['kind']=='restrained_guarded':
                    opts=sp.Options(seed=42,relax='restrained',restraint_k=condition['restraint_k'],max_iterations=condition['max_iterations'],relax_platform='CUDA')
                    prepass=None
                    if condition.get('prepass'):
                        positions,prepass=rescue.repair(system,top,positions,[],selection='all',restraint_k=10000,bond_scale=10,max_iterations=2000)
                        positions=hg.initial_positions(system,top,positions,[a.index for a in top.atoms() if a.element.symbol=='H'])
                    positions,first=sp._minimise(top,positions,opts,freeze_heavy=True)
                    positions,second=rescue.restrained_with_guard(system,top,positions,restraint_k=condition['restraint_k'],max_iterations=condition['max_iterations'],reference=before,chirality_strength=condition.get('chirality_strength',1e12))
                    report['heavy_repair']={'kind':'full restrained Amber with biological chirality guard','geometry_prepass':prepass,'hydrogen_stage':first,'restrained_stage':second,'seconds':first['seconds']+second['seconds']+(prepass['seconds'] if prepass else 0)}
                elif condition['kind'] not in ('none','local'):raise ValueError('Unqualified rescue condition.')
                else:report['heavy_repair']={'kind':'none needed' if not bad else 'none','seconds':0}
                report['failing_centres_after']=len(failing_centres(system,top,positions))
                report['heavy_after']=heavy_metrics(system,top,positions)
                before_bonds,before_angles=heavy_error_sets(system,top,before);after_bonds,after_angles=heavy_error_sets(system,top,positions)
                report.update(new_heavy_bond_errors_over_0_10_A=len(after_bonds-before_bonds),new_heavy_angle_errors_over_20_deg=len(after_angles-before_angles))
                report['original_motifs_after']=assess_original_motifs(bad,positions)
                report['original_motifs_restored']=all(m['restored'] for m in report['original_motifs_after'])
                report['movements']=movements(top,before,positions,set(sp._keys(original.topology)))
                chiral_before=rescue.chiral_volumes(top,sp._positions(before));chiral_after=rescue.chiral_volumes(top,sp._positions(positions))
                report['heavy_chiral_inversions']=sum(chiral_before[k]*chiral_after[k]<=0 for k in chiral_before)
                biological_before=rescue.chiral_volumes(top,sp._positions(before),True);biological_after=rescue.chiral_volumes(top,sp._positions(positions),True)
                report['biological_heavy_chiral_inversions']=sum(biological_before[k]*biological_after[k]<=0 for k in biological_before)
                # Record intermediate heavy geometry before final H placement.
                intermediate=app.Modeller(top,positions);intermediate.delete([a for a in top.atoms() if a.element.symbol=='H'])
                (folder/'heavy_repaired.pdb').write_text(sp._pdb(intermediate.topology,intermediate.positions))
                h_started=time.perf_counter()
                intermediate.addHydrogens(sp._forcefield(),pH=7,variants=variants,platform=mm.Platform.getPlatformByName('CUDA'),platformProperties={'Precision':'mixed','DeterministicForces':'true'},hydrogenOptimizer=hg.optimise)
                report['final_H_seconds']=time.perf_counter()-h_started;report['final_H_optimisation']=dict(hg.optimise.last_report)
                assert sp._keys(intermediate.topology)==sp._keys(top)
                assert ss_keys(intermediate.topology)==ss_keys(top)
                report['final_geometry']=sp._geometry(intermediate.topology,intermediate.positions)
                path=folder/'rescued.prepared.pdb';path.write_text(sp._pdb(intermediate.topology,intermediate.positions))
                stored=app.PDBFile(str(path));assert sp._keys(stored.topology)==sp._keys(top) and ss_keys(stored.topology)==ss_keys(top)
                checks=score(path);planes=score_planes(path)
                (folder/'geometry_detail_271026.json').write_text(json.dumps({'valence':checks,'planes':planes},indent=2)+'\n')
                report['hydrogen_geometry']={**checks['summary'],**planes['summary']}
                errors=['H_bond_errors_over_0_10_A','H_angle_errors_over_30_deg','alpha_H_wrong_side','tetrahedral_H_wrong_side','tetrahedral_H_signed_volume_inversions','planar_H_over_1_deg','inward_planar_H_over_1_deg']
                report['hydrogen_geometry_passed']=not any(report['hydrogen_geometry'][k] for k in errors)
                report['status']='completed' if report['hydrogen_geometry_passed'] and not report['biological_heavy_chiral_inversions'] and report['original_motifs_restored'] and not report['new_heavy_bond_errors_over_0_10_A'] and not report['new_heavy_angle_errors_over_20_deg'] else 'geometry_failed'
            except Exception as exc:
                report.update(status='failed',error=type(exc).__name__+': '+str(exc))
            report['preparation_and_scoring_seconds']=time.perf_counter()-started
            assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==case['raw_sha256']
            (folder/'rescue_audit_271026.json').write_text(json.dumps(report,indent=2)+'\n');rows.append(report)
            (root/(protocol['directory']+'_271026.json')).write_text(json.dumps({'completed':len(rows)==len(protocol['inputs'])*len(protocol['conditions']),'protocol':protocol,'rows':rows},indent=2)+'\n')
            print(name,label,report['status'],round(report['preparation_and_scoring_seconds'],3),report.get('error',''),flush=True)


if __name__=='__main__':main()
