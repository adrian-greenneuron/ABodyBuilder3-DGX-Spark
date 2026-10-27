"""Identify fixed-heavy hydrogen failures without changing the heavy geometry."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import openmm as mm
from openmm import app, unit

sys.path.insert(0, '/opt/abodybuilder3/scripts')
import hydrogen_geometry as hg
import structure_preparation as sp


def heavy_metrics(system, topology, positions):
    xyz=sp._positions(positions)*10
    atoms=list(topology.atoms());hs={a.index for a in atoms if a.element.symbol=='H'}
    bonds=[];angles=[]
    for force in system.getForces():
        if isinstance(force,mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                a,b,r,k=force.getBondParameters(i)
                if a in hs or b in hs:continue
                error=np.linalg.norm(xyz[a]-xyz[b])-r.value_in_unit(unit.angstrom)
                bonds.append(float(error))
        elif isinstance(force,mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                a,b,c,t,k=force.getAngleParameters(i)
                if hs.intersection((a,b,c)):continue
                u,v=xyz[a]-xyz[b],xyz[c]-xyz[b]
                actual=math.degrees(math.acos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))
                angles.append(actual-t.value_in_unit(unit.degree))
    return {'heavy_bonds':len(bonds),'heavy_bond_errors_over_0_10_A':sum(abs(e)>.1 for e in bonds),
            'max_heavy_bond_error_A':max(map(abs,bonds),default=0),
            'heavy_angles':len(angles),'heavy_angle_errors_over_20_deg':sum(abs(e)>20 for e in angles),
            'max_heavy_angle_error_deg':max(map(abs,angles),default=0)}


def failing_centres(system, topology, positions):
    xyz=sp._positions(positions);atoms=list(topology.atoms());keys=sp._keys(topology)
    mobile=[a.index for a in atoms if a.element.symbol=='H'];ms=set(mobile)
    frames=hg.planar_frames(topology,xyz,mobile)
    adjacent={a.index:[] for a in atoms}
    for a,b in topology.bonds():adjacent[a.index].append(b.index);adjacent[b.index].append(a.index)
    targets={};bond_targets={}
    for force in system.getForces():
        if isinstance(force,mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                a,b,r,k=force.getBondParameters(i);bond_targets[tuple(sorted((a,b)))]=r.value_in_unit(unit.angstrom)
        if isinstance(force,mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                a,b,c,t,k=force.getAngleParameters(i);targets[a,b,c]=t.value_in_unit(unit.radian)
    records=[]
    for index in frames:
        h=mobile[index];p=adjacent[h][0];heavy=[i for i in adjacent[p] if i not in ms]
        if len(heavy)!=2 or len([i for i in adjacent[p] if i in ms])!=1:continue
        u,v=xyz[heavy]-xyz[p];delta=math.acos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1))
        ts=[targets.get((h,p,i),targets.get((i,p,h))) for i in heavy]
        try:hg.planar_sector(delta,*ts)
        except ValueError:
            records.append({'parent_index':p,'neighbour_indices':heavy,'parent':keys[p],
                            'neighbours':[keys[i] for i in heavy], 'heavy_angle_deg':math.degrees(delta),
                            'H_angle_targets_deg':list(map(math.degrees,ts)),
                            'heavy_angle_target_deg':math.degrees(targets.get((heavy[0],p,heavy[1]),targets.get((heavy[1],p,heavy[0])))),
                            'heavy_bond_targets_A':[bond_targets[tuple(sorted((p,i)))] for i in heavy],
                            'heavy_distances_A':[float(np.linalg.norm(xyz[i]-xyz[p])*10) for i in heavy]})
    return records


def heavy_error_sets(system,topology,positions):
    xyz=sp._positions(positions)*10;heavy={a.index for a in topology.atoms() if a.element.symbol!='H'};bonds=set();angles=set()
    for force in system.getForces():
        if isinstance(force,mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                a,b,r,k=force.getBondParameters(i)
                if {a,b}<=heavy and abs(np.linalg.norm(xyz[a]-xyz[b])-r.value_in_unit(unit.angstrom))>.1:bonds.add(tuple(sorted((a,b))))
        elif isinstance(force,mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                a,b,c,t,k=force.getAngleParameters(i)
                if not {a,b,c}<=heavy:continue
                u,v=xyz[a]-xyz[b],xyz[c]-xyz[b];actual=math.degrees(math.acos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))
                if abs(actual-t.value_in_unit(unit.degree))>20:angles.add(min((a,b,c),(c,b,a)))
    return bonds,angles


def assess_original_motifs(bad,positions):
    xyz=sp._positions(positions)*10;rows=[]
    for motif in bad:
        p=motif['parent_index'];a,b=motif['neighbour_indices'];u,v=xyz[a]-xyz[p],xyz[b]-xyz[p]
        angle=math.degrees(math.acos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))
        bond_errors=[float(np.linalg.norm(xyz[i]-xyz[p])-target) for i,target in zip([a,b],motif['heavy_bond_targets_A'])]
        error=angle-motif['heavy_angle_target_deg']
        rows.append({'heavy_angle_deg':angle,'angle_error_deg':error,'bond_errors_A':bond_errors,'restored':abs(error)<=10 and max(map(abs,bond_errors))<=.05})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    protocol=json.loads((args.root/'protocol_271026.json').read_text());rows=[]
    for case in protocol['inputs']:
        random.seed(42);np.random.seed(42);started=time.perf_counter()
        raw=(args.root/'inputs'/(case['target']+'.pdb')).read_text();audit={}
        original,fixer=sp._repair(raw,sp.Options(seed=42),audit)
        capture={}
        def callback(system,topology,positions,mobile,platform,properties):
            capture.update(heavy=heavy_metrics(system,topology,positions),bad=failing_centres(system,topology,positions))
            return positions
        modeller=app.Modeller(fixer.topology,fixer.positions)
        modeller.addHydrogens(sp._forcefield(),pH=7,platform=mm.Platform.getPlatformByName('CPU'),platformProperties={'Threads':'1','DeterministicForces':'true'},hydrogenOptimizer=callback)
        name=case['target'];(args.root/'diagnosis'/(name+'.initial_hydrogen_topology.pdb')).write_text(sp._pdb(modeller.topology,modeller.positions))
        record={**case,**capture,'raw_geometry':sp._geometry(original.topology,original.positions),
                'repair':audit,'seconds':time.perf_counter()-started}
        rows.append(record)
        print(name,case['group'],'failing centres',len(record['bad']),
              'angles',[round(x['heavy_angle_deg'],3) for x in record['bad']],flush=True)
    (args.root/'diagnosis/heavy_rescue_diagnosis_271026.json').write_text(json.dumps({'completed':True,'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
