"""Certify a subset of unavoidable H/heavy distance-screen contacts.

Enumerate extrema on the unit sphere intersected with linear angle bands.
Other H atoms and most stereochemical restrictions are omitted, giving a
superset of allowed directions: an impossible separation here is a lower
bound on the problem, not proof that every remaining contact is inevitable.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np


def minimum_linear_on_sphere(c,A,b):
    c=np.asarray(c,float);A=np.asarray(A,float).reshape(-1,3);b=np.asarray(b,float);candidates=[]
    def add(x):
        if np.all(A@x<=b+1e-8):candidates.append(float(c@x))
    if np.linalg.norm(c)>1e-15:add(-c/np.linalg.norm(c))
    for a,rhs in zip(A,b):
        norm=np.dot(a,a)
        if norm<1e-20:continue
        base=a*rhs/norm;r2=1-np.dot(base,base)
        if r2 < -1e-10:continue
        q=c-a*np.dot(c,a)/norm
        if np.linalg.norm(q)<1e-12:q=np.cross(a,np.eye(3)[np.argmin(abs(a))])
        add(base-math.sqrt(max(0,r2))*q/np.linalg.norm(q))
        add(base+math.sqrt(max(0,r2))*q/np.linalg.norm(q))
    for i,j in itertools.combinations(range(len(A)),2):
        pair=A[[i,j]];normal=np.cross(*pair);length=np.linalg.norm(normal)
        if length<1e-10:continue
        base=pair.T@np.linalg.solve(pair@pair.T,b[[i,j]]);r2=1-np.dot(base,base)
        if r2>=-1e-10:
            offset=math.sqrt(max(0,r2))*normal/length;add(base+offset);add(base-offset)
    if not candidates:
        if not len(A) and np.linalg.norm(c)<1e-15:return 0.
        raise ValueError('No feasible extremum found; bounds are unqualified.')
    return min(candidates)


def evaluate(path):
    import openmm as mm
    from openmm import app,unit
    sys.path.insert(0,'/opt/abodybuilder3/scripts');import structure_preparation as sp
    pdb=app.PDBFile(str(path));atoms=list(pdb.topology.atoms());keys=sp._keys(pdb.topology);by_key={tuple(k):i for i,k in enumerate(keys)}
    xyz=sp._positions(pdb.positions)*10;parent={};hydrogens={a.index for a in atoms if a.element.symbol=='H'}
    for a,b in pdb.topology.bonds():
        if a.index in hydrogens:parent[a.index]=b.index
        if b.index in hydrogens:parent[b.index]=a.index
    system=sp._forcefield().createSystem(pdb.topology,nonbondedMethod=app.CutoffNonPeriodic,constraints=None,rigidWater=False)
    lengths={};conditions={h:([],[]) for h in hydrogens}
    for force in system.getForces():
        if isinstance(force,mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                a,b,r,k=force.getBondParameters(i)
                if a in hydrogens:lengths[a]=r.value_in_unit(unit.angstrom)
                elif b in hydrogens:lengths[b]=r.value_in_unit(unit.angstrom)
        elif isinstance(force,mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                a,p,c,t,k=force.getAngleParameters(i)
                if a in hydrogens and c not in hydrogens:h,heavy=a,c
                elif c in hydrogens and a not in hydrogens:h,heavy=c,a
                else:continue
                u=xyz[heavy]-xyz[p];u/=np.linalg.norm(u);theta=t.value_in_unit(unit.radian);limit=math.pi/6
                A,b=conditions[h];A.extend([u,-u]);b.extend([math.cos(max(0,theta-limit)),-math.cos(min(math.pi,theta+limit))])
    detail=json.loads(path.with_suffix('.geometry.json').read_text());total=0;certified=0
    for clash in detail['severe_H_clashes']:
        a,b=[by_key[tuple(k)] for k in clash['atoms']]
        if a in hydrogens and b not in hydrogens:h,heavy=a,b
        elif b in hydrogens and a not in hydrogens:h,heavy=b,a
        else:continue
        total+=1;P=xyz[parent[h]];w=xyz[heavy]-P;A,bounds=conditions[h]
        actual=(xyz[h]-P)/np.linalg.norm(xyz[h]-P)
        assert np.all(np.asarray(A)@actual<=np.asarray(bounds)+1e-6)
        lower_dot=minimum_linear_on_sphere(w,A,bounds)
        maximum=math.sqrt(max(0,np.dot(w,w)+lengths[h]**2-2*lengths[h]*lower_dot))
        certified+=maximum < clash['threshold_angstrom']-1e-6
    return {'target':path.name,'H_heavy_contacts':total,'certified_unavoidable_with_angle_limit_30_deg':int(certified)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args();rows=[]
    for stage,case in [('confirmation128','openmm_gpu_fixed12_r1'),('validation256','openmm_gpu_fixed8_r1')]:
        for path in sorted((args.root/stage/'cli'/case/'pdbs').glob('*.prepared.pdb')):rows.append({'stage':stage,**evaluate(path)})
        print('BOUNDS',stage,flush=True)
    (args.root/'fixed_heavy_bounds_271026.json').write_text(json.dumps({'completed':True,'protocol':__doc__,'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
