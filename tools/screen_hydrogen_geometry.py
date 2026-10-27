"""Screen the shared conditioned H optimiser and check force equivalence."""
import argparse
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
from measure_chemical_geometry import score


def equivalence(topology,positions):
    original=sp._forcefield().createSystem(topology,rigidWater=False,nonbondedMethod=app.CutoffNonPeriodic)
    mobile=[a.index for a in topology.atoms() if a.element.symbol=='H']
    for a in topology.atoms():
        if a.index not in mobile:original.setParticleMass(a.index,0)
    reduced=hg.conditioned_system(original,mobile)
    contexts=[];integrators=[];values=[]
    try:
        for system in [original,reduced]:
            integrator=mm.VerletIntegrator(0);integrators.append(integrator)
            contexts.append(mm.Context(system,integrator,mm.Platform.getPlatformByName('Reference')))
        xyz=np.array(positions.value_in_unit(unit.nanometer));perturbed=xyz.copy();perturbed[mobile]+=np.random.default_rng(42).normal(0,.0001,(len(mobile),3))
        for context in contexts:
            state=[]
            for coords in [xyz,perturbed]:
                context.setPositions(coords*unit.nanometer);s=context.getState(getEnergy=True,getForces=True)
                state.append((s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)[mobile]))
            values.append(state)
        result={'energy_difference_change_kj_mol':(values[1][1][0]-values[1][0][0])-(values[0][1][0]-values[0][0][0]),
                'max_mobile_force_error_kj_mol_nm':max(float(np.max(abs(values[0][i][1]-values[1][i][1]))) for i in [0,1])}
        assert abs(result['energy_difference_change_kj_mol'])<1e-5,result
        assert result['max_mobile_force_error_kj_mol_nm']<1e-4,result
        return result
    finally:contexts.clear();integrators.clear()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--resume',action='store_true');parser.add_argument('--modes',nargs='+',default=['conditioned','constrained']);parser.add_argument('--engines',nargs='+',default=['CPU','CUDA']);parser.add_argument('--name',default='screen');args=parser.parse_args()
    output=args.study/(args.name+'_271026.json')
    names=json.loads((args.study/'protocol_271026.json').read_text())['diagnosis_targets'];records=json.loads(output.read_text())['rows'] if args.resume else [];folder=args.study/args.name;folder.mkdir(exist_ok=True)
    for name in names:
        random.seed(42);np.random.seed(42);original,fixer=sp._repair((args.source/(name+'.pdb')).read_text(),sp.Options(seed=42),{})
        for mode in args.modes:
            for engine in args.engines:
                if any(r['target']==name and r['mode']==mode and r['engine']==engine for r in records):continue
                random.seed(42);np.random.seed(42);os.environ['AB3_EXPERIMENTAL_GEOMETRY']=mode
                m=app.Modeller(fixer.topology,fixer.positions);platform=mm.Platform.getPlatformByName(engine)
                properties={'Precision':'mixed','DeterministicForces':'true'} if engine=='CUDA' else {'Threads':'1','DeterministicForces':'true'}
                start=time.perf_counter();m.addHydrogens(sp._forcefield(),pH=7,platform=platform,platformProperties=properties,hydrogenOptimizer=hg.optimise);elapsed=time.perf_counter()-start
                p=folder/(name+'_'+engine+'_'+mode+'.prepared.pdb');p.write_text(sp._pdb(m.topology,m.positions));geometry=score(p)
                assert sp._changes(fixer.topology,fixer.positions,m.topology,m.positions)['max_original_heavy_displacement_angstrom']<1e-5
                (p.with_suffix('.geometry.json')).write_text(json.dumps(geometry,indent=2)+'\n')
                row={'target':name,'engine':engine,'mode':mode,'seconds':elapsed,'geometry':geometry['summary'],'optimisation':hg.optimise.last_report}
                if not records and mode=='conditioned':row['force_equivalence']=equivalence(m.topology,m.positions)
                records.append(row);print(json.dumps(row),flush=True)
                output.write_text(json.dumps({'completed':len(records)==len(names)*len(args.engines)*len(args.modes),'rows':records},indent=2)+'\n')


if __name__=='__main__':main()
