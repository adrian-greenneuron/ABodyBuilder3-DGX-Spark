"""Capture bounded minimiser trajectories and force-group energies locally."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import openmm as mm
from openmm import app, unit
sys.path.insert(0, "/opt/abodybuilder3/scripts")
import structure_preparation as sp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args();names=json.loads((args.study/'protocol_271026.json').read_text())['diagnosis_targets']
    records=[]
    for name in names:
        random.seed(42);np.random.seed(42)
        original,fixer=sp._repair((args.source/(name+'.pdb')).read_text(),sp.Options(seed=42),{})
        for engine in ['CPU','CUDA']:
            random.seed(42);np.random.seed(42)
            modeller=app.Modeller(fixer.topology,fixer.positions)
            original_min=mm.LocalEnergyMinimizer.minimize
            ff=sp._forcefield()
            def minimise(context,tolerance,iterations):
                class Reporter(mm.MinimizationReporter):
                    def report(self,iteration,x,grad,values):
                        if iteration%25==0:
                            curve.append({'iteration':iteration,'energy':values['system energy'],'gradient_rms':float(np.sqrt(np.mean(np.square(grad))))})
                        return False
                curve=[];system=context.getSystem()
                for i,force in enumerate(system.getForces()):force.setForceGroup(i)
                context.reinitialize(preserveState=True)
                mobile=[i for i in range(system.getNumParticles()) if system.getParticleMass(i).value_in_unit(unit.dalton)>0]
                def state():
                    value=context.getState(getEnergy=True,getForces=True);f=value.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
                    return {'energy_kj_mol':value.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                            'mobile_force_rms':float(np.sqrt(np.mean(np.sum(f[mobile]**2,axis=1)))),
                            'force_energies':{type(force).__name__:context.getState(getEnergy=True,groups={i}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) for i,force in enumerate(system.getForces())}}
                stages=[{'iterations':0,**state()}]
                for n in [50,450]:
                    start=time.perf_counter();original_min(context,tolerance,n,Reporter());stages.append({'iterations':n,'seconds':time.perf_counter()-start,**state()})
                records.append({'target':name,'engine':engine,'stages':stages,'curve':curve})
            mm.LocalEnergyMinimizer.minimize=minimise
            try:
                properties={'Precision':'mixed','DeterministicForces':'true'} if engine=='CUDA' else None
                kwargs={} if properties is None else {'platformProperties':properties}
                modeller.addHydrogens(ff,pH=7,platform=mm.Platform.getPlatformByName(engine),**kwargs)
                folder=args.study/'diagnosis';folder.mkdir(exist_ok=True)
                (folder/(name+'_'+engine+'.pdb')).write_text(sp._pdb(modeller.topology,modeller.positions))
            finally:
                mm.LocalEnergyMinimizer.minimize=original_min
            print(name,engine,records[-1]['stages'],flush=True)
            (args.study/'diagnosis_271026.json').write_text(json.dumps({'completed':len(records)==2*len(names),'rows':records},indent=2)+'\n')


if __name__=='__main__':main()
