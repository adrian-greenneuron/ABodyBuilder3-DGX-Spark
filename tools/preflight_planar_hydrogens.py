"""Check frozen-heavy compatibility before strict planar qualification.

This preflight uses stored predictions; it does not edit geometry or discard
records. A rejection is recorded explicitly and kept for the failure tests.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import openmm as mm
from openmm import app
sys.path.insert(0,'/opt/abodybuilder3/scripts')
import structure_preparation as sp
import hydrogen_geometry as hg


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args();rows=[]
    panels=[('tuned128','openmm_gpu_fixed12_gt5_r1','antibodies_128.csv'),('validation256','openmm_gpu_fixed8_r1','validation_256.csv'),('fresh128','openmm_gpu_fixed12_gt5_r1','fresh_128.csv')]
    for stage,case,filename in panels:
        inputs={r['name']:r for r in csv.DictReader((args.root/stage/'inputs'/filename).open())}
        for path in sorted((args.root/stage/'cli'/case/'pdbs').glob('*.prepared.pdb')):
            d=app.PDBFile(str(path));xyz=sp._positions(d.positions);mobile=[a.index for a in d.topology.atoms() if a.element.symbol=='H'];system=sp._forcefield().createSystem(d.topology,nonbondedMethod=app.CutoffNonPeriodic,rigidWater=False)
            name=path.name.removesuffix('.prepared.pdb');item={'stage':stage,'target':name,'passed':True}
            try:frames=hg.planar_frames(d.topology,xyz,mobile);hg.planar_sectors(system,d.topology,xyz,mobile,frames)
            except ValueError as exc:item.update(passed=False,reason=str(exc))
            rows.append(item)
        print('PREFLIGHT',stage,sum(r['passed'] for r in rows if r['stage']==stage),'/',len(inputs),flush=True)
    (args.root/'planar_preflight_271026.json').write_text(json.dumps({'completed':True,'protocol':__doc__,'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
