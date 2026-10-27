"""Independent geometry checks on stored outputs; never edits structures."""
import argparse
import json
from pathlib import Path
from measure_chemical_geometry import score
from score_hydrogen_planes import score_planes


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True);parser.add_argument('--tag',default='271026');parser.add_argument('--allow-incomplete',action='store_true',help='Validate only explicitly completed successful jobs; preserve the incomplete-stage status.');args=parser.parse_args()
    data=json.loads((args.study/f'cli_measurements_{args.tag}.json').read_text());assert data['completed'] or args.allow_incomplete;rows=[];failed=[]
    for case in data['rows']:
        if case['replicate']!=1:continue
        for path in sorted((args.study/'cli'/case['label']/'pdbs').glob('*.prepared.pdb')):
            value=score(path);planes=score_planes(path);s={**value['summary'],**planes['summary']}
            value['planarity']=planes
            checks=['H_bond_errors_over_0_10_A','H_angle_errors_over_30_deg','alpha_H_wrong_side','tetrahedral_H_wrong_side','tetrahedral_H_signed_volume_inversions']
            if case.get('geometry')=='robust_planar':checks+=['planar_H_over_1_deg','inward_planar_H_over_1_deg']
            if case.get('geometry') in ('robust','robust_planar') and any(s[k] for k in checks):failed.append((case['label'],path.name,s))
            (path.parent/(path.stem+'.geometry.json')).write_text(json.dumps(value,indent=2)+'\n')
            rows.append({'label':case['label'],'condition':case['condition'],'route':case['route'],'target':path.name,'replicate':case['replicate'],**s})
        print('VALIDATED',case['label'],flush=True)
    (args.study/f'chemical_geometry_{args.tag}.json').write_text(json.dumps({'completed':True,'source_stage_completed':data['completed'],'scope':'Successfully measured jobs only; incomplete/failed attempts are not converted to successes.','rows':rows,'failed':failed},indent=2)+'\n')
    if failed:raise RuntimeError(f'{len(failed)} outputs fail independent geometry; details retained.')


if __name__=='__main__':main()
