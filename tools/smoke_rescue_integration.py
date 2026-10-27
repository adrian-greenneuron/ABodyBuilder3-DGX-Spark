"""Check rejection-only dispatch through the real preparation worker controller."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,'/opt/abodybuilder3/scripts')
from structure_preparation import Options
from preparation_pool import prepare_paths


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',choices=['local','restrained'],required=True)
    p.add_argument('--route',choices=['openmm','reduce'],required=True);args=p.parse_args()
    cases=json.loads((args.source/'protocol_271026.json').read_text())['inputs'];args.output.mkdir()
    paths=[]
    for case in cases:
        target=args.output/(case['target']+'.pdb');shutil.copy2(args.source/'inputs'/target.name,target);paths.append(target)
    options=Options(seed=42,ph=7.,hydrogen_method=args.route,sidechain_flips=args.route=='reduce')
    records=prepare_paths(paths,options,3);assert len(records)==6
    result=[]
    for case,path in zip(cases,paths):
        report=json.loads(path.with_suffix('.preparation.json').read_text());applied=report.get('rescue',{}).get('applied',False)
        assert applied==(case['group']=='rejected')
        assert report['status'] in ['completed','completed_with_warnings']
        prepared=path.with_suffix('.prepared.pdb')
        previous=args.source/'qualification'/f'{args.route}_{args.policy}_run1'/'preparation'/case['target']/'rescued.prepared.pdb'
        if case['group']=='compatible_control' or args.policy=='local':
            assert prepared.read_bytes()==previous.read_bytes(),('Output changed from qualified standalone path',case['target'])
        result.append(dict(target=case['target'],applied=applied,prepared_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),status=report['status']))
    (args.output/'integration_271026.json').write_text(json.dumps(dict(completed=True,rows=result),indent=2)+'\n')
    print('INTEGRATION PASSED',args.route,args.policy,'exactly 3 rescues; controls unchanged',flush=True)


if __name__=='__main__':main()
