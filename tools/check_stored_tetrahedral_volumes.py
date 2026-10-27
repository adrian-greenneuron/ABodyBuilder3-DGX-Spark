"""Extend stored geometry summaries with a four-substituent volume check.

The original independent bond/angle results and all coordinate files remain
unchanged. Added fields are identified in the output metadata.
"""
import argparse
import json
from pathlib import Path

from score_hydrogen_planes import score_planes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    for stage in ['planar127','adaptive127','adaptive382','adaptive382_followup']:
        path=args.root/stage/'chemical_geometry_271026.json'
        data=json.loads(path.read_text())
        for row in data['rows']:
            pdb=args.root/stage/'cli'/row['label']/'pdbs'/row['target']
            new=score_planes(pdb)['summary']
            row['tetrahedral_H_signed_volume_inversions']=new['tetrahedral_H_signed_volume_inversions']
            if 'adaptive' in row['condition']:
                assert new['tetrahedral_H_signed_volume_inversions']==0,(stage,row['target'])
        data['additional_validation']='Independent full four-substituent oriented-volume check; stored coordinates and original valence counts unchanged.'
        path.write_text(json.dumps(data,indent=2)+'\n')
        print('VOLUMES',stage,len(data['rows']),flush=True)


if __name__=='__main__':main()
