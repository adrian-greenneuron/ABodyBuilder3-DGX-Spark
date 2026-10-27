"""Check full-CLI output accounting before the next frozen condition starts.

Explicit chemical rejections are valid experimental observations. Missing raw
predictions, missing audits, corrupt outputs and unaccounted targets are not.
This host-side check changes no scientific output or geometry.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--policy',choices=['off','local','restrained'],required=True);args=p.parse_args()
    expected=[r['name'] for r in csv.DictReader(args.input.open())];pdbs=args.directory/'pdbs'
    batching=json.loads((args.directory/'batching.json').read_text());assert batching['saved']==len(expected) and not batching['skipped']
    assert {p.stem for p in pdbs.glob('*.preparation.json')}=={name+'.preparation' for name in expected}
    rows=[]
    for name in expected:
        raw=pdbs/(name+'.pdb');prepared=pdbs/(name+'.prepared.pdb');audit=json.loads((pdbs/(name+'.preparation.json')).read_text())
        assert hashlib.sha256(raw.read_bytes()).hexdigest()==audit['raw_sha256']
        applied=audit.get('rescue',{}).get('applied',False)
        if applied:
            assert args.policy!='off' and audit['rescue']['policy']==args.policy
            assert audit['rescue']['trigger_failure']['status']=='failed'
            assert audit['rescue']['geometry_gate']['passed']
        success=audit['status'] in ['completed','completed_with_warnings']
        if success:
            assert prepared.exists() and prepared.read_bytes().endswith(b'END\n')
            for line in prepared.read_text().splitlines():
                if line.startswith(('ATOM  ','HETATM')):assert all(math.isfinite(float(line[i:i+8])) for i in [30,38,46])
            if not applied:
                changes=audit['preparation_changes']
                if audit['options']['hydrogen_method']=='openmm':assert changes['max_original_heavy_displacement_angstrom']<=1e-5
                else:assert all(r['atom'][-1] not in ('N','CA','C','O') for r in changes['moved_heavy_atoms'])
        else:assert audit['status']=='failed' and not prepared.exists()
        rows.append(dict(target=name,status=audit['status'],rescue_applied=applied,raw_sha256=audit['raw_sha256'],
                         prepared_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest() if success else None,error=audit.get('error')))
    assert {p.name for p in pdbs.glob('*.pdb')}=={name+'.pdb' for name in expected}|{r['target']+'.prepared.pdb' for r in rows if r['prepared_sha256']}
    result=dict(completed=True,targets=len(expected),successful=sum(bool(r['prepared_sha256']) for r in rows),rescued=sum(r['rescue_applied'] for r in rows),rows=rows)
    (args.directory/'completion_271026.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ACCOUNTED',args.directory.name,result['targets'],'targets',result['successful'],'prepared',result['rescued'],'rescued',flush=True)


if __name__=='__main__':main()
