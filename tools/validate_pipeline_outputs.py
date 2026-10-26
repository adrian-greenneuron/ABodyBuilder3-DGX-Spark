"""Validate every raw fingerprint and PDB in matched pipeline conditions."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--reference',default='eager')
    p.add_argument('--candidate',default='graph')
    p.add_argument('--date-code',default='261026')
    args=p.parse_args();root=args.study
    rows=json.loads((root/f'measurements_{args.date_code}.json').read_text());checks=[]
    for row in rows:
        if row['mode']!=args.candidate:continue
        ref=next(r for r in rows if r['mode']==args.reference and
                 (r['count'],r['model'],r['precision'],r['replicate'])==
                 (row['count'],row['model'],row['precision'],row['replicate']))
        actual=root/row['relative_output'];reference=root/ref['relative_output']
        a,b=[json.loads((path/'fingerprints.json').read_text()) for path in [actual,reference]]
        assert len(a)==len(b)==row['count']
        assert [r['name'] for r in a]==[r['name'] for r in b]
        raw=sum(x==y for x,y in zip(a,b))
        pdb=sum((actual/'pdbs'/f"{r['name']}.pdb").read_bytes()==
                (reference/'pdbs'/f"{r['name']}.pdb").read_bytes() for r in a)
        usage=actual/'formatter_usage.json'
        if usage.exists():assert json.loads(usage.read_text())['native_calls']==row['count']
        checks.append({k:row[k] for k in ['count','model','precision','replicate']}|
                      {'raw_exact':raw,'pdb_exact':pdb})
        print(checks[-1],flush=True)
    (root/f'exact_output_validation_{args.date_code}.json').write_text(json.dumps(checks,indent=2)+'\n')
    assert all(r['raw_exact']==r['pdb_exact']==r['count'] for r in checks)
    print('All',sum(r['count'] for r in checks),'matched predictions and PDBs exact',flush=True)


if __name__=='__main__':main()
