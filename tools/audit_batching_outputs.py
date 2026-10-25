"""Check every saved benchmark PDB against the matched main prediction."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def parse(raw):
    lines=[x for x in raw.decode().splitlines() if x.startswith('ATOM  ')]
    keys=[(x[12:27],x[76:78]) for x in lines]
    coords=np.array([[float(x[30:38]),float(x[38:46]),float(x[46:54])] for x in lines])
    ca=np.array([x[12:16].strip()=='CA' for x in lines])
    assert np.isfinite(coords).all()
    return keys,coords,ca


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root;refs={};cache={};cases=[];total=0
    for path in sorted(root.glob('*/*/measurements.json')):
        records=json.loads(path.read_text());case=path.parent.parent.name;model=path.parent.name
        for row in records:
            assert row['completed'],path
            folder=path.parent/f"run{row['replicate']}"/'pdbs'
            input_file=root/'frozen_main'/f"antibodies_{row['count']}.csv"
            expected=list(csv.DictReader(input_file.open()))
            assert {p.name for p in folder.glob('*.pdb')}=={x['name']+'.pdb' for x in expected}
            exact=0;ca_max=0.;atom_max=0.;rms_max=0.
            for item in expected:
                name=item['name'];fixture=name.split('_',1)[1]
                key=(model,fixture)
                if key not in refs:
                    candidates=sorted((root/'large-main'/model/'run1/pdbs').glob('*_'+fixture+'.pdb')) if model!='language' else sorted((root/'language-main/language/run1/pdbs').glob('*_'+fixture+'.pdb'))
                    raw=candidates[0].read_bytes();refs[key]=(hashlib.sha256(raw).hexdigest(),parse(raw))
                raw=(folder/(name+'.pdb')).read_bytes();digest=hashlib.sha256(raw).hexdigest()
                exact+=digest==refs[key][0]
                ckey=(key,digest)
                if ckey not in cache:
                    keys,coords,ca=parse(raw);rk,rc,rm=refs[key][1]
                    assert keys==rk,(path,name,'atom identity mismatch')
                    delta=np.abs(coords-rc)
                    cache[ckey]=(float(delta[ca].max()),float(delta.max()),float(np.sqrt(np.mean(delta**2))))
                c,a,r=cache[ckey];ca_max=max(ca_max,c);atom_max=max(atom_max,a);rms_max=max(rms_max,r)
            # Serial formatting is byte-preserving. Batching permits measured roundoff.
            strict=row.get('batch_size') in (None,'1')
            assert not strict or exact==row['count'],(path,exact,row['count'])
            assert ca_max<=.011 and rms_max<.011,(path,ca_max,rms_max)
            report=folder.parent/'batching.json'
            if row.get('batch_size') not in (None,'1'):
                audit=json.loads(report.read_text());assert audit['completed'] and audit['saved']==row['count']
            cases.append(dict(case=case,model=model,replicate=row['replicate'],warmup=row['warmup'],count=row['count'],byte_identical=exact,max_ca_coordinate_difference_angstrom=ca_max,max_atom_coordinate_difference_angstrom=atom_max,max_atom_coordinate_rms_angstrom=rms_max))
            total+=row['count']
    result=dict(passed=True,files_checked=total,distinct_content_comparisons=len(cache),
                rounded_pdb_ca_tolerance_angstrom=.011,cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'files_checked':total,'max_ca':max(x['max_ca_coordinate_difference_angstrom'] for x in cases),'max_atom':max(x['max_atom_coordinate_difference_angstrom'] for x in cases)}))

if __name__=='__main__':main()
