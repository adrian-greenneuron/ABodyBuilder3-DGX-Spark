"""Finite interleaved dev inference comparison, with exact raw-output gates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def compare(reference,candidate):
    a=json.loads((reference/'prediction_fingerprints.json').read_text())
    b=json.loads((candidate/'prediction_fingerprints.json').read_text())
    if a!=b:
        changed=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
        raise ValueError(f'Raw predictions differ: {candidate}; first indices {changed[:10]}')
    files=sorted((reference/'pdbs').glob('*.pdb'))
    if [p.name for p in files]!=sorted(p.name for p in (candidate/'pdbs').glob('*.pdb')):
        raise ValueError('Missing/extra candidate PDBs')
    for p in files:
        if p.read_bytes()!=(candidate/'pdbs'/p.name).read_bytes():
            raise ValueError(f'PDB bytes differ: {p.name}')
    return {'raw_predictions_exact':True,'pdb_bytes_exact':True,'predictions':len(a)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',default='abodybuilder3-dev-spark:batched-251026')
    p.add_argument('--candidate',default='abodybuilder3-dev-spark:optimised-buffer-251026')
    args=p.parse_args();root=args.root.resolve()
    tool=Path(__file__).with_name('benchmark_cli.py')
    agreement=[]
    def run(dataset,model,count,repeat,condition):
        reference=condition=='reference'
        frozen=root/('frozen_baseline' if reference else 'frozen_candidate')
        if dataset=='unique':frozen=root/('unique_baseline' if reference else 'unique_candidate')
        output=root/'complete'/dataset/model/f'n{count}'/condition
        cmd=[sys.executable,str(tool),'--condition','modernised','--model',model,
             '--image',args.reference if reference else args.candidate,
             '--label',f'opt-{dataset}-{condition}','--replicate',str(repeat),
             '--count',str(count),'--workers','1','--batch-size','auto','--cpu-workers','2',
             '--fingerprint','--frozen',str(frozen),'--output',str(output)]
        if not reference:cmd+=['--inference-mode',condition]
        print('START',dataset,model,count,repeat,condition,flush=True)
        subprocess.run(cmd,check=True)
        return output/f'run{repeat}'
    # Real installed CLI first; then scale. No silent fallback on disagreement.
    for model in ('base','plddt'):
        a=run('repeated',model,5,1,'reference')
        b=run('repeated',model,5,1,'optimised')
        agreement.append(dict(dataset='repeated',model=model,count=5,repeat=1,condition='optimised',**compare(a,b)))
    for dataset,count in [('repeated',100),('repeated',1000),('unique',1000)]:
        for model in ('base','plddt'):
            conditions=['reference','optimised']
            if dataset=='repeated' and count==1000 and model=='plddt':
                conditions=['reference','lookup','lookup-final','optimised']
            for repeat in range(3):
                for condition in (conditions if repeat!=2 else conditions[::-1]):
                    run(dataset,model,count,repeat,condition)
                reference=root/'complete'/dataset/model/f'n{count}'/'reference'/f'run{repeat}'
                for condition in conditions[1:]:
                    candidate=root/'complete'/dataset/model/f'n{count}'/condition/f'run{repeat}'
                    agreement.append(dict(dataset=dataset,model=model,count=count,repeat=repeat,
                                          warmup=repeat==0,condition=condition,**compare(reference,candidate)))
                (root/'complete_job_agreement_251026.json').write_text(json.dumps(agreement,indent=2)+'\n')
    (root/'complete_job_agreement_251026.json').write_text(json.dumps(agreement,indent=2)+'\n')
    (root/'completed_251026.json').write_text(json.dumps({'completed':True,'comparisons':len(agreement)},indent=2)+'\n')
    print('COMPLETED',len(agreement),'exact matched comparisons',flush=True)


if __name__=='__main__':main()
