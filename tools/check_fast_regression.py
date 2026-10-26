"""Matched fast-path regression with timing stopped before output validation."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);args=p.parse_args();root=args.study.resolve();rows=[]
    images={'accepted':'abodybuilder3-dev-spark:aot-dev-261026','language':'abodybuilder3-dev-spark:direct-language-candidate-261026'}
    for model in ('base','plddt'):
        for rep in (1,2):
            outputs={}
            for condition in (('accepted','language') if rep==1 else ('language','accepted')):
                out=root/'matched_fast_checks'/model/condition/f'run{rep}';out.mkdir(parents=True);outputs[condition]=out
                cmd=['docker','run','--rm','--network','none','--gpus','all','--ipc=host','--user','1000:1000','-e','HOME=/tmp',
                     '-v',f'{root}/frozen_dev_tools:/measurement:ro','-v',f'{args.inputs.resolve()}:/inputs:ro','-v',f'{out}:/results',images[condition],
                     'python','-u','/measurement/measure_deployment.py','--input','/inputs/antibodies_1000.csv','--output','/results','--model',model]
                (out/'command_261026.json').write_text(json.dumps(cmd,indent=2)+'\n');start=time.perf_counter()
                with (out/'inference.log').open('w') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
                duration=time.perf_counter()-start
                if result.returncode:raise RuntimeError(f'Fast path failed: {out}')
                rows.append({'model':model,'condition':condition,'replicate':rep,'container_seconds':duration,'metrics':json.loads((out/'metrics_261026.json').read_text())})
                (root/'matched_fast_checks_261026.json').write_text(json.dumps(rows,indent=2)+'\n')
                print('DONE',model,condition,rep,duration,flush=True)
            left,right=outputs['accepted'],outputs['language']
            if json.loads((left/'fingerprints_261026.json').read_text())!=json.loads((right/'fingerprints_261026.json').read_text()):raise ValueError('Changed fast-path raw results')
            for path in (left/'pdbs').glob('*.pdb'):
                if path.read_bytes()!=(right/'pdbs'/path.name).read_bytes():raise ValueError('Changed fast-path PDB bytes')
            print('EXACT_FAST_PATH',model,rep,flush=True)


if __name__=='__main__':main()
