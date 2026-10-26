"""Finite fresh-process/resident comparison of the same small CLI requests."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    args=p.parse_args();root=args.study.resolve();records=[]
    component=root/'frozen_component'
    for count,jobs in ((5,10),(100,5)):
        for model in ('base','plddt'):
            for rep in (1,2):
                folders={}
                for mode in (('fresh','resident') if rep==1 else ('resident','fresh')):
                    dest=root/f'n{count}_jobs{jobs}'/model/mode/f'run{rep}'
                    dest.mkdir(parents=True,exist_ok=False);folders[mode]=dest
                    durations=[];metrics=[]
                    for job in (range(jobs) if mode=='fresh' else [0]):
                        out=dest/f'process{job}';out.mkdir()
                        cmd=['docker','run','--rm','--network','none','--gpus','all','--ipc=host',
                             '--user','1000:1000','-e','HOME=/tmp','-e','PYTHONHASHSEED=0',
                             '-v',f'{component}:/component:ro','-v',f'{args.inputs.resolve()}:/inputs:ro',
                             '-v',f'{out}:/results','abodybuilder3-dev-spark:aot-dev-261026',
                             'python','-u','/component/tools/measure_session.py','--input',f'/inputs/antibodies_{count}.csv',
                             '--output','/results','--model',model,'--jobs',str(jobs if mode=='resident' else 1)]
                        if mode=='resident':cmd+=['--resident']
                        (out/'command_261026.json').write_text(json.dumps(cmd,indent=2)+'\n')
                        print('START',count,jobs,model,rep,mode,job,flush=True);start=time.perf_counter()
                        with (out/'inference.log').open('w') as stream:result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
                        if result.returncode:
                            print((out/'inference.log').read_text()[-6000:],flush=True);raise RuntimeError(mode)
                        durations.append(time.perf_counter()-start)
                        metrics.append(json.loads((out/'metrics_261026.json').read_text()))
                    row={'inputs_per_job':count,'jobs':jobs,'model':model,'replicate':rep,'mode':mode,
                         'total_container_seconds':sum(durations),'individual_container_seconds':durations,
                         'metrics':metrics,'output':str(dest.relative_to(root))}
                    records.append(row)
                    (root/'measurements_261026.json').write_text(json.dumps(records,indent=2)+'\n')
                    print('DONE_GROUP',count,jobs,model,rep,mode,sum(durations),flush=True)
                for job in range(jobs):
                    left=folders['fresh']/f'process{job}/job0'
                    right=folders['resident']/f'process0/job{job}'
                    if json.loads((left/'fingerprints_261026.json').read_text())!=json.loads((right/'fingerprints_261026.json').read_text()):
                        raise ValueError(f'Resident prediction changed: {count} {model} {job}')
                    for path in (left/'pdbs').glob('*.pdb'):
                        if path.read_bytes()!=(right/'pdbs'/path.name).read_bytes():
                            raise ValueError('Resident PDB differs.')
                print('EXACT_ALL_REQUESTS',count,jobs,model,rep,flush=True)
    print('STUDY_COMPLETE',flush=True)


if __name__=='__main__':main()
