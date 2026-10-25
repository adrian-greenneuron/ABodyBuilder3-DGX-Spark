"""Finite sequential screen then matched complete-process batching comparison."""
import argparse, json, statistics, subprocess, sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate',default='abodybuilder3-dev-spark:latest')
    p.add_argument('--reference',default='abodybuilder3-official-spark:latest')
    p.add_argument('--prott5',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve()
    tool=Path(__file__).with_name('benchmark_cli.py')
    def run(folder,model,count,rep,*,batch=None,cpu=2,reference=False,workers=1):
        dest=root/folder/model
        cmd=[sys.executable,str(tool),'--condition','modernised','--model',model,
             '--image',args.reference if reference else args.candidate,
             '--label','pipeline-'+folder.replace('/','-'),'--replicate',str(rep),
             '--count',str(count),'--workers',str(workers),
             '--frozen',str(root/('frozen_main' if reference else 'frozen_dev')),'--output',str(dest)]
        if batch is not None:cmd+=['--batch-size',str(batch),'--cpu-workers',str(cpu)]
        if model=='language':cmd+=['--prott5',str(args.prott5.resolve())]
        print('START',folder,model,count,rep,flush=True)
        subprocess.run(cmd,check=True)
    # Installed CLI first, before scaling the workload.
    for model in ['base','plddt']:
        run('small',model,5,1,batch='auto')
    screen=[(16,2),(32,1),(32,2),(32,4),(64,2),(1,1)]
    scores=[]
    for batch,cpu in screen:
        folder=f'screen-b{batch}-c{cpu}'
        for rep in range(3):run(folder,'plddt',100,rep,batch=batch,cpu=cpu)
        rows=json.loads((root/folder/'plddt/measurements.json').read_text())
        scores.append(dict(batch=batch,cpu=cpu,median=statistics.median(x['complete_process_seconds'] for x in rows if not x['warmup'])))
    best=min((x for x in scores if x['batch']>1),key=lambda x:x['median'])
    (root/'screen_selection_251026.json').write_text(json.dumps({'screen':scores,'selected':best},indent=2)+'\n')
    for model in ['base','plddt']:
        conditions=[('main',None,2,True,8),('serial',1,1,False,1),('batched',best['batch'],best['cpu'],False,1)]
        for rep in range(3):
            for folder,batch,cpu,reference,workers in (conditions if rep!=2 else conditions[::-1]):
                run('large-'+folder,model,1000,rep,batch=batch,cpu=cpu,reference=reference,workers=workers)
    # Keep costly language embeddings separate, using all five distinct pairs.
    for rep in range(3):
        for reference in ([True,False] if rep!=2 else [False,True]):
            run('language-main' if reference else 'language-dev','language',5,rep,
                batch=None if reference else 1,reference=reference)
    (root/'completed_251026.json').write_text(json.dumps({'completed':True,'selected':best},indent=2)+'\n')

if __name__=='__main__':main()
