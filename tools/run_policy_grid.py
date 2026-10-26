"""Queue an explicit finite batch-policy grid using the shared pipeline tool."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dev',type=Path,required=True)
    p.add_argument('--main',type=Path,required=True)
    p.add_argument('--truth',type=Path,required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[16,32,64,128])
    p.add_argument('--workers',type=int,nargs='+',default=[2])
    p.add_argument('--counts',type=int,nargs='+',default=[1000])
    args=p.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    plan=[{'batch_size':b,'workers':w} for b in args.batch_sizes for w in args.workers]
    (root/'plan_261026.json').write_text(json.dumps(plan,indent=2)+'\n')
    for choice in plan:
        study=root/f"batch{choice['batch_size']}_workers{choice['workers']}"
        command=[sys.executable,str(Path(__file__).with_name('run_pipeline_study.py')),
                 '--study',str(study),'--dev',str(args.dev),'--main',str(args.main),'--previous',str(args.truth),
                 '--image',args.image,'--native-formatter','--modes','eager','--date-code','261026',
                 '--batch-size',str(choice['batch_size']),'--cpu-workers',str(choice['workers']),
                 '--batch-memory-gib','24','--counts',*[str(n) for n in args.counts]]
        print('POLICY',choice,flush=True)
        subprocess.run(command,check=True)
    (root/'completed_261026.json').write_text(json.dumps({'completed':True,'policies':len(plan)})+'\n')


if __name__=='__main__':main()
