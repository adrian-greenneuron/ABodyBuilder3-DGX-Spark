"""Unmodified Probe contact validation; no hydrogen addition or coordinate edits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess


def run(job):
    binary,path,meta=job
    command=[binary,'-u','-condense','-once','-mc','-het','-stdbonds','-nuclear','-onlybadout','all','all',path]
    r=subprocess.run(command,capture_output=True,text=True,check=True)
    pairs={}
    for line in r.stdout.splitlines():
        f=line.split(':')
        if len(f)>8 and f[2]=='bo':
            key=tuple(sorted(f[3:5]));pairs[key]=min(pairs.get(key,0),float(f[6]))
    h={p:v for p,v in pairs.items() if any(a.split()[-1].startswith('H') for a in p)}
    atoms=sum(line.startswith(('ATOM  ','HETATM')) for line in Path(path).read_text().splitlines())
    Path(path+'.probe.txt').write_text(r.stdout)
    return {**meta,'atoms':atoms,'all_clash_pairs':len(pairs),'H_clash_pairs':len(h),'all_pairs_per_1000_atoms':1000*len(pairs)/atoms,
            'H_pairs_per_1000_atoms':1000*len(h)/atoms,'max_H_overlap_A':-min(h.values(),default=0)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--stages',nargs='+',default=['smoke6','grid64','confirmation128','validation256']);p.add_argument('--output',default='probe_validation_271026.json');args=p.parse_args()
    jobs=[]
    for stage in args.stages:
        d=json.loads((args.root/stage/'cli_measurements_271026.json').read_text())
        for case in d['rows']:
            if case['replicate']!=1:continue
            for path in sorted((args.root/stage/'cli'/case['label']/'pdbs').glob('*.prepared.pdb')):
                meta={'stage':stage,'condition':case['condition'],'route':case['route'],'label':case['label'],'target':path.name,'geometry':case['geometry']}
                jobs.append((str(args.probe),str(path),meta))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(run,jobs,chunksize=4))
    (args.root/args.output).write_text(json.dumps({'completed':True,'probe_commit':'f3e0fadd3986d228f3b37c17f4ec20a7391daf37',
        'protocol':'Unmodified legacy Probe, nuclear radii, standard residue bonding, unique unordered bad-overlap pairs; no hydrogen regeneration. See exact command in source.','rows':rows},indent=2)+'\n')
    print('PROBE COMPLETE',len(rows),'structures',flush=True)


if __name__=='__main__':main()
