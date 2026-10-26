"""Bounded shape/batch screen with accepted AOT controls and raw-output gates."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np


def agreement(left, right):
    rows = []
    if {p.name for p in left.glob('*.npz')} != {p.name for p in right.glob('*.npz')}:
        raise ValueError('Unmatched captured targets.')
    for path in left.glob('*.npz'):
        with np.load(path) as first, np.load(right / path.name) as second:
            if not np.array_equal(first['masks'], second['masks']):
                raise ValueError('Changed atom masks.')
            mask = first['masks'].astype(bool)
            a, b = first['positions'].astype(float), second['positions'].astype(float)
            delta = np.linalg.norm(a-b, axis=-1)
            x, y = a[mask[:, 1], 1], b[mask[:, 1], 1]
            cm, ct = x.mean(0), y.mean(0)
            u, _, vt = np.linalg.svd((x-cm).T @ (y-ct))
            correction = np.eye(3); correction[-1, -1] = np.linalg.det(u @ vt)
            rotation = u @ correction @ vt
            rmsd = float(np.sqrt(np.mean(np.sum(((x-cm) @ rotation - (y-ct))**2, axis=-1))))
            rows.append({'name': path.stem, 'max_CA_A': float(delta[mask[:,1], 1].max()),
                         'max_atom_A': float(delta[mask].max()), 'aligned_CA_RMSD_A': rmsd,
                         'raw_exact': all(np.array_equal(first[key], second[key]) for key in first.files)})
    return {'captured_targets': len(rows), 'max_CA_A': max(row['max_CA_A'] for row in rows),
            'max_atom_A': max(row['max_atom_A'] for row in rows),
            'max_aligned_CA_RMSD_A': max(row['aligned_CA_RMSD_A'] for row in rows),
            'mean_aligned_CA_RMSD_A': statistics.mean(row['aligned_CA_RMSD_A'] for row in rows),
            'raw_exact_targets': sum(row['raw_exact'] for row in rows), 'per_target': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--count', type=int, default=1000)
    p.add_argument('--repetitions', type=int, default=2)
    p.add_argument('--conditions', nargs='+', default=['accepted32','dynamic16','dynamic32','dynamic64','dynamic128','fixed32'])
    args = p.parse_args(); root = args.study.resolve()
    component = root / 'frozen_measurement'
    definitions = {'accepted32':(32,6,False,False), 'dynamic16':(16,6,True,False),
                   'dynamic32':(32,6,True,False), 'dynamic64':(64,12,True,False),
                   'dynamic128':(128,24,True,False), 'fixed32':(32,6,True,True)}
    records = []
    for model in ('base','plddt'):
        baseline = None
        for rep in range(1, args.repetitions+1):
            order = args.conditions if rep%2 else list(reversed(args.conditions))
            for name in order:
                cap, memory, new_bank, fixed = definitions[name]
                out = root / f'n{args.count}' / model / name / f'run{rep}'
                out.mkdir(parents=True, exist_ok=False)
                cmd=['docker','run','--rm','--network','none','--gpus','all','--ipc=host',
                     '--user','1000:1000','-e','HOME=/tmp','-e','PYTHONHASHSEED=0',
                     '-v',f'{component}:/component:ro','-v',f'{args.inputs.resolve()}:/inputs:ro',
                     '-v',f'{root}/dynamic128:/schedule/dynamic128:ro',
                     '-v',f'{root}/fixed240:/schedule/fixed240:ro','-v',f'{root}/fixed248:/schedule/fixed248:ro',
                     '-v',f'{out}:/results','abodybuilder3-dev-spark:aot-dev-261026',
                     'python','-u','/component/tools/measure_pipeline.py','--mode','reference',
                     '--input',f'/inputs/antibodies_{args.count}.csv','--output','/results','--model',model,
                     '--batch-size',str(cap),'--batch-memory-gib',str(memory)]
                if new_bank:cmd+=['--aot-root','/schedule/dynamic128']
                if fixed:cmd+=['--fixed-bank-root','/schedule']
                (out/'command_261026.json').write_text(json.dumps(cmd,indent=2)+'\n')
                print('START',args.count,model,name,rep,flush=True);start=time.perf_counter()
                with (out/'inference.log').open('w') as stream:
                    result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:
                    print((out/'inference.log').read_text()[-6000:],flush=True);raise RuntimeError(name)
                elapsed=time.perf_counter()-start
                metrics=json.loads((out/'metrics_261026.json').read_text())
                batches=json.loads((out/'batching.json').read_text())
                if batches['saved']!=args.count or batches['oom_retries']:
                    raise ValueError('Incomplete screen output or unexpected OOM.')
                if baseline is None:
                    if name!='accepted32':raise ValueError('First condition must be the accepted control.')
                    baseline=out
                paired=agreement(baseline/'raw',out/'raw')
                passed=paired['max_CA_A']<=.001 and paired['max_atom_A']<=.01 and paired['max_aligned_CA_RMSD_A']<=.0001
                row={'count':args.count,'model':model,'condition':name,'replicate':rep,
                     'container_seconds':elapsed,'metrics':metrics,'numerical':paired,
                     'numerical_gate_passed':passed,'batch_sizes':[b['size'] for b in batches['batches']],
                     'output':str(out.relative_to(root))}
                records.append(row)
                (root/f'measurements_n{args.count}_261026.json').write_text(json.dumps(records,indent=2)+'\n')
                print('DONE',args.count,model,name,rep,round(elapsed,3),'gate',passed,
                      'exact_targets',paired['raw_exact_targets'],flush=True)
    print('SCREEN_COMPLETE',flush=True)


if __name__=='__main__':main()
