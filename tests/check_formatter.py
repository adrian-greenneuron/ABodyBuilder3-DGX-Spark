"""Byte-for-byte regression and warmed CPU screen of the isolated formatter."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference.format_pdb import to_pdb_fast as original
from candidate.format_pdb import to_pdb_fast as native
from candidate import _pdb_native


def fixture(n,seed,dtype=np.float32):
    rng=np.random.default_rng(seed)
    return SimpleNamespace(aatype=rng.integers(0,21,n),
        atom_positions=rng.normal(0,50,(n,37,3)).astype(dtype),
        atom_mask=(rng.random((n,37))>.45).astype(np.float32),
        residue_index=np.arange(n,dtype=np.int64),b_factors=rng.uniform(-30,130,(n,37)).astype(dtype),
        chain_index=(np.arange(n)//max(1,n//3))%2,remark='format regression α' if seed%2 else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,help='Optional retained raw predictions for the extended real-output screen.')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    data=[fixture(n,i) for i,n in enumerate([0,1,2,21,39,120,242,350])]
    for i in range(200):data.append(fixture(1+i%100,1000+i,np.float64 if i%2 else np.float32))
    edges=fixture(42,99,np.float64);edges.aatype=np.arange(-21,21)
    edge_values=[-0.,0.,.0005,-.0005,2.675,-2.675,9999.9995,-999.9995,1e100,-1e100]
    edges.atom_positions[:]=np.resize(edge_values,edges.atom_positions.shape)
    edges.b_factors[:]=np.resize(edge_values,edges.b_factors.shape)
    data.append(edges)
    no_chain=fixture(5,101);no_chain.chain_index=None;data.append(no_chain)
    negatives=fixture(5,102);negatives.chain_index[:]=-1;data.append(negatives)
    empty=fixture(4,103);empty.atom_mask[:]=0;data.append(empty)
    nonfinite=fixture(1,104);nonfinite.atom_positions[0,0,0]=float('nan');data.append(nonfinite)
    # Real predictions cover both checkpoints and all 100 experimental targets.
    real_cases=0
    for model in (['base','plddt'] if args.predictions else []):
        paths=sorted((args.predictions/f'n1000/{model}/eager_normal/run1/raw').glob('*.npz'))
        assert len(paths)==100, 'Extended regression needs all 100 raw targets for each checkpoint'
        for path in paths:
            with np.load(path) as d:
                n=len(d['aatype'])
                data.append(SimpleNamespace(aatype=d['aatype'].copy(),atom_positions=d['positions'].copy(),
                    atom_mask=d['masks'].copy(),residue_index=np.arange(n),
                    b_factors=np.zeros_like(d['masks']),chain_index=1-d['heavy'].astype(int),remark=None))
                real_cases+=1
    assert len(data)==213+real_cases
    for i,item in enumerate(data):
        a,b=original(item),native(item)
        if a!=b:
            differing=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
            raise AssertionError(f'Fixture {i} differs at byte {differing}: {a[max(0,differing-60):differing+60]!r} vs {b[max(0,differing-60):differing+60]!r}')
    selected=data[-200:]
    expected=[original(x) for x in selected]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(native,selected))==expected
    timings={}
    for name,fn in [('original',original),('native',native)]:
        fn(selected[0]);values=[]
        for _ in range(3):
            start=time.perf_counter()
            for item in selected:fn(item)
            values.append(time.perf_counter()-start)
        timings[name]=values
    result={'completed':True,'fixtures':len(data),'real_prediction_cases':real_cases,
            'timing_cases':'200 retained predictions' if real_cases else '200 generated fixtures',
            'all_bytes_exact':True,'threaded_cases':200,
            'native_calls':_pdb_native.calls(),'seconds_per_200':timings,
            'speedup_medians':statistics.median(timings['original'])/statistics.median(timings['native'])}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
