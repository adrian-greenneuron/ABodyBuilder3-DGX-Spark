"""Offline, isolated-condition structural/numerical comparison on published data.

Run one condition per fresh retained Docker runtime. Never install candidate
code or change prediction CLI defaults. Model passes use a fixed matched batch
shape; dummy rows are discarded. Timings exclude scoring, PDB I/O and startup.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'candidate'))
sys.path.insert(0,'/opt/abodybuilder3/scripts')
from compare_component import fixtures, import_file
from benchmark_runtime import Memory
from abodybuilder3.batched_inference import prepare_cpu_batch, batch_add_atom37_to_output


def load_targets(data):
    ids=json.loads((data/'test_ids_251026.json').read_text())
    rows=[]
    for name in ids:
        d=torch.load(data/'experimental'/f'{name}.pt',map_location='cpu',weights_only=False)
        sequence=d['sequence'];h=int(d['is_heavy'].sum());n=len(d['aatype'])
        assert len(sequence)==n and d['structure']==name
        assert torch.equal(d['is_heavy'].long(),torch.tensor([1]*h+[0]*(n-h)))
        assert torch.equal(d['residue_index'],torch.cat([torch.arange(h),torch.arange(n-h)+500]))
        assert torch.isfinite(d['all_atom_positions']).all()
        rows.append({'name':name,'heavy':sequence[:h],'light':sequence[h:],
            'truth':d,'regions':d['region'],'experimental':True})
    for row in fixtures(True):
        rows.append({**row,'truth':None,'regions':['fwh1']*len(row['heavy'])+['fwl1']*len(row['light']),
                     'experimental':False})
    return rows


def groups(rows,size):
    result=[]
    for start in range(0,len(rows),size):
        real=rows[start:start+size]
        result.append((real+([real[-1]]*(size-len(real))),len(real)))
    return result


def fixed_batch(group,compact,size):
    pairs=[(x['heavy'],x['light']) for x in group]
    batch,inputs=(compact.prepare_compact_batch if compact else prepare_cpu_batch)(pairs)
    n=batch['aatype'].shape[1]
    assert n<=size,(n,size)
    padded={}
    for k,v in batch.items():
        shape=(len(group),size,size) if k=='pair_code' else ((len(group),size,size,v.shape[-1]) if k=='pair' else (len(group),size,*v.shape[2:]))
        value=387 if k=='pair_code' else 0
        out=torch.full(shape,value,dtype=v.dtype)
        if k in ('pair','pair_code'):out[:,:n,:n]=v
        else:out[:,:n]=v
        padded[k]=out
    return padded,inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=['base','plddt'],required=True)
    p.add_argument('--variant',choices=['reference','accepted','attention','point_ieee','compiled','strict_reference','strict_attention','strict_compiled'],required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--padded-length',type=int,default=272)
    p.add_argument('--repeats',type=int,default=2)
    args=p.parse_args()
    # Torch can create its explicit cache directory during import, before main.
    # Accept that empty/cache-only destination, never an existing prediction.
    args.output.mkdir(parents=True,exist_ok=True)
    if any(p.suffix in ('.npz','.pdb') or p.name=='metadata_251026.json' for p in args.output.iterdir()):
        raise FileExistsError(f'Prediction outputs already exist: {args.output}')
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    original_tf32=torch.backends.cuda.matmul.allow_tf32
    original_precision=torch.get_float32_matmul_precision()
    strict=args.variant.startswith('strict_')
    if strict:torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    rows=load_targets(args.data)
    work=groups(rows,args.batch_size)
    ref=import_file('structure_reference',ROOT/'reference/structure_module.py')
    accepted=import_file('structure_accepted',ROOT/'accepted/structure_module.py')
    candidate=import_file('structure_experiment',ROOT/'candidate/structure_module.py')
    compact=import_file('features_accepted',ROOT/'accepted/features.py')
    cli=import_file('structure_cli','/opt/abodybuilder3/scripts/predict.py');cli.VERBOSE=False
    is_ref=args.variant in ('reference','strict_reference')
    use_compact=args.variant not in ('reference','strict_reference','strict_attention','strict_compiled')
    module=ref if is_ref else (candidate if args.variant in ('attention','point_ieee','strict_attention') else accepted)
    model=cli.load_model(args.model,torch.device('cuda'))
    model.__class__=module.StructureModule
    for layer in model.ipa_layers:layer.__class__=module.InvariantPointAttention
    model.eval()
    kwargs={} if is_ref else {'_inference_final_only':True,'_inference_pair_buffer':True}
    if use_compact:kwargs['_inference_pair_projected']=True
    if args.variant in ('attention','strict_attention'):kwargs['_inference_point_matmul']=True
    if args.variant=='point_ieee':kwargs['_inference_point_ieee']=True
    projector=compact.PairProjection(model.linear_in_edge) if use_compact else None
    execute=torch.compile(model,mode='default',dynamic=False) if args.variant.endswith('compiled') else model
    cpu_batches=[fixed_batch(group,compact if use_compact else None,args.padded_length) for group,_ in work]
    predictions={};measurements=[]

    def forward(index,active=execute):
        cpu,inputs=cpu_batches[index]
        batch={k:v.to('cuda') for k,v in cpu.items()}
        if use_compact:batch['pair']=projector(batch.pop('pair_code'))
        output=active({'single':batch['single'],'pair':batch['pair']},batch['aatype'],batch['seq_mask'],**kwargs)
        return output,batch,inputs

    # The first real pass warms the compiler/kernels and saves unrounded outputs.
    first_batch_seconds=None
    with torch.no_grad():
        for j,(group,real_count) in enumerate(work):
            torch.cuda.synchronize();start=time.perf_counter()
            out,batch,inputs=forward(j)
            torch.cuda.synchronize()
            if j==0:first_batch_seconds=time.perf_counter()-start
            cpu=batch_add_atom37_to_output(out,batch['aatype'],[len(x['aatype']) for x in inputs])
            for row,result,inp in zip(group[:real_count],cpu[:real_count],inputs[:real_count]):
                name=row['name'];truth=row['truth']
                fields={'positions':result['atom37'],'masks':result['atom37_atom_exists'].astype(bool),
                    'confidence_logits':result.get('plddt',np.empty((0,),np.float32)),
                    'heavy':inp['is_heavy'].numpy().astype(bool),'regions':np.array(row['regions'])}
                if truth is not None:
                    fields.update(truth_positions=truth['all_atom_positions'].numpy(),
                        truth_masks=truth['all_atom_mask'].numpy().astype(bool),
                        source_residue_index=truth['residue_index'].numpy())
                np.savez_compressed(args.output/f'{name}.npz',**fields)
                (args.output/f'{name}.pdb').write_text(cli.generate_pdb(result,inp))
                predictions[name]={'experimental':row['experimental'], 'heavy':row['heavy'],'light':row['light'],
                    'regions':row['regions'],'resolution':float(truth['resolution']) if truth is not None else None,
                    'source_sha256':hashlib.sha256((args.data/'experimental'/f'{name}.pt').read_bytes()).hexdigest() if truth is not None else None}
            del out,batch,cpu
        # Matched warmed model passes include feature transfer/projection, but
        # exclude atom conversion, writing, scoring, and model/compile startup.
        for rep in range(args.repeats):
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats();start_alloc=torch.cuda.memory_allocated()
            with Memory() as memory:
                start=time.perf_counter()
                for j in range(len(work)):
                    out,batch,_=forward(j)
                    del out,batch
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
            measurements.append({'repeat':rep,'seconds':elapsed,'real_targets':len(rows),
                'executed_rows':len(work)*args.batch_size,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
                'persistent_cuda_allocated_before_pass_bytes':start_alloc,**memory.result()})
    # Diagnostic IPA outputs from the first batch only. No hooks in timed passes
    # or compiled execution; compiler boundaries and measured kernels stay intact.
    reduction_probe=None
    if args.variant=='point_ieee':
        import point_precision
        untouched_weighted_points=point_precision.weighted_points
        def probed_weighted_points(a,points):
            nonlocal reduction_probe
            full_a,full_points=a,points
            if reduction_probe is None:
                a,points=a[:1],points[:1]
                precise=(a.double()[...,None,:,:,None]*points.double()[...,None,:,:]).sum(-2)
                broadcast=(a[...,None,:,:,None]*points[...,None,:,:]).sum(-2)
                ieee=untouched_weighted_points(a,points)
                prior=torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.backends.cuda.matmul.allow_tf32=True
                    accelerated=torch.stack([a @ v.to(a.dtype) for v in torch.unbind(points,dim=-3)],dim=-3)
                finally:torch.backends.cuda.matmul.allow_tf32=prior
                reduction_probe={'target':work[0][0][0]['name'],'layer':0,
                    'scope':'First target, first IPA point reduction, same inputs; FP64 mathematical-sum oracle; not experimental truth.',
                    'errors_against_fp64':{name:{'max_absolute':float((value.double()-precise).abs().max()),
                        'rms':float((value.double()-precise).square().mean().sqrt())}
                        for name,value in [('original_broadcast',broadcast),('strict_matmul',ieee),('tf32_permitted_matmul',accelerated)]}}
                # Preserve the complete batch in the actual diagnostic forward.
            return untouched_weighted_points(full_a,full_points)
        point_precision.weighted_points=probed_weighted_points
    layers={};handles=[]
    if not args.variant.endswith('compiled'):
        for i,layer in enumerate(model.ipa_layers):
            handles.append(layer.register_forward_hook(lambda m,a,o,i=i:layers.update({str(i):o.detach().cpu().numpy()})))
    with torch.no_grad():
        out,batch,inputs=forward(0)
        repeated=batch_add_atom37_to_output(out,batch['aatype'],[len(x['aatype']) for x in inputs])
    for handle in handles:handle.remove()
    for row,result in zip(work[0][0][:work[0][1]],repeated):
        saved=np.load(args.output/(row['name']+'.npz'))
        assert np.array_equal(saved['positions'],result['atom37']),('Repeatability failed',row['name'])
        assert np.array_equal(saved['confidence_logits'],result.get('plddt',np.empty((0,),np.float32))),('Confidence repeatability failed',row['name'])
    if layers:
        np.savez_compressed(args.output/'ipa_first_batch.npz',**layers)
    del out,batch,repeated
    assert torch.backends.cuda.matmul.allow_tf32==(False if strict else original_tf32),'Local precision control leaked'
    paths=[ROOT/'reference/structure_module.py',ROOT/'accepted/structure_module.py',
           ROOT/'accepted/features.py',ROOT/'candidate/structure_module.py',ROOT/'candidate/point_precision.py',Path(__file__)]
    metadata={'model':args.model,'variant':args.variant,'torch':torch.__version__,'cuda':torch.version.cuda,
        'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'normal_runtime_tf32':original_tf32,'normal_runtime_precision':original_precision,
        'first_batch_forward_seconds':first_batch_seconds,'measurements':measurements,
        'batch_size':args.batch_size,'padded_length':args.padded_length,'targets':predictions,
        'repeatability_checked_targets':work[0][1],
        'weighted_point_reduction_probe':reduction_probe,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'scope':'Isolated offline numerical/experimental test. No candidate installed; same weights. Timings are warmed model passes, not complete CLI jobs.'}
    (args.output/'metadata_251026.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'completed':True,'model':args.model,'variant':args.variant,
        'targets':len(rows),'first_batch_forward_seconds':first_batch_seconds,
        'seconds':[m['seconds'] for m in measurements]}),flush=True)


if __name__=='__main__':main()
