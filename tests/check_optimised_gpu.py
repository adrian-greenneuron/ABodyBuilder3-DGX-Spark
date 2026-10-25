"""Strict deployed-inference regression: no tolerance for new differences."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(42)
    sys.path.insert(0,'/opt/abodybuilder3/scripts')
    spec=importlib.util.spec_from_file_location('optimised_cli','/opt/abodybuilder3/scripts/predict.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli);cli.VERBOSE=False
    from batch_pipeline import BatchPolicy,plan_batches
    from abodybuilder3.batched_inference import prepare_cpu_batch,infer_prepared_batch,batch_add_atom37_to_output
    fixtures=json.loads(Path('/tests/benchmark_antibodies.json').read_text())
    cases=[(x['name'],x['heavy'],x['light']) for x in fixtures]
    first=fixtures[0]
    cases += [('padding_short',first['heavy'][:-20],first['light'][:-10]),
              ('padding_long',first['heavy']+'G'*20,first['light']+'S'*15)]
    for i in range(33):
        h=list(first['heavy']);p=22+i*2
        h[p]='S' if h[p]!='S' else 'A'
        cases.append((f'engineering_substitution_{i}',''.join(h),first['light']))
    rows=[]
    for model_name in ('base','plddt'):
        model=cli.load_model(model_name,torch.device('cuda'))
        for cap in (1,4,16,32):
            for group in plan_batches(cases,BatchPolicy(cap)):
                pairs=[(h,l) for _,h,l in group]
                dense,inputs=prepare_cpu_batch(pairs)
                batch={k:v.to('cuda') for k,v in dense.items()}
                with torch.no_grad():
                    original=model({'single':batch['single'],'pair':batch['pair']},
                                   batch['aatype'],batch['seq_mask'])
                    reference=batch_add_atom37_to_output(original,batch['aatype'],[len(x['aatype']) for x in inputs])
                del original,batch
                for variant,final_only,reuse in [('lookup',False,False),('lookup_final',True,False),('exact_all',True,True)]:
                    actual,actual_inputs=infer_prepared_batch(model,prepare_cpu_batch(pairs,compact=True),
                                                              'cuda',final_only=final_only,reuse_pair_buffer=reuse)
                    for (name,_,_),a,b,inp in zip(group,reference,actual,actual_inputs):
                        coords=np.array_equal(a['atom37'],b['atom37'])
                        confidence=np.array_equal(a.get('plddt',[]),b.get('plddt',[]))
                        masks=np.array_equal(a['atom37_atom_exists'],b['atom37_atom_exists'])
                        pa=cli.generate_pdb(a,inp);pb=cli.generate_pdb(b,inp)
                        row=dict(model=model_name,variant=variant,name=name,batch_cap=cap,
                                 actual_batch_size=len(group),coordinates_exact=coords,confidence_exact=confidence,
                                 masks_exact=masks,pdb_bytes_exact=pa==pb,
                                 pdb_sha256=hashlib.sha256(pb.encode()).hexdigest(),
                                 max_coordinate_difference=float(np.abs(a['atom37']-b['atom37']).max()))
                        rows.append(row)
                        assert coords and confidence and masks and pa==pb,row
        del model
        torch.cuda.empty_cache()
    (args.output/'exact_agreement_251026.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'passed':True,'comparisons':len(rows),'max_coordinate_difference':
                      max(x['max_coordinate_difference'] for x in rows)}),flush=True)


if __name__=='__main__':main()
