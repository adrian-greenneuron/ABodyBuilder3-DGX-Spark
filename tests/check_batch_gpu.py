"""Bounded numerical qualification of dev batching in the pinned GPU runtime."""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
import ml_collections
from abodybuilder3.utils import string_to_input
from abodybuilder3.batched_inference import prepare_cpu_batch, infer_prepared_batch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.serialization.add_safe_globals([ml_collections.ConfigDict])
    sys.path.insert(0,'/opt/abodybuilder3/scripts')
    spec=importlib.util.spec_from_file_location('cli','/opt/abodybuilder3/scripts/predict.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    from batch_pipeline import BatchPolicy, plan_batches
    fixtures=json.loads(Path('/tests/benchmark_antibodies.json').read_text())
    cases=[(x['name'],x['heavy'],x['light']) for x in fixtures]
    first=fixtures[0]
    # Artificial length changes exercise padding, not structural accuracy.
    cases += [('padding_short',first['heavy'][:-20],first['light'][:-10]),
              ('padding_long',first['heavy']+'G'*20,first['light']+'S'*15)]
    batch,inputs=prepare_cpu_batch([(h,l) for _,h,l in cases])
    for i,(name,h,l) in enumerate(cases):
        ref=string_to_input(h,l,device='cpu');n=len(h)+len(l)
        assert torch.equal(batch['single'][i,:n],ref['single'][0]),name
        assert torch.equal(batch['pair'][i,:n,:n],ref['pair'][0]),name
        assert not batch['single'][i,n:].count_nonzero(),name
        assert not batch['pair'][i,n:].count_nonzero(),name
        assert not batch['pair'][i,:,n:].count_nonzero(),name
    rows=[]
    for model_name in ('base','plddt'):
        torch.manual_seed(42)
        model=cli.load_model(model_name,torch.device('cuda'))
        generated = []
        for group in plan_batches(cases, BatchPolicy()):
            outs, ins = infer_prepared_batch(model, prepare_cpu_batch([(h,l) for _,h,l in group]), 'cuda')
            generated.extend(zip(group, outs, ins))
        for (name,h,l),out,inp in generated:
            ref_in=cli.prepare_input(h,l)
            ref=cli.run_inference(model,ref_in,torch.device('cuda'))
            coords=ref['atom37'].detach().cpu().numpy()
            mask=ref['atom37_atom_exists'].detach().cpu().numpy()
            assert np.array_equal(mask,out['atom37_atom_exists']),name
            assert np.array_equal(inp['aatype'],ref_in['aatype']),name
            maximum=float(np.abs(coords[mask.astype(bool)]-out['atom37'][mask.astype(bool)]).max())
            ca=float(np.abs(coords[:,1,:]-out['atom37'][:,1,:]).max())
            assert ca < .01,(model_name,name,ca)
            assert maximum < .06,(model_name,name,maximum)
            # CPU transfer must preserve serial PDB bytes exactly.
            raw=cli.output_to_pdb(ref,ref_in)
            assert cli.generate_pdb(ref,ref_in)==raw,name
            (args.output/f'{model_name}_{name}.pdb').write_text(cli.generate_pdb(out,inp))
            score_diff = None
            if 'plddt' in ref:
                def score(x):
                    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
                    x = x.reshape(-1,50); e=np.exp(x-x.max(axis=-1,keepdims=True))
                    return (e/e.sum(axis=-1,keepdims=True)*np.arange(1,100,2)).sum(axis=-1)
                score_diff=float(np.abs(score(ref['plddt'])-score(out['plddt'])).max())
                assert score_diff < .1,(name,score_diff)
            rows.append(dict(model=model_name,name=name,max_plddt_score_difference=score_diff,max_ca_coordinate_difference=ca,
                             max_atom_coordinate_difference=maximum,
                             atom_coordinate_rms=float(np.sqrt(np.mean((coords[mask.astype(bool)]-out["atom37"][mask.astype(bool)])**2)))))
        del generated,outs,model,ref
        torch.cuda.empty_cache()
    (args.output/'agreement.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert max(x['atom_coordinate_rms'] for x in rows) < .01, rows
    print(json.dumps({'passed':True,'comparisons':len(rows),'max_ca':max(x['max_ca_coordinate_difference'] for x in rows)}),flush=True)

if __name__=='__main__':main()
