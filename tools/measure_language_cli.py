"""Capture the installed language CLI's exact embeddings and raw predictions."""
import time
START=time.perf_counter()
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from benchmark_runtime import Memory
from prediction_fingerprint import fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',choices=['trainer','direct'],required=True)
    parser.add_argument('--precision',choices=['strict','tf32'],default='strict')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for name in ('pdbs','raw','embeddings'):(args.output/name).mkdir()
    spec=importlib.util.spec_from_file_location('installed_language_cli','/opt/abodybuilder3/scripts/predict.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    current={};fingerprints=[];stages=[]
    original_single=cli.process_single_antibody
    def single(*values,**options):
        current['name']=values[0]
        return original_single(*values,**options)
    cli.process_single_antibody=single
    original_load=cli.load_prott5
    def load(*values,**options):
        start=time.perf_counter();encoder=original_load(*values,**options)
        stages.append({'stage':'encoder_load','seconds':time.perf_counter()-start})
        original_embeddings=encoder.get_embeddings
        def embeddings(*values,**options):
            start=time.perf_counter();result=original_embeddings(*values,**options)
            value=result[0].detach().cpu().numpy()
            np.save(args.output/'embeddings'/(current['name']+'.npy'),value)
            stages.append({'stage':'embedding','name':current['name'],'seconds':time.perf_counter()-start})
            return result
        encoder.get_embeddings=embeddings
        return encoder
    cli.load_prott5=load
    original_infer=cli.run_inference
    def infer(model,inputs,device):
        result=original_infer(model,inputs,device)
        name=current['name'];fingerprints.append({'name':name,**fingerprint(result,inputs)})
        np.savez(args.output/'raw'/(name+'.npz'),positions=result['atom37'].detach().cpu().numpy(),
                 masks=result['atom37_atom_exists'].detach().cpu().numpy(),heavy=inputs['is_heavy'].cpu().numpy().astype(bool),
                 aatype=inputs['aatype'].cpu().numpy(),confidence_logits=result.get('plddt',torch.empty(0)).detach().cpu().numpy())
        return result
    cli.run_inference=infer
    sys.argv=['predict.py','predict','--csv',str(args.input),'--output',str(args.output/'pdbs'),
              '--model','language','--workers','1','--batch-size','1','--seed','42','--quiet',
              '--embedding-runtime',args.runtime,'--precision',args.precision]
    with Memory() as memory:
        try:cli.app()
        except SystemExit as error:
            if error.code not in (0,None):raise
        finished=time.perf_counter()
    result={'completed':True,'runtime':args.runtime,'precision':args.precision,'count':len(fingerprints),
            'script_seconds':finished-START,'stages':stages,'torch_cpu_threads':torch.get_num_threads(),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),**memory.result()}
    (args.output/'metrics_261026.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'fingerprints_261026.json').write_text(json.dumps(fingerprints,indent=2)+'\n')


if __name__=='__main__':main()
