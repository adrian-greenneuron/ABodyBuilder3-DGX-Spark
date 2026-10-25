"""Queue finite isolated conditions, then score all saved raw predictions."""
import argparse
import json
from pathlib import Path
import subprocess
import time

IMAGE='sha256:0a3ffb4edcef8d608d65eca49787fcb4ea820098a63e6dfdca853baee0527afa'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--measurement-tools',type=Path,required=True)
    p.add_argument('--component',type=Path)
    p.add_argument('--image',default=IMAGE,help='Pinned reference image, or a verified rebuild of dev 07bfe1e.')
    p.add_argument('--variants',nargs='+',default=['reference','accepted','attention','point_ieee','compiled','strict_reference','strict_attention'])
    args=p.parse_args();root=args.study.resolve();frozen=(args.component or root/'frozen_component').resolve()
    assert frozen.is_dir()
    image=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    common=['docker','run','--rm','--network','none','--user','1000:1000',
        '-e','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
        '-e','PYTHONPATH=/measurement:/component/tools',
        '-v',f'{frozen}:/component:ro','-v',f'{root}:/study',
        '-v',f'{args.measurement_tools.resolve()}:/measurement:ro']
    variants=args.variants
    records=json.loads((root/'commands_251026.json').read_text()) if (root/'commands_251026.json').exists() else []
    for model in ['base','plddt']:
        (root/model).mkdir(exist_ok=True)
        for variant in variants:
            dest=root/model/variant
            if (dest/'metadata_251026.json').exists():
                print('Reusing completed condition',model,variant,flush=True)
                continue
            assert not dest.exists(),f'Incomplete condition must be investigated: {dest}'
            command=common+['--gpus','all','-e',f'TORCHINDUCTOR_CACHE_DIR=/study/{model}/{variant}/compiler_cache',
                '-e',f'TRITON_CACHE_DIR=/study/{model}/{variant}/triton_cache',
                '-e','TORCHINDUCTOR_COMPILE_THREADS=4',
                '--entrypoint','python',image,'-u','/component/tools/check_structures.py',
                '--data','/study/data','--output',f'/study/{model}/{variant}',
                '--model',model,'--variant',variant]
            start=time.time();print('Starting',model,variant,flush=True)
            with (root/f'{model}_{variant}_251026.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            records.append({'model':model,'variant':variant,'exit_status':result.returncode,
                'wall_seconds':time.time()-start,'command':command,'image':image})
            (root/'commands_251026.json').write_text(json.dumps(records,indent=2)+'\n')
            if result.returncode:
                print((root/f'{model}_{variant}_251026.log').read_text()[-14000:],flush=True)
                raise SystemExit(result.returncode)
            assert (dest/'metadata_251026.json').exists()
            print('Completed',model,variant,round(time.time()-start,2),'s',flush=True)
    command=common+['--entrypoint','python',image,'-u','/component/tools/analyse_structures.py','--study','/study']
    with (root/'analysis_251026.log').open('w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    completed=sum(1 for model in ['base','plddt'] for folder in (root/model).iterdir() if (folder/'metadata_251026.json').exists())
    (root/'completed_251026.json').write_text(json.dumps({'completed':True,'conditions':completed})+'\n')
    print('All conditions and structural scoring completed',flush=True)


if __name__=='__main__':main()
