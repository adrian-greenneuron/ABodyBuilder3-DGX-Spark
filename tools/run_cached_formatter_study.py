"""Scale packaged compilation plus the exact-output formatter to bulk jobs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

DEV_IMAGE='sha256:18456b5147ea1ec35d8de4f664962baeea6af1bb473ad9107bb1cbfa37fd844d'
NATIVE_IMAGE='sha256:a53b5613e495565b4ebf71eab5659a5d81b8ecb09378fba6819a3cc82d40b8bf'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--cache-study',type=Path,required=True)
    p.add_argument('--formatter',type=Path,required=True)
    p.add_argument('--input-study',type=Path,required=True)
    args=p.parse_args();root=args.study.resolve();root.mkdir(parents=True,exist_ok=False)
    context=root/'package_context';context.mkdir()
    for name in ['candidate','reference','tools']:
        if name=='tools':
            (context/name).mkdir();shutil.copy2(args.formatter/name/'build_native.sh',context/name/'build_native.sh')
        else:shutil.copytree(args.formatter/name,context/name,ignore=shutil.ignore_patterns('*.so','__pycache__'))
    shutil.copy2(args.formatter/'native_activation.py',context/'native_activation.py')
    shutil.copytree(args.input_study/'frozen_dev',context/'inputs')
    (context/'Dockerfile').write_text('ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\nCOPY candidate/ /formatter/candidate/\nCOPY reference/ /formatter/reference/\nCOPY native_activation.py /formatter/native_activation.py\nCOPY tools/ /formatter/tools/\nRUN sh /formatter/tools/build_native.sh && python3 -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], \'abb3_formatter_experiment.pth\').write_text(\'/formatter\\nimport native_activation\\n\')"\nCOPY inputs/antibodies_10000.csv /frozen/antibodies_10000.csv\nCOPY inputs/input_identities_251026.json /frozen/input_identities_251026.json\nENV ABB3_NATIVE_FORMATTER=0\n')
    cached=json.loads((args.cache_study/'cached_images_261026.json').read_text())
    images={}
    for key,record in cached.items():
        tag='abodybuilder3-component-spark:'+key.replace('/','-')+'-cached-native-261026'
        with (root/(key.replace('/','_')+'_build.log')).open('w') as log:
            subprocess.run(['docker','build','--network','none','--pull=false','--build-arg',f"BASE_IMAGE={record['tag']}",'-t',tag,str(context)],check=True,stdout=log,stderr=subprocess.STDOUT)
        images[key]={'tag':tag,'image':subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip(),
                     'parent_cached_image':record['image']}
        (root/'images_261026.json').write_text(json.dumps(images,indent=2)+'\n')
    protocol={'scope':'100-input probes precede 10,000-input bulk comparison. Same 100 experimental antibodies repeated; no prediction cache.',
              'conditions':['current_dev','native_writer','cached_compiled_native','cached_combined_native'],
              'precision':['normal','strict'],'models':['base','plddt'],'repetitions':2,
              'batching':'Unchanged auto cap32, target6GiB, two CPU preparation and writer threads.',
              'current_dev_image':DEV_IMAGE,'native_writer_image':NATIVE_IMAGE,'cached_images':images,
              'main_and_dev_unchanged':True}
    (root/'protocol_261026.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records=[]
    def run(count,model,precision,mode,native,rep,label):
        key=f'{model}/{mode}_{precision}'
        image=images[key]['image'] if mode!='eager' else (NATIVE_IMAGE if native else DEV_IMAGE)
        dest=root/f'n{count}'/model/precision/label/f'run{rep}';dest.mkdir(parents=True)
        cmd=['docker','run','--rm','--network','none','--gpus','all','--ipc=host','--user','1000:1000',
             '-e','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True','-e','PYTHONHASHSEED=0',
             '-e',f'ABB3_NATIVE_FORMATTER={int(native)}','-e','ABB3_FORMATTER_USAGE=/results/formatter_usage.json']
        if mode=='eager':
            cmd+=['-e','PYTHONPATH=/component/tools:/measurement:/frozen','-e','TORCHINDUCTOR_CACHE_DIR=/results/compiler_cache',
                  '-v',f'{args.input_study.resolve()}/frozen_component:/component:ro',
                  '-v',f'{args.input_study.resolve()}/frozen_measurement:/measurement:ro',
                  '-v',f'{args.input_study.resolve()}/frozen_dev:/frozen:ro']
        cmd+=['-v',f'{dest}:/results','--entrypoint','python',image,'-u','/component/tools/measure_pipeline.py',
              '--mode',mode,'--precision',precision,'--model',model,'--count',str(count),'--output','/results','--date-code','261026']
        (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        print('START',count,model,precision,label,rep,flush=True);start=time.perf_counter()
        with (dest/'inference.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:
            print((dest/'inference.log').read_text()[-12000:],flush=True);raise SystemExit(r.returncode)
        metrics=json.loads((dest/'metrics_261026.json').read_text())
        if native:assert json.loads((dest/'formatter_usage.json').read_text())['native_calls']==count
        if mode!='eager':assert metrics['packaged_cache_manifest_sha256'] is not None
        row={'count':count,'model':model,'precision':precision,'mode':mode,'native_formatter':native,'label':label,
             'replicate':rep,'image':image,'container_seconds':time.perf_counter()-start,
             'relative_output':str(dest.relative_to(root)),'metrics':metrics,
             'batching':json.loads((dest/'batching.json').read_text())}
        assert row['batching']['saved']==count and not row['batching']['oom_retries']
        records.append(row);(root/'measurements_261026.json').write_text(json.dumps(records,indent=2)+'\n')
        print('DONE',count,model,precision,label,rep,round(row['container_seconds'],3),flush=True)
    for model in ['base','plddt']:
        for precision in ['normal','strict']:
            for mode in ['compiled','combined']:run(100,model,precision,mode,True,1,'probe_'+mode)
    conditions=[('eager',False,'current_dev'),('eager',True,'native_writer'),
                ('compiled',True,'cached_compiled_native'),('combined',True,'cached_combined_native')]
    for model in ['base','plddt']:
        for precision in ['normal','strict']:
            for rep in [1,2]:
                order=conditions if (rep==1)==(model=='base') else conditions[::-1]
                for mode,native,label in order:run(10000,model,precision,mode,native,rep,label)
    (root/'completed_261026.json').write_text(json.dumps({'completed':True,'jobs':len(records)})+'\n')


if __name__=='__main__':main()
