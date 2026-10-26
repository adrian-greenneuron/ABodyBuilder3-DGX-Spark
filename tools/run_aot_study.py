"""Prepare, package and compare AOT deployment in fresh retained runtimes.

All compiler preparation occurs outside timed inference jobs. Uses the same
real CLI, batch scheduler, native writer and retained inputs as the cached JIT
study. Only the execution adapter differs; main/dev are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(cmd, log):
    if log.exists():
        suffix=1
        while log.with_name(log.stem+f'_attempt{suffix}'+log.suffix).exists():suffix+=1
        shutil.copy2(log,log.with_name(log.stem+f'_attempt{suffix}'+log.suffix))
    with log.open('w') as output:
        result = subprocess.run(cmd,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-10000:],flush=True)
        raise RuntimeError(f'Command failed ({result.returncode}): {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--bulk-study',type=Path,required=True)
    parser.add_argument('--counts',type=int,nargs='+',default=[100,1000,10000])
    parser.add_argument('--resume',action='store_true',help='Reuse completed, verified preparation and jobs after a controller failure.')
    args = parser.parse_args()
    root = args.study.resolve()
    component = root/'frozen_component'
    original_images = json.loads((args.bulk_study/'images_261026.json').read_text())
    parents = {m:original_images[f'{m}/compiled_strict']['image'] for m in ['base','plddt']}
    parent_tags = {m:original_images[f'{m}/compiled_strict']['tag'] for m in ['base','plddt']}
    for model in parents:
        actual = subprocess.check_output(['docker','image','inspect',parent_tags[model],'--format','{{.Id}}'],text=True).strip()
        assert actual==parents[model], 'The tagged local parent must match the pinned image identity.'
    manifest = json.loads((root/'aot_packages_261026.json').read_text())
    def docker(image,dest,*,gpu=True):
        cmd = ['docker','run','--rm','--network','none','--ipc=host','--user','1000:1000']
        if gpu:cmd+=['--gpus','all']
        cmd+=['-e','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True','-e','PYTHONHASHSEED=0',
              '-e','ABB3_NATIVE_FORMATTER=1','-e','TORCHINDUCTOR_COMPILE_THREADS=4',
              '-v',f'{component}:/aot_component:ro','-v',f'{dest}:/results',
              '--entrypoint','python',image,'-u']
        return cmd
    # --resume retains the first base package from the feasibility test as well
    # as any other completed, verified package; a fresh study prepares all four.
    for model,domain in [('base','multi'),('base','single'),('plddt','multi'),('plddt','single')]:
        relative = Path(manifest[model][domain]).parent
        dest = root/relative
        if args.resume and (dest/'model.pt2').exists():
            metadata=json.loads((dest/'preparation_261026.json').read_text())
            assert metadata['model']==model and metadata['precision']=='strict_fp32'
            assert metadata['identity_cache_bypass_eager_exact']
            assert sha(dest/'model.pt2')==metadata['package_sha256']
            print('REUSE_PREPARATION',model,domain,flush=True)
            continue
        cmd = docker(parents[model],root)
        image_at = cmd.index('--entrypoint')
        cmd[image_at:image_at] = ['-e','TORCHINDUCTOR_CACHE_DIR=/results/compiler_cache',
                                  '-e','TRITON_CACHE_DIR=/results/triton_cache']
        cmd+=['/aot_component/tools/prepare_aot_package.py','--model',model,
              '--output',f'/results/{relative}','--dynamic','--batch-size','1' if domain=='single' else '32']
        (root/f'prepare_{model}_{domain}_command_261026.json').write_text(json.dumps(cmd,indent=2)+'\n')
        print('PREPARE',model,domain,flush=True)
        command(cmd,root/f'prepare_{model}_{domain}_261026.log')
        assert (dest/'model.pt2').is_file()
        print('PREPARED',model,domain,flush=True)
    # Bounded fresh-process checks for all four package domains.
    for model in ['base','plddt']:
        for domain in ['multi','single']:
            existing=root/f'check_{model}_{domain}/check_261026.json'
            if args.resume and existing.exists():
                record=json.loads(existing.read_text())
                assert all(v['repeat_exact'] for v in record['outputs'].values())
                print('REUSE_CHECK',model,domain,flush=True)
                continue
            cmd = docker(parents[model],root)
            cmd+=['/aot_component/tools/check_aot_package.py','--package',
                  f'/results/{manifest[model][domain]}','--output',f'/results/check_{model}_{domain}']
            print('CHECK',model,domain,flush=True)
            command(cmd,root/f'check_{model}_{domain}_261026.log')
            record = json.loads((root/f'check_{model}_{domain}/check_261026.json').read_text())
            assert all(v['repeat_exact'] for v in record['outputs'].values())
            print('CHECKED',model,domain,record['valid_atom_metrics'],flush=True)
    # Deploy only the small executable packages, not exported preparation graphs.
    images = {}
    for model in ['base','plddt']:
        context = root/f'package_context_{model}'
        (context/'tools').mkdir(parents=True,exist_ok=args.resume)
        for name in ['measure_pipeline.py','aot_model.py']:
            shutil.copy2(component/'tools'/name,context/'tools'/name)
        package_manifest = {model:{}}
        for domain in ['multi','single']:
            source = root/manifest[model][domain]
            dest = context/'packages'/model/domain
            dest.mkdir(parents=True,exist_ok=args.resume)
            for filename in ['model.pt2','preparation_261026.json']:
                shutil.copy2(source.parent/filename,dest/filename)
            assert sha(source)==sha(dest/'model.pt2')
            package_manifest[model][domain] = f'{model}/{domain}/model.pt2'
        (context/'packages/aot_packages_261026.json').write_text(json.dumps(package_manifest,indent=2)+'\n')
        (context/'Dockerfile').write_text('ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\nCOPY tools/ /component/tools/\nCOPY packages/ /aot_packages/\n')
        tag = f'abodybuilder3-component-spark:{model}-strict-fp32-aot-native-261026'
        command(['docker','build','--network','none','--pull=false','--build-arg',
                 f'BASE_IMAGE={parent_tags[model]}','-t',tag,str(context)],root/f'build_{model}_aot_261026.log')
        image = subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip()
        images[model] = {'image':image,'tag':tag,'parent_image':parents[model],
                         'package_sha256':{d:sha(root/manifest[model][d]) for d in ['multi','single']}}
        (root/'images_261026.json').write_text(json.dumps(images,indent=2)+'\n')
    records = json.loads((root/'measurements_261026.json').read_text()) if args.resume and (root/'measurements_261026.json').exists() else []
    for count in args.counts:
        for model in ['base','plddt']:
            for rep in [1,2]:
                order = ['compiled','aot'] if (rep==1)==(model=='base') else ['aot','compiled']
                for mode in order:
                    label = 'cached_compiled_native' if mode=='compiled' else 'aot_native'
                    dest = root/f'n{count}'/model/'strict'/label/f'run{rep}'
                    if any((r['count'],r['model'],r['mode'],r['replicate'])==(count,model,mode,rep) for r in records):
                        assert args.resume and (dest/'metrics_261026.json').is_file()
                        print('REUSE_JOB',count,model,mode,rep,flush=True)
                        continue
                    dest.mkdir(parents=True,exist_ok=False)
                    image = images[model]['image'] if mode=='aot' else parents[model]
                    cmd = docker(image,dest)
                    # Keep identical new measurement code for both conditions.
                    insert = cmd.index('--entrypoint')
                    cmd[insert:insert] = ['-v',f'{component}/tools/measure_pipeline.py:/component/tools/measure_pipeline.py:ro',
                                          '-v',f'{component}/tools/aot_model.py:/component/tools/aot_model.py:ro',
                                          '-e','ABB3_FORMATTER_USAGE=/results/formatter_usage.json']
                    if mode=='aot':
                        insert = cmd.index('--entrypoint')
                        cmd[insert:insert] = ['-e','TORCHINDUCTOR_CACHE_DIR=/results/unused_compiler_cache',
                                              '-e','TRITON_CACHE_DIR=/results/unused_triton_cache',
                                              '-e','AOTI_RUNTIME_CHECK_INPUTS=1']
                    cmd+=['/component/tools/measure_pipeline.py','--mode',mode,'--precision','strict',
                          '--model',model,'--count',str(count),'--output','/results','--date-code','261026']
                    if mode=='aot':cmd+=['--aot-root','/aot_packages']
                    (dest/'command_261026.json').write_text(json.dumps(cmd,indent=2)+'\n')
                    print('START',count,model,mode,rep,flush=True)
                    started = time.perf_counter()
                    command(cmd,dest/'inference.log')
                    elapsed = time.perf_counter()-started
                    metrics = json.loads((dest/'metrics_261026.json').read_text())
                    batching = json.loads((dest/'batching.json').read_text())
                    assert batching['saved']==count and not batching['oom_retries']
                    assert metrics['allow_tf32'] is False
                    assert metrics['float32_matmul_precision']=='highest'
                    assert json.loads((dest/'formatter_usage.json').read_text())['native_calls']==count
                    if mode=='aot':
                        assert metrics['dynamo_counters'].get('stats',{}).get('unique_graphs',0)==0
                    record = {'count':count,'model':model,'precision':'strict','mode':mode,
                              'label':label,'native_formatter':True,'replicate':rep,'image':image,
                              'container_seconds':elapsed,'relative_output':str(dest.relative_to(root)),
                              'metrics':metrics,'batching':batching}
                    records.append(record)
                    (root/'measurements_261026.json').write_text(json.dumps(records,indent=2)+'\n')
                    print('DONE',count,model,mode,rep,round(elapsed,3),flush=True)
    (root/'completed_261026.json').write_text(json.dumps({'completed':True,'jobs':len(records)})+'\n')


if __name__ == '__main__':main()
