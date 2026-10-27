"""Package compact sequence-free evidence and preserve the prior component code."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--component',type=Path,required=True)
    args=parser.parse_args();root=args.root;component=args.component
    parent='967b4fe3ddef04888376f1d01122d8ee5fcc1bc2'
    for name in ['openmm_modeller.py','structure_preparation.py']:
        before=subprocess.check_output(['git','show',f'{parent}:candidate/{name}'],cwd=component)
        reference=component/'reference'/('gpu_'+name);reference.write_bytes(before)
        candidate=component/'candidate'/name
        patch=''.join(difflib.unified_diff(reference.read_text().splitlines(True),candidate.read_text().splitlines(True),fromfile=str(reference.relative_to(component)),tofile=str(candidate.relative_to(component))))
        (component/'candidate'/('hydrogen_geometry_'+name.removesuffix('.py')+'_271026.patch')).write_text(patch)
    source={}
    for directory in ['reference','candidate','tests','tools']:
        for path in sorted((component/directory).glob('*.py')):
            source[str(path.relative_to(component))]=digest(path.read_bytes())
    for path in component.glob('Dockerfile*'):source[path.name]=digest(path.read_bytes())
    images={}
    immutable={
        'earlier':'sha256:3b13290a647b29766a8d0c65689fbbe0dea40bdf600318a9b26dcc0bd6ce2ce4',
        'robust_r3':'sha256:b794a0b09039f7fadc086384812a0708591c724bccca9d887f1028e562de72bb',
        'robust_r4':'sha256:4f8f1031be5cfd43740739b4e46d98d03b5ac1bf050151345827165125256637',
        'planar_r7':'sha256:ef13b76eb63cedfd4cb089b0f03511e81d4968070dd52e3207e0c968c1e39b23',
        'adaptive_r8':'sha256:ca85c1a16c618bb6674ca48432e43e7db5d7911821a1a96cf7e1374d580dd558',
    }
    snapshots={}
    runtime_files={'openmm_modeller.py':'/usr/local/lib/python3.12/dist-packages/openmm/app/modeller.py',
                   'structure_preparation.py':'/opt/abodybuilder3/scripts/structure_preparation.py',
                   'hydrogen_geometry.py':'/opt/abodybuilder3/scripts/hydrogen_geometry.py'}
    for label,image in immutable.items():
        inspect=json.loads(subprocess.check_output(['docker','image','inspect',image]))[0]
        assert inspect['Id']==image
        hashes={}
        for name,path in runtime_files.items():
            if label=='earlier' and name=='hydrogen_geometry.py':continue
            content=subprocess.check_output(['docker','run','--rm','--network','none','--entrypoint','/bin/cat',image,path])
            hashes[name]=digest(content);snapshots[f'runtime_sources/{label}/{name}']=content
            if label=='adaptive_r8':assert digest((component/'candidate'/name).read_bytes())==hashes[name],name
        images[label]={'id':image,'created':inspect['Created'],'source_sha256':hashes}
    protected={}
    for label,repo,expected in [('main','ABodyBuilder3-DGX-Spark','c20d119957bcbbd24d8534fddbdc3e9fc7d57795'),('dev','abodybuilder3-dev','e6fabdf73aa9590431377de5f923c256576829fd')]:
        path=component.parent/repo
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
        assert head==expected and not subprocess.check_output(['git','status','--porcelain'],cwd=path)
        protected[label]=head
    for tag,expected in [('abodybuilder3-dev-spark:latest','sha256:a4d29734d1b2ce98a09c4508e9047ad714a6da94be610fd5b8dda6b15285b590'),('abodybuilder3-official-spark:latest','sha256:ab981f1494f2c817995a3cf767702181330af8d3f97584120d533dedd9432fd0')]:
        identity=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',tag],text=True).strip()
        assert identity==expected
        protected[tag]=identity
    manifest={'generation_date':'2026-10-27','component_parent':parent,'branch':'components/hydrogen-pipeline-261026',
              'sha256':source,'images':images,'protected_unchanged':protected,
              'model_checkpoint_sha256':'421aa6c52a5152f781b35007411985146fb520a90286a955161cb8e50584d615',
              'AOT_manifest_sha256':'947bbaf04b6633cb9b0a073fb62d090b55610e1b2964b750466c95840bc6203e',
              'build_note':'Dockerfile.geometry is an incremental layer on the retained earlier GPU image. Verify the immutable base identity before using its tag. Historical image sources are included separately; current tools do not retroactively identify old runs.',
              'licence_note':'Original OpenMM Modeller and MIT notice retained; Probe source is unmodified upstream code, not a candidate rewrite.'}
    path=component/'docs/data/hydrogen_geometry_sources_271026.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    entries=dict(snapshots)
    for relative in ['protocol_271026.json','planar_preflight_271026.json','planar_cohort_manifest_271026.json','measurement_filename_correction_271026.json','steric_probe_271026.json','probe_steric127_271026.json']:
        entries[relative]=(root/relative).read_bytes()
    for path in root.glob('tests*_271026.log'):entries[path.name]=path.read_bytes()
    for path in root.glob('build*_271026.log'):entries[path.name]=path.read_bytes()
    entries['final_analysis_271026.log']=(root/'final_analysis_271026.log').read_bytes()
    for stage in sorted(p.parent for p in root.glob('*/cli_protocol_271026.json')):
        prefix=stage.name+'/'
        entries[prefix+'cli_protocol_271026.json']=(stage/'cli_protocol_271026.json').read_bytes()
        records=stage/'cli_measurements_271026.json'
        if records.exists():
            data=json.loads(records.read_text())
            for row in data['rows']:row.pop('audits',None)
            entries[prefix+'compact_measurements_271026.json']=(json.dumps(data,indent=2)+'\n').encode()
        if (stage/'cli').exists():
            for case in (stage/'cli').iterdir():
                if not case.is_dir():continue
                for filename in ['command_271026.json','execution.json','system_memory_trace_271026.csv','gpu_utilisation.csv','batching.json']:
                    path=case/filename
                    if path.exists():entries[prefix+'cli/'+case.name+'/'+filename]=path.read_bytes()
                failures=[]
                for path in (case/'pdbs').glob('*.preparation.json'):
                    audit=json.loads(path.read_text())
                    if audit['status']=='failed':failures.append({'target':path.name,'status':'failed','error':audit.get('error'),'raw_sha256':audit['raw_sha256']})
                if failures:entries[prefix+'cli/'+case.name+'/failure_summary_271026.json']=(json.dumps(failures,indent=2)+'\n').encode()
    for path in (root/'planar_rejections').glob('*/rejection_checks_271026.json'):
        entries[str(path.relative_to(root))]=path.read_bytes()
    for directory in ['reference','candidate','tests','tools']:
        for path in (component/directory).glob('*.py'):entries['source/'+str(path.relative_to(component))]=path.read_bytes()
    entries['source/Dockerfile.geometry']=(component/'Dockerfile.geometry').read_bytes()
    for path in [component/'docs/hydrogen_geometry_271026.md',component/'docs/data/hydrogen_geometry_results_271026.json',component/'docs/data/hydrogen_geometry_sources_271026.json']:
        entries[str(path.relative_to(component/'docs'))]=path.read_bytes()
    for path in (component/'docs/assets').glob('hydrogen_geometry_*_271026.*'):
        entries['assets/'+path.name]=path.read_bytes()
    for name,data in entries.items():
        assert not name.endswith(('.pdb','.npz','.ckpt'))
        if name.endswith('.csv'):assert name.endswith(('system_memory_trace_271026.csv','gpu_utilisation.csv'))
        if name.endswith(('.json','.log')):
            text=data.decode();assert not re.search(r'[ACDEFGHIKLMNPQRSTVWY]{60,}',text),name
            assert not any(key in text for key in ['"hydrogen_groups":','"moved_heavy_atoms":','"missing_terminal_atoms":']),name
    hashes={name:digest(data) for name,data in entries.items()}
    entries['SHA256.json']=(json.dumps(hashes,indent=2)+'\n').encode()
    entries['README.txt']=b'Protocols, compact measurements, commands, telemetry, failed-attempt summaries, test logs and exact runtime sources. No input CSVs, PDBs, checkpoints or per-residue identity lists. Failed attempts and parameter trials remain labelled; selected completed measurements are identified in data/hydrogen_geometry_results_271026.json. Main and dev are unchanged.\n'
    archive=component/'docs/data/hydrogen_geometry_evidence_271026.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
        for name,data in sorted(entries.items()):stream.writestr(name,data)
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
        for name,value in hashes.items():assert digest(stream.read(name))==value
    artifacts=[component/'README.md',component/'docs/hydrogen_geometry_271026.md']
    artifacts+=list((component/'docs/data').glob('hydrogen_geometry_*_271026.*'))
    artifacts+=list((component/'docs/assets').glob('hydrogen_geometry_*_271026.*'))
    checksums=component/'docs/data/hydrogen_geometry_checksums_271026.txt'
    checksums.write_text(''.join(f'{digest(p.read_bytes())}  {p.relative_to(component)}\n' for p in sorted(artifacts) if p!=checksums))
    print('Packaged',len(hashes),'files;',archive.stat().st_size,'bytes; protected refs/images unchanged.')


if __name__=='__main__':main()
