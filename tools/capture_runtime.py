"""Read the retained image without changing it; save public source and identities."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path('/opt/abodybuilder3')
dest = Path('/capture')
source = dest / 'original_source'
source.mkdir(exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def command(args):
    p = subprocess.run(args, capture_output=True, text=True)
    return {'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

tracked = command(['git', '-C', str(root), 'ls-files', '-z'])
paths = tracked['stdout'].split('\0') if tracked['exit_code'] == 0 else []
if not paths:
    paths = [str(p.relative_to(root)) for p in root.rglob('*')
             if p.is_file() and not any(x in p.parts for x in ['.git', 'output', '__pycache__'])]
# The pinned model checkout predates this optional module. Docker COPY adds it
# without changing the model commit; include it in the deployment identity.
for optional in ('scripts/structure_preparation.py', 'scripts/batch_pipeline.py',
                 'src/abodybuilder3/batched_inference.py', 'src/abodybuilder3/inference_features.py'):
    if (root / optional).is_file():
        paths = sorted(set(paths) | {optional})
hashes = {}
for name in paths:
    p = root / name
    if p.is_file() and '.git' not in p.parts:
        hashes[name] = sha(p)
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
weights = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
           for p in sorted((root / 'output').rglob('*.ckpt'))]
versions = {d.metadata['Name']: d.version for d in metadata.distributions() if d.metadata.get('Name')}
direct = {}
for name in ['OpenMM', 'pdbfixer', 'abodybuilder3']:
    try:
        d = metadata.distribution(name)
        direct[name] = d.read_text('direct_url.json')
    except metadata.PackageNotFoundError:
        direct[name] = None
import openmm.version
manifest = {
    'git_head': command(['git', '-C', str(root), 'rev-parse', 'HEAD']),
    'git_status': command(['git', '-C', str(root), 'status', '--porcelain']),
    'source_hashes': hashes, 'weights': weights, 'versions': versions,
    'direct_urls': direct, 'openmm_full_version': openmm.version.full_version,
    'openmm_git_revision': openmm.version.git_revision,
    'reduce_version': command(['reduce', '-version']),
    'reduce_binary_sha256': sha(Path('/usr/local/bin/reduce')),
    'reduce_dictionary_sha256': sha(Path('/usr/local/reduce_wwPDB_het_dict.txt')),
    'hf_cached_repositories': [str(p) for p in Path('/root/.cache/huggingface/hub').glob('models--*')],
}
(dest / 'original_runtime_manifest_251026.json').write_text(json.dumps(manifest, indent=2) + '\n')
for p in [dest, *dest.rglob('*')]:
    os.chown(p, int(os.environ.get("CAPTURE_UID", "1000")), int(os.environ.get("CAPTURE_GID", "1000")))
print(json.dumps({'source_files': len(hashes), 'weights': len(weights),
                  'git_head': manifest['git_head'],
                  'openmm': manifest['openmm_full_version'], 'direct_urls': direct}, indent=2))
