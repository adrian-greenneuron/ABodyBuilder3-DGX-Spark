"""Download/verify the authors' archive and extract the published 100 test tensors.

Uses only Python's standard library. Existing archives can be reused without
downloading. Never extracts arbitrary tar paths or overwrites existing tensors.
The released tensors must be loaded only in the pinned trusted-data runtime.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ARCHIVE='https://zenodo.org/records/11354577/files/data.tar.gz?download=1'
SPLIT='https://raw.githubusercontent.com/Exscientia/abodybuilder3/main/data/split.csv'
MD5='73d66051b6e8cb435f1804e10ee4aa2f'
SPLIT_SHA256='621e2509620245403abfc65e32936ad8fa2b57ed00a5afbee1ea29f5bbf5c31a'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archive',type=Path,help='Reuse the complete 3.02 GB published archive.')
    parser.add_argument('--verify-only',action='store_true',help='Check an existing prepared dataset without modifying it.')
    args=parser.parse_args();root=args.output
    archive=args.archive or root/'upstream_data.tar.gz'
    if args.verify_only:
        split=(root/'split.csv').read_bytes()
    else:
        root.mkdir(parents=True,exist_ok=True)
        if (root/'experimental').exists():raise FileExistsError('Choose a fresh output or --verify-only.')
        with urllib.request.urlopen(SPLIT,timeout=60) as response:split=response.read()
        if not archive.exists():
            with urllib.request.urlopen(ARCHIVE,timeout=120) as source,archive.open('xb') as dest:
                while block:=source.read(8*1024*1024):dest.write(block)
    assert hashlib.sha256(split).hexdigest()==SPLIT_SHA256,'Upstream split changed; investigate before using another set.'
    with archive.open('rb') as file:checksum=hashlib.file_digest(file,'md5').hexdigest()
    assert checksum==MD5,'Archive checksum mismatch.'
    ids=[r['structure'] for r in csv.DictReader(io.StringIO(split.decode())) if r['split']=='test']
    assert len(ids)==100 and len(set(ids))==100
    if args.verify_only:
        assert json.loads((root/'test_ids_251026.json').read_text())==ids
        manifest=json.loads((root/'experimental_manifest_251026.json').read_text())
        for name in ids:
            data=(root/'experimental'/(name+'.pt')).read_bytes()
            assert hashlib.sha256(data).hexdigest()==manifest[name]['sha256'],name
        print(json.dumps({'verified':True,'targets':len(ids),'archive_md5':checksum}),flush=True)
        return
    dest=root/'experimental';dest.mkdir(exist_ok=False);found={}
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            path=Path(member.name)
            if not member.isfile() or not member.name.startswith('data/structures/structures/') or path.suffix!='.pt' or path.stem not in ids:continue
            if path.stem in found:raise ValueError('Duplicate experimental archive member.')
            data=tar.extractfile(member).read();(dest/path.name).write_bytes(data)
            found[path.stem]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'archive_member':member.name}
    assert set(found)==set(ids),'Missing test tensors.'
    (root/'split.csv').write_bytes(split)
    (root/'test_ids_251026.json').write_text(json.dumps(ids,indent=2)+'\n')
    (root/'experimental_manifest_251026.json').write_text(json.dumps(found,indent=2)+'\n')
    print(json.dumps({'completed':True,'targets':len(ids),'archive_md5':checksum}),flush=True)


if __name__=='__main__':main()
