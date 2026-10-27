"""Subset frozen native pairs for validation; no sequence generation or mixing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--exclude',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=256);args=p.parse_args()
    used={(r['heavy'],r['light']) for r in csv.DictReader(args.exclude.open())};valid=set('ACDEFGHIKLMNPQRSTVWY');eligible=[];seen=set();invalid=0
    for row in csv.DictReader(args.source.open()):
        key=(row['heavy'],row['light'])
        if key in used or key in seen:continue
        if not row['heavy'] or not row['light'] or not set(''.join(key))<=valid:invalid+=1;continue
        seen.add(key);eligible.append(row)
    eligible.sort(key=lambda r:(len(r['heavy'])+len(r['light']),r['name']))
    assert len(eligible)>=args.count
    selected=[eligible[int((i+.5)*len(eligible)/args.count)] for i in range(args.count)]
    with args.output.open('w') as f:
        w=csv.DictWriter(f,['name','heavy','light'],lineterminator='\n');w.writeheader()
        for i,row in enumerate(selected):w.writerow({'name':f'validation_{i:06d}','heavy':row['heavy'],'light':row['light']})
    assert not used & {(r['heavy'],r['light']) for r in selected}
    metadata={'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),'excluded_panel_sha256':hashlib.sha256(args.exclude.read_bytes()).hexdigest(),
              'output_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'count':args.count,'eligible_distinct_pairs':len(eligible),'invalid_excluded':invalid,
              'combined_length_range':[min(len(r['heavy'])+len(r['light']) for r in selected),max(len(r['heavy'])+len(r['light']) for r in selected)],
              'selection':'Deterministic equally spaced order-statistic positions over combined-length-sorted distinct unused native pairs. Partners preserved; not a probability sample.'}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata))


if __name__=='__main__':main()
