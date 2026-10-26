"""Freeze diverse natural heavy/light inputs from committed public OAS pairs."""
from pathlib import Path
import json,csv,hashlib,collections,math,time,argparse
import duckdb,pyarrow as pa
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True);parser.add_argument('--structure-metadata',type=Path,required=True);args=parser.parse_args();root=args.study.resolve()
manifest=json.loads((root/'paired_sources_261026.json').read_text());files=[f['path'] for f in manifest['files']]
(root/'inputs').mkdir();db=duckdb.connect(str(root/'selection.duckdb'));db.execute("SET threads=8");db.execute("SET memory_limit='6GB'");db.execute(f"SET temp_directory='{root}/duckdb_tmp'")
print('READ_NATIVE_PAIRS',len(files),flush=True)
db.execute('''CREATE TABLE eligible AS SELECT * FROM read_parquet(?, union_by_name=false)
 WHERE quality_tier='complete' AND NOT benchmark_excluded AND light_chain IN ('IGK','IGL')
 AND regexp_full_match(heavy_sequence,'[ACDEFGHIKLMNPQRSTVWY]+')
 AND regexp_full_match(light_sequence,'[ACDEFGHIKLMNPQRSTVWY]+')
 QUALIFY row_number() OVER(PARTITION BY pair_id ORDER BY study,unit_id,row_index)=1''',[files])
counts=dict(db.execute('SELECT study,count(*) FROM eligible GROUP BY study ORDER BY study').fetchall());print('ELIGIBLE_DISTINCT',sum(counts.values()),counts,flush=True)
target=100000;assert sum(counts.values())>=target
quota={k:0 for k in counts};remaining=target
while remaining:
 active=[k for k in counts if quota[k]<counts[k]];assert active
 amount=max(1,remaining//len(active))
 for k in active:
  add=min(amount,counts[k]-quota[k],remaining);quota[k]+=add;remaining-=add
  if remaining==0:break
q=pa.table({'study':list(quota),'quota':list(quota.values())});db.register('quotas',q)
rows=db.execute('''WITH ranked AS (SELECT *,row_number() OVER(PARTITION BY study ORDER BY sha256('abodybuilder3-261026:'||hex(pair_id)),unit_id,row_index) AS pick_rank FROM eligible)
 SELECT ranked.* EXCLUDE(quality_tier,benchmark_excluded),quota FROM ranked JOIN quotas USING(study)
 WHERE pick_rank<=quota ORDER BY pick_rank,sha256('study:'||study),hex(pair_id)''').fetch_arrow_table().to_pylist()
assert len(rows)==target;seen=set();meta=[]
for i,row in enumerate(rows):
 h=row['heavy_sequence'];l=row['light_sequence'];hid=hashlib.sha256(h.encode()).digest();lid=hashlib.sha256(l.encode()).digest();pid=hashlib.sha256(hid+lid).digest()
 assert row['heavy_id']==hid and row['light_id']==lid and row['pair_id']==pid
 assert pid not in seen;seen.add(pid)
 row['name']=f'oas_{i+1:06d}_{pid.hex()[:12]}'
 meta.append({'name':row['name'],'pair_id':pid.hex(),'unit_id':row['unit_id'],'study':row['study'],'row_index':row['row_index'],'light_chain':row['light_chain'],'heavy_length':len(h),'light_length':len(l),'length':len(h)+len(l)})
input_records=[]
def write_csv(name,selected):
 path=root/'inputs'/name
 with path.open('w',newline='') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['name','heavy','light']);w.writerows((r['name'],r['heavy_sequence'],r['light_sequence']) for r in selected)
 return {'file':name,'rows':len(selected),'distinct_pairs':len({(x['heavy_sequence'],x['light_sequence']) for x in selected}),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'min_length':min(len(x['heavy_sequence'])+len(x['light_sequence']) for x in selected),'max_length':max(len(x['heavy_sequence'])+len(x['light_sequence']) for x in selected),'study_counts':dict(collections.Counter(x['study'] for x in selected)),'light_counts':dict(collections.Counter(x['light_chain'] for x in selected)),'length_histogram':dict(sorted(collections.Counter(len(x['heavy_sequence'])+len(x['light_sequence']) for x in selected).items()))}
for n in [1000,10000,100000]:input_records.append(write_csv(f'oas_{n}.csv',rows[:n]))
# Independently include every valid distinct natural pair in the released structure metadata.
source=args.structure_metadata.resolve();unique={}
for x in csv.DictReader(source.open()):
 h=x['heavy'];l=x['light']
 if h and l and set(h+l)<=set('ACDEFGHIKLMNPQRSTVWYX'):
  unique.setdefault((h,l),{'name':'struct_'+x['structure'],'heavy_sequence':h,'light_sequence':l,'study':x['species'],'light_chain':x['light_chain_type']})
structural=list(unique.values());structural.sort(key=lambda x:hashlib.sha256((x['heavy_sequence']+'|'+x['light_sequence']).encode()).hexdigest())
input_records.append(write_csv('structural_diverse.csv',structural))
(root/'pair_selection_local_261026.json').write_text(json.dumps(meta,indent=2)+'\n')
selection={'created_unix':time.time(),'scope':'Throughput and memory engineering panels, not model-accuracy benchmarks. OAS selection is diversity balanced by study, not a natural repertoire-frequency sample. Native pairing preserved; no synthetic combinations or sequence repeats within any panel.','source_manifest_sha256':hashlib.sha256((root/'paired_sources_261026.json').read_bytes()).hexdigest(),'upstream_structure_csv_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'eligible_unique_pairs':sum(counts.values()),'eligible_counts':counts,'study_quotas':quota,'global_deduplication':'Exact pair identity; lexicographically first study/unit/row occurrence kept. Original source evidence retained locally.','ordering':'Smallest seeded SHA256 per study; equal study quotas capped at availability; pick-rank interleaving. Nested 1k/10k/100k CSVs.','filters':'Committed native pairs, both complete, not benchmark-excluded, canonical 20AA, IGK/IGL. Historical source quality is not a new annotation audit.','selection_metadata_scope':'Sequence IDs and original row/unit metadata local only; report aggregate counts and input hashes.','inputs':input_records,'duckdb_version':duckdb.__version__}
(root/'selection_261026.json').write_text(json.dumps(selection,indent=2)+'\n');print('ALL_INPUTS_PREPARED',[(x['file'],x['rows'],x['min_length'],x['max_length']) for x in input_records],flush=True)
db.close()
