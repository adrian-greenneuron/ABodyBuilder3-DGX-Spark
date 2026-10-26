"""Observe process-tree proportional resident pages; leave inference unchanged."""
from pathlib import Path
import sys,runpy,json,time,threading
import psutil
root=psutil.Process();stop=threading.Event();samples=[];errors=[]
def sample():
    values={'pss':0,'private':0,'rss':0};count=0
    for p in [root,*root.children(recursive=True)]:
        try:
            lines=Path(f'/proc/{p.pid}/smaps_rollup').read_text().splitlines()
            fields={k:int(v.split()[0])*1024 for line in lines if ':' in line for k,v in [line.split(':',1)]}
            values['pss']+=fields['Pss'];values['rss']+=fields['Rss'];values['private']+=fields.get('Private_Clean',0)+fields.get('Private_Dirty',0);count+=1
        except (OSError,psutil.Error,KeyError):pass
    samples.append({'monotonic':time.monotonic(),'processes':count,**values})
def loop():
    while not stop.wait(.2):sample()
sample();thread=threading.Thread(target=loop,daemon=True);thread.start();completed=False
try:
    sys.argv=['/opt/abodybuilder3/scripts/predict.py',*sys.argv[1:]]
    try:runpy.run_path('/opt/abodybuilder3/scripts/predict.py',run_name='__main__');completed=True
    except SystemExit as e:
        if e.code not in (0,None):raise
        completed=True
finally:
    stop.set();thread.join();sample()
    Path('/results/process_pss_261026.json').write_text(json.dumps({'completed':completed,'samples':samples,'peak_pss_bytes':max(x['pss'] for x in samples),'peak_private_bytes':max(x['private'] for x in samples),'peak_summed_rss_bytes':max(x['rss'] for x in samples)},indent=2)+'\n')
