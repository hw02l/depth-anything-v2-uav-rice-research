from pathlib import Path
import os,sys,json,hashlib,subprocess,csv,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,torch

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
assert not torch.cuda.is_available(),'CPU-only analysis: do not consume GPU'
work=Path(__file__).resolve().parent.parent
source=Path(__file__).parent
project=work.parent.parent
spec=json.loads((work/'analysis_spec.json').read_text())
spec['rgb_root']=str(project/'data/uav3dcrop_rgb');spec['depth_root']=str(project/'data/uav3dcrop_depth')
assert spec['model_caps']=={'A':65,'B':80,'C':80}
assert sha(spec['manifest'])==spec['validation_sha256']
rows=list(csv.DictReader(Path(spec['manifest']).open(encoding='utf-8-sig')))
assert len(rows)==128 and all(r['split']=='validation' for r in rows)
weights={m:sha(Path(p)/'model.safetensors') for m,p in spec['models'].items()}
cache=Path(spec['output'])/'prediction_cache'
def verify(entry):
    m,i=entry;p=cache/f'{m}_{i:03d}.npy';ctx=json.loads(p.with_suffix('.json').read_text())
    assert ctx['weights_sha256']==weights[m] and ctx['manifest_sha256']==spec['validation_sha256']
    assert ctx['image_relpath']==rows[i]['image_relpath'] and ctx['image_size']==518 and ctx['model_cap']==spec['model_caps'][m] and ctx['amp_inference']==False
    a=np.load(p);assert a.shape==(518,686) and a.dtype==np.float32 and np.isfinite(a).all() and (a>0).all()
    return str(p),sha(p)
verified={};start=time.monotonic()
with ThreadPoolExecutor(max_workers=8) as pool:
    for future in as_completed([pool.submit(verify,(m,i)) for m in spec['models'] for i in range(128)]):
        p,h=future.result();verified[p]=h
        if len(verified)%64==0:print('PREDICTION_CACHE_VERIFY',len(verified),'/384',flush=True)
with (work/'prediction_cache_integrity.json').open('x') as f:json.dump({'models_weights_sha256':weights,'files':verified,'elapsed_seconds':time.monotonic()-start},f,indent=2)
specpath=work/'analysis_spec_cpu.json'
with specpath.open('x') as f:json.dump(spec,f,indent=2)
script=source/'shape_analysis_v5.py'
subprocess.run([sys.executable,'-u',str(script),'--spec',str(specpath),'--phase','analyze'],check=True)
print('STATISTICS_COMPLETE; creating representative comparison figures',flush=True)
subprocess.run([sys.executable,'-u',str(script),'--spec',str(specpath),'--phase','figures'],check=True)
print('ALL_CPU_ANALYSIS_COMPLETE',flush=True)
