"""Byte-exact verification and one-way local/Colab exchange, without network or ML runs."""
from pathlib import Path
import ast
import csv
import hashlib
import io
import json
import re
import zipfile

FULL_RUN = 'metric_outdoor_full_20261014_s42_518_cap80_gt65'

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def verify_checkpoint(path):
    path = Path(path)
    marker = read_json(path / 'complete.json')
    expected = marker['file_sha256']
    names = ('model.safetensors', 'config.json', 'trainer_state.pt')
    for name in names:
        if not (path/name).is_file() or sha256(path/name) != expected.get(name):
            raise ValueError('Incomplete or mismatched checkpoint: ' + name)
    return {'completed_epoch': marker['completed_epoch'], 'verified_files': list(names)}

def verify(root, checkpoint=None):
    root = Path(root).resolve()
    snapshots = read_json(root/'checksums/source_snapshots.json')['snapshots']
    checked = parsed = 0
    for relative, hashes in snapshots.items():
        folder = root/relative
        actual_names = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        if actual_names != set(hashes):
            raise ValueError('Snapshot file set changed: '+relative)
        for name, digest in hashes.items():
            path = folder/name
            if sha256(path) != digest:
                raise ValueError('Snapshot bytes changed: '+relative+'/'+name)
            checked += 1
            if path.suffix == '.py':
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
                parsed += 1
    experiment = root/'experiments'/FULL_RUN
    metadata = read_json(experiment/'run_metadata/run_metadata.json')
    source = experiment/'source_full_v1'
    for name, digest in metadata['resume_signature']['code_sha256'].items():
        path = source/('config/scene_split_plan.json' if name == 'scene_split_plan.json' else name)
        if sha256(path) != digest:
            raise ValueError('Canonical run signature mismatch: '+name)
    for split in ('train','validation'):
        if sha256(source/f'config/verified_plan/{split}.csv') != metadata[split+'_manifest_sha256']:
            raise ValueError('Manifest mismatch: '+split)
    status = read_json(experiment/'run_metadata/run_status.json')
    if (status['status'],status['completed_epochs'],status['remaining_epochs']) != ('paused_after_epoch',5,15):
        raise ValueError('Unexpected saved run progress')
    args = metadata['arguments']
    for name,value in {'epochs':20,'max_depth':65.0,'model_max_depth':80.0,'batch_size':1,'grad_accumulation':4,'seed':42}.items():
        if args[name] != value:
            raise ValueError('Unexpected training setting: '+name)
    with (experiment/'results/epoch_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows = list(csv.DictReader(stream))
    if [int(r['epoch']) for r in rows] != [1,2,3,4,5]:
        raise ValueError('Epoch results are not exactly epochs 1--5')
    if abs(float(rows[-1]['validation_rmse_m'])-1.0094461878368737)>1e-10:
        raise ValueError('Unexpected epoch 5 RMSE')
    figures = list((root/'experiments/metric_shape_20261013T123006472484Z/analysis/figures').glob('*.png'))
    if len(figures)!=29:
        raise ValueError('Expected 29 trial2 figures')
    result = {'status':'pass','snapshot_files':checked,'python_files_parsed':parsed,'canonical_signature_files':len(metadata['code_sha256']), 'train_pairs':384,'validation_pairs':128,'epochs':[1,2,3,4,5],'run_state':status,'trial2_figures':len(figures),'gpu_used':False,'checkpoint_bytes_rechecked':False}
    if checkpoint is not None:
        result['checkpoint'] = verify_checkpoint(checkpoint)
        result['checkpoint_bytes_rechecked'] = True
    return result

def make_bundle(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root):
        raise ValueError('Bundle destination must be within this project')
    verify(root)
    relative = 'experiments/'+FULL_RUN
    source = root/relative/'source_full_v1'
    paths = [p for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    paths += [root/relative/'run_metadata'/n for n in ('run_metadata.json','run_status.json')]
    manifest = {p.relative_to(root).as_posix():sha256(p) for p in paths}
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        payload = [(p.relative_to(root).as_posix(),p.read_bytes()) for p in paths]
        payload += [('BUNDLE_MANIFEST.json',(json.dumps({'file_sha256':manifest,'checkpoint_included':False,'gpu_execution_authorized_by_bundle':False},indent=2)+'\n').encode())]
        for name,data in payload:
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    data=buffer.getvalue()
    if output.exists() and output.read_bytes()!=data:
        raise FileExistsError('Refusing to replace a different bundle')
    output.parent.mkdir(parents=True,exist_ok=True)
    if not output.exists(): output.write_bytes(data)
    return {'path':str(output),'files':len(manifest),'bytes':len(data),'sha256':sha256(output)}

def import_results(root, incoming, experiment):
    root,incoming=Path(root).resolve(),Path(incoming).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}',experiment):
        raise ValueError('Unsafe experiment identifier')
    allowed={'.json','.jsonl','.csv','.md','.txt','.log','.png','.ipynb','.yaml','.yml'}
    paths=[p for p in sorted(incoming.rglob('*')) if p.is_file()]
    if not paths: raise ValueError('Incoming folder is empty')
    for p in paths:
        if not p.resolve().is_relative_to(incoming) or p.suffix.lower() not in allowed or p.stat().st_size>20_000_000:
            raise ValueError('Unsupported or unsafe incoming file: '+p.name)
    inventory={p.relative_to(incoming).as_posix():sha256(p) for p in paths}
    key=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()[:16]
    target=root/'experiments'/experiment/'imports'/key
    for p in paths:
        dest=target/p.relative_to(incoming)
        if dest.exists() and sha256(dest)!=inventory[p.relative_to(incoming).as_posix()]:
            raise FileExistsError('Refusing to overwrite imported result')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): dest.write_bytes(p.read_bytes())
        assert sha256(dest)==inventory[p.relative_to(incoming).as_posix()]
    marker=target/'IMPORT_MANIFEST.json'
    if not marker.exists(): marker.write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    return {'path':str(target),'files':len(paths),'id':key,'existing_results_overwritten':False}
