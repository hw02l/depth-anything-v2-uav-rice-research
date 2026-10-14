"""Offline safety checks using small artificial files and mocked HTTP responses."""
import ast
import contextlib
import copy
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import requests
import tifffile
import torch
from safetensors.torch import save_file

from uav_metric_finetuning import analyze_gt, audit_data, fetch_selected_data, runtime_state, train, memory_preflight

CODE = Path(runtime_state.__file__).parent
SCRATCH = CODE.parent / 'work' / 'uav_metric_finetuning' / 'final_tests'
SCRATCH.mkdir(parents=True, exist_ok=True)
NOTEBOOK = json.loads((CODE / 'colab_train.ipynb').read_text(encoding='utf-8'))


def quiet_main(fn, argv):
    with patch.object(sys, 'argv', ['test'] + argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='uav-final-check-', dir=SCRATCH)
        assert Path(self.temp.name).resolve().is_relative_to(SCRATCH.resolve())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = self.root/'plan'
        self.plan.mkdir()
        self.rgb = self.root/'rgb'
        self.depth = self.root/'depth'
        self.work = self.root/'work'
        self.work.mkdir()
        self.existing = self.root/'results/uav3dcrop/all_selected_images.csv'
        self.existing.parent.mkdir(parents=True)
        self.existing.write_text('scene\n2025/Day001_Corn\n',encoding='utf-8')
        (self.plan/'plan_summary.json').write_text(json.dumps({'scene_metadata':[]}),encoding='utf-8')
        for split, day in (('train','Day019'),('validation','Day047'),('independent_test','Day052')):
            scene = f'2025/{day}_Corn'
            row = dict(split=split,scene=scene,day=f'2025/{day}',crop='corn',view='nadir',
                       image_relpath=f'{scene}/images/1.JPG',depth_relpath=f'{scene}_depth/1_depth.tif')
            a,b=self.rgb/row['image_relpath'],self.depth/row['depth_relpath']
            a.parent.mkdir(parents=True); b.parent.mkdir(parents=True)
            cv2.imwrite(str(a),np.full((14,14,3),128,np.uint8))
            gt=np.linspace(1,10,196,dtype=np.float32).reshape(14,14)
            gt[0,0]=0; gt[0,1]=np.nan; gt[0,2]=-1
            tifffile.imwrite(b,gt)
            row.update(image_size_bytes=str(a.stat().st_size),depth_size_bytes=str(b.stat().st_size))
            with (self.plan/f'{split}.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
        self.audit_args=dict(manifests={s:self.plan/f'{s}.csv' for s in ('train','validation','independent_test')},
                             rgb_root=self.rgb,depth_root=self.depth,physical_splits=['train','validation'],kind='audit',
                             extra_files={'plan_summary':self.plan/'plan_summary.json','existing_eval':self.existing})
        self.gt_args=dict(manifests={'train':self.plan/'train.csv'},rgb_root=None,depth_root=self.depth,
                          physical_splits=['train'],kind='gt',parameters={'seed':42,'max_pixels_per_frame':20000})

    def audit(self):
        out=self.work/'data_audit_test.json'
        quiet_main(audit_data.main,['--train',str(self.plan/'train.csv'),'--validation',str(self.plan/'validation.csv'),
            '--independent-test',str(self.plan/'independent_test.csv'),'--physical-splits','train','validation',
            '--rgb-root',str(self.rgb),'--depth-root',str(self.depth),'--existing-eval',str(self.existing),
            '--plan-summary',str(self.plan/'plan_summary.json'),'--output',str(out)])
        return json.loads(out.read_text(encoding='utf-8'))

    def gt(self):
        out=self.work/'gt_distribution_train_test.json'
        quiet_main(analyze_gt.main,['--manifest',str(self.plan/'train.csv'),'--depth-root',str(self.depth),
                                   '--output',str(out)])
        return json.loads(out.read_text(encoding='utf-8'))

    def test_audit_reuse_success_and_invalidations(self):
        report=self.audit()
        self.assertTrue(runtime_state.reuse_report(report,**self.audit_args)[0])
        self.assertEqual(report['pixel_audit']['invalid_nonfinite_pixels'],2)
        cases=[]
        wrong=copy.deepcopy(report); wrong['reuse_context']['code_sha256']['audit_data.py']='changed'; cases.append((wrong,self.audit_args))
        cases.append((report,self.audit_args|{'rgb_root':self.root/'other'}))
        wrong=copy.deepcopy(report); wrong.pop('reuse_context'); cases.append((wrong,self.audit_args))
        for item,args in cases:
            self.assertFalse(runtime_state.reuse_report(item,**args)[0])
        path=self.rgb/'2025/Day019_Corn/images/1.JPG'
        old=path.stat()
        os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns+1000000000))
        self.assertFalse(runtime_state.reuse_report(report,**self.audit_args)[0])
        path.write_bytes(b'bad')
        self.assertFalse(runtime_state.reuse_report(report,**self.audit_args)[0])
        path.unlink()
        self.assertFalse(runtime_state.reuse_report(report,**self.audit_args)[0])

    def test_manifest_and_gt_conditions_invalidate(self):
        report=self.gt()
        self.assertTrue(runtime_state.reuse_report(report,**self.gt_args)[0])
        self.assertFalse(runtime_state.reuse_report(report,**(self.gt_args|{'parameters':{'seed':43,'max_pixels_per_frame':20000}}))[0])
        with (self.plan/'train.csv').open('a',encoding='utf-8') as f: f.write('\n')
        self.assertFalse(runtime_state.reuse_report(report,**self.gt_args)[0])

    def test_missing_data_skip_and_scene_leakage(self):
        (self.depth/'2025/Day019_Corn_depth/1_depth.tif').unlink()
        report=self.audit()
        self.assertEqual(report['status'],'skipped_missing_files')
        self.assertEqual(report['local_file_presence']['train']['missing_depth_files'],1)
        rows=train.read_manifest(self.plan/'train.csv')
        val=[dict(rows[0],split='validation')]
        with self.assertRaises(ValueError): train.check_split_isolation(rows,val)
        protected=[dict(rows[0],scene='2025/Day001_Corn',day='2025/Day001')]
        with self.assertRaises(ValueError): train.check_split_isolation(protected,train.read_manifest(self.plan/'validation.csv'))
        with self.assertRaises(SystemExit):
            quiet_main(analyze_gt.main,['--manifest',str(self.plan/'validation.csv'),'--depth-root',str(self.depth),'--output',str(self.work/'invalid.json')])

    def test_runtime_reinitialization_reuses_without_pixels(self):
        self.audit(); self.gt()
        runtime_state.save_settings(self.work/'session_settings.json',{'MAX_DEPTH_M':15.0})
        source=NOTEBOOK['cells'][1]['source']
        bootstrap=source[source.index('# ランタイム変更・セッション終了後'):]
        namespace=dict(Path=Path,json=json,WORK_DIR=self.work,PLAN_DIR=self.plan,
                       DRIVE_RGB_ROOT=self.rgb,DRIVE_DEPTH_ROOT=self.depth,PROJECT_DIR=self.root)
        with patch.object(cv2,'imread',side_effect=AssertionError('pixel reread')),patch.object(tifffile,'imread',side_effect=AssertionError('GT reread')),contextlib.redirect_stdout(io.StringIO()):
            with patch.object(sys,'path',[str(CODE)]+sys.path):
                exec(bootstrap,namespace)
        self.assertTrue(namespace['SESSION_READY']); self.assertTrue(namespace['AUDIT_OK'])
        self.assertIsNotNone(namespace['GT_REPORT']); self.assertEqual(namespace['MAX_DEPTH_M'],15.0)
        for name in ('RUN_DOWNLOAD','ALLOW_MODEL_DOWNLOAD','RUN_GPU_PREFLIGHT','RUN_TRAINING','RUN_INDEPENDENT_TEST'):
            self.assertFalse(namespace[name])

    def test_notebook_gates_and_missing_initialization(self):
        for i,cell in enumerate(NOTEBOOK['cells']):
            if cell['cell_type']!='code': continue
            ast.parse(cell['source'])
            if i!=1:
                with self.assertRaisesRegex(RuntimeError,'初期化|最初|Drive'):
                    exec(cell['source'],{})
        training=next(c['source'] for c in NOTEBOOK['cells'] if 'RUN_TRAINING = False' in c.get('source','') and c['cell_type']=='code' and c is not NOTEBOOK['cells'][1])
        with patch.object(torch.cuda,'is_available',side_effect=AssertionError('GPU accessed')),contextlib.redirect_stdout(io.StringIO()):
            exec(training,{'SESSION_READY':True})
        with patch.object(requests.Session,'get',side_effect=AssertionError('network accessed')):
            with self.assertRaises(SystemExit): quiet_main(fetch_selected_data.main,[])
        download=next(c['source'] for c in NOTEBOOK['cells'] if 'SPLITS_TO_DOWNLOAD =' in c.get('source','') and c['cell_type']=='code')
        with patch.object(subprocess,'run',side_effect=AssertionError('transfer launched')),contextlib.redirect_stdout(io.StringIO()):
            exec(download,dict(SESSION_READY=True,dataset_status=runtime_state.dataset_status,
                               PLAN_DIR=self.plan,DRIVE_RGB_ROOT=self.rgb,DRIVE_DEPTH_ROOT=self.depth))

    def test_hub_empty_cache_is_offline(self):
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        with patch.object(requests.Session,'request',side_effect=AssertionError('network accessed')) as request:
            with self.assertRaises(LocalEntryNotFoundError):
                snapshot_download('depth-anything/Depth-Anything-V2-Small-hf',revision='5426e4f0f36572d16453bbda7a8389317b1bef99',cache_dir=self.root/'empty_cache',local_files_only=True)
            request.assert_not_called()

    def test_model_download_forbidden(self):
        source=next(c['source'] for c in NOTEBOOK['cells'] if 'from huggingface_hub import snapshot_download' in c.get('source',''))
        calls=[]
        def fake(**kwargs):
            calls.append(kwargs)
            self.assertTrue(kwargs['local_files_only'])
            raise FileNotFoundError('offline empty cache')
        with patch('huggingface_hub.snapshot_download',side_effect=fake),patch.object(sys,'path',[str(CODE)]+sys.path),contextlib.redirect_stdout(io.StringIO()):
            exec(source,dict(SESSION_READY=True,TRAINING_MODES=['relative_encoder'],MODEL_CACHE_DIR=self.root/'cache'))
        self.assertEqual(len(calls),1)


class Response:
    def __init__(self,data,status=200,headers=None,fail=False):
        self.data,self.status_code,self.headers,self.fail=data,status,headers or {},fail
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def raise_for_status(self): pass
    def iter_content(self,chunk_size):
        yield self.data[:3]
        if self.fail: raise requests.ConnectionError('artificial interruption')
        yield self.data[3:]


class DownloadTests(unittest.TestCase):
    def test_interruption_range_resume_and_completed_skip(self):
        data=b'0123456789'
        with tempfile.TemporaryDirectory(dir=SCRATCH) as td:
            target=Path(td)/'image'
            session=unittest.mock.Mock()
            session.get.return_value=Response(data,fail=True)
            digest=hashlib.sha256(data).hexdigest()
            with self.assertRaises(requests.ConnectionError): fetch_selected_data.stream_one(session,'https://example.invalid/a',target,len(data),digest)
            self.assertFalse(target.exists()); self.assertEqual(target.with_name('image.uavmetric.part').stat().st_size,3)
            session.get.return_value=Response(data[3:],206,{'Content-Range':'bytes 3-9/10'})
            transferred,_=fetch_selected_data.stream_one(session,'https://example.invalid/a',target,len(data),digest)
            self.assertEqual(transferred,7); self.assertEqual(target.read_bytes(),data)
            self.assertEqual(session.get.call_args.kwargs['headers'],{'Range':'bytes=3-'})
            session.get.side_effect=AssertionError('duplicate transfer')
            self.assertEqual(fetch_selected_data.stream_one(session,'unused',target,10)[0],0)
            with self.assertRaises(FileExistsError): fetch_selected_data.stream_one(session,'unused',target,11)

    def test_invalid_range_and_sha_preserve_final_target(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as td:
            target=Path(td)/'x'; session=unittest.mock.Mock()
            session.get.return_value=Response(b'abcdefghij',fail=True)
            with self.assertRaises(requests.ConnectionError): fetch_selected_data.stream_one(session,'https://example.invalid/a',target,10)
            session.get.return_value=Response(b'defghij',206,{'Content-Range':'bytes 4-9/10'})
            with self.assertRaises(IOError): fetch_selected_data.stream_one(session,'https://example.invalid/a',target,10)
            self.assertFalse(target.exists())
            session.get.return_value=Response(b'abcdefghij',200)
            fetch_selected_data.stream_one(session,'https://example.invalid/a',target,10)
            self.assertEqual(target.read_bytes(),b'abcdefghij')
            other=Path(td)/'bad'; session.get.return_value=Response(b'abcd')
            with self.assertRaises(IOError): fetch_selected_data.stream_one(session,'https://example.invalid/b',other,4,'0'*64)
            self.assertFalse(other.exists())


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.layer=torch.nn.Linear(2,1)
    def save_pretrained(self,path,safe_serialization=True):
        path=Path(path); path.mkdir(exist_ok=True)
        save_file(self.state_dict(),str(path/'model.safetensors'))
        (path/'config.json').write_text('{}',encoding='utf-8')


class CheckpointTests(unittest.TestCase):
    def test_amp_skip_does_not_advance_scheduler(self):
        for overflow in (False,True):
            optimizer=unittest.mock.Mock(); scheduler=unittest.mock.Mock(); scaler=unittest.mock.Mock()
            scaler.get_scale.side_effect=[128.,64. if overflow else 128.]
            self.assertEqual(train.scaled_optimizer_step(optimizer,scheduler,scaler),not overflow)
            self.assertEqual(scheduler.step.call_count,0 if overflow else 1)

    def test_preflight_rejects_nonfinite_gradients(self):
        parameter=torch.nn.Parameter(torch.ones(2))
        parameter.grad=torch.ones(2)
        memory_preflight.require_finite_gradients(torch.tensor(1.),[parameter])
        parameter.grad[0]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'数値異常'):
            memory_preflight.require_finite_gradients(torch.tensor(1.),[parameter])

    @staticmethod
    def make():
        m=TinyModel(); o=torch.optim.AdamW(m.parameters(),lr=.01)
        s=torch.optim.lr_scheduler.PolynomialLR(o,total_iters=4,power=.9)
        g=torch.amp.GradScaler('cuda',enabled=False)
        return m,o,s,g
    @staticmethod
    def step(m,o,s):
        o.zero_grad(); m.layer(torch.ones(1,2)).square().mean().backward(); o.step(); s.step()
    def test_restore_optimizer_scheduler_and_next_step(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as td:
            m,o,s,g=self.make(); self.step(m,o,s)
            checkpoint=train.save_checkpoint(Path(td),1,m,o,s,g,1.0,1)
            n,p,t,h=self.make()
            state=train.restore_checkpoint(checkpoint,n,p,t,h,torch.device('cpu'))
            self.assertEqual(state['completed_epoch'],1); self.assertEqual(s.state_dict(),t.state_dict())
            self.assertEqual(o.param_groups[0]['lr'],p.param_groups[0]['lr'])
            self.step(m,o,s); self.step(n,p,t)
            for a,b in zip(m.parameters(),n.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(s.state_dict(),t.state_dict())
            with self.assertRaises(FileExistsError): train.save_checkpoint(Path(td),1,m,o,s,g,1.0,1)
            with self.assertRaises(ValueError): train.validate_resume_signature({'epochs':1},{'epochs':20})
            with (checkpoint/'model.safetensors').open('ab') as f: f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'完全性'): train.restore_checkpoint(checkpoint,n,p,t,h,torch.device('cpu'))

    def test_trial_full_settings(self):
        self.assertEqual(runtime_state.load_settings(Path('nonexistent-final-test-settings.json'))['EPOCHS'],1)
        with self.assertRaises(ValueError): runtime_state.validate_settings({'EPOCHS':20})
        self.assertEqual(runtime_state.validate_settings({'RUN_KIND':'full','EPOCHS':20,'RUN_LABEL':'full_new'})['EPOCHS'],20)


class VerifiedPlanTests(unittest.TestCase):
    def test_old_incorrect_transfer_total_is_blocked(self):
        with patch.object(requests.Session, 'get', side_effect=AssertionError('network accessed')):
            with self.assertRaises(SystemExit):
                quiet_main(fetch_selected_data.main, ['--plan-dir',str(CODE/'config/verified_plan'),
                           '--download','--confirm-total-bytes','7850370776'])

    def test_original_plan_counts_sizes_revisions_and_isolation(self):
        plan_dir=CODE/'config/verified_plan'
        summary=json.loads((plan_dir/'plan_summary.json').read_text(encoding='utf-8'))
        expected_hashes={'train':'13ee3432515250eff50eeb5238dfc178c8853239074eb8cbbea2b510272b0602',
                         'validation':'96c4eab0eb0075ac141ff9b17c22a45734f242f7cc895f8eaf064a4f14cc871e',
                         'independent_test':'9932fd511dcee66ecb6194260e2d595089d9735f477b95ff4d3b12346b384a0b'}
        manifests={}
        for split,count in (('train',384),('validation',128),('independent_test',319)):
            path=plan_dir/f'{split}.csv'
            self.assertEqual(runtime_state.sha256(path),expected_hashes[split])
            rows=train.read_manifest(path); manifests[split]=rows
            self.assertEqual(len(rows),count)
            total=sum(int(r['image_size_bytes'])+int(r['depth_size_bytes']) for r in rows)
            self.assertEqual(total,summary['selected_bytes_by_split'][split]['bytes'])
            for row in rows:
                self.assertEqual(Path(row['depth_relpath']).stem,Path(row['image_relpath']).stem+'_depth')
                self.assertIn('/resolve/'+summary['rgb_revision']+'/',row['image_url'])
                self.assertIn('/resolve/'+summary['depth_revision']+'/',row['depth_url'])
                self.assertFalse(row['day'] in ('2025/Day001','2025/Day007'))
        for a,b in (('train','validation'),('train','independent_test'),('validation','independent_test')):
            for key in ('scene','day'):
                self.assertFalse({r[key] for r in manifests[a]} & {r[key] for r in manifests[b]})
        self.assertEqual(sum(summary['selected_bytes_by_split'][s]['bytes'] for s in ('train','validation')),7850330776)
        train.check_split_isolation(manifests['train'],manifests['validation'])


if __name__=='__main__': unittest.main()
