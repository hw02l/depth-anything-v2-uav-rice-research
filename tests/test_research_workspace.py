"""Offline migration checks and tiny CPU checkpoint-continuation tests; no model downloads."""
from pathlib import Path
import importlib.util
import importlib
import json
import random
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from uav_metric_research.provenance import verify, make_bundle, import_results

class ExchangeTests(unittest.TestCase):
    def test_frozen_sources_and_canonical_run_signature(self):
        self.assertEqual(verify(ROOT)['snapshot_files'],81)

    def test_bundle_preserves_snapshot_bytes(self):
        import hashlib,zipfile
        scratch=ROOT/'work/test_local_exchange'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            p=Path(directory)/'bundle.zip'
            a=make_bundle(ROOT,p)
            self.assertEqual(a,make_bundle(ROOT,p))
            with zipfile.ZipFile(p) as z:
                self.assertIsNone(z.testzip())
                hashes=json.loads(z.read('BUNDLE_MANIFEST.json'))['file_sha256']
                for name,digest in hashes.items():
                    self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),digest)

    def test_import_is_idempotent_and_does_not_replace_old_results(self):
        scratch=ROOT/'work/test_local_exchange'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            base=Path(directory); incoming=base/'incoming'; incoming.mkdir()
            (incoming/'metrics.csv').write_text('epoch,rmse\n5,1.0\n',encoding='utf-8')
            a=import_results(base,incoming,'toy')
            self.assertEqual(a,import_results(base,incoming,'toy'))
            old=Path(a['path'])/'metrics.csv'
            (incoming/'metrics.csv').write_text('epoch,rmse\n6,0.9\n',encoding='utf-8')
            b=import_results(base,incoming,'toy')
            self.assertNotEqual(a['id'],b['id'])
            self.assertIn('5,1.0',old.read_text())
            with self.assertRaises(ValueError): import_results(base,incoming,'../escape')
            (incoming/'weights.pt').write_bytes(b'not-a-real-checkpoint')
            with self.assertRaises(ValueError): import_results(base,incoming,'toy')

class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65/source_full_v1'
        spec=importlib.util.spec_from_file_location('_frozen_trial3',path/'__init__.py',submodule_search_locations=[str(path)])
        module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
        cls.train=importlib.import_module('_frozen_trial3.train')

    def test_signature_changes_are_rejected(self):
        self.train.validate_resume_signature({'epochs':20},{'epochs':20})
        with self.assertRaises(ValueError): self.train.validate_resume_signature({'epochs':20},{'epochs':5})

    def test_cpu_continuation_restores_model_optimizer_scheduler_scaler_rng(self):
        import numpy as np
        import torch
        from safetensors.torch import save_file
        class Tiny(torch.nn.Linear):
            def save_pretrained(self,path,safe_serialization=True):
                save_file(self.state_dict(),str(Path(path)/'model.safetensors'))
                (Path(path)/'config.json').write_text('{"toy":true}',encoding='utf-8')
        def setup():
            model=Tiny(2,1)
            opt=torch.optim.AdamW(model.parameters(),lr=.01)
            sch=torch.optim.lr_scheduler.PolynomialLR(opt,total_iters=1920,power=.9)
            scaler=torch.amp.GradScaler('cpu',init_scale=128.)
            return model,opt,sch,scaler
        def step(model,opt,sch,scaler):
            opt.zero_grad(set_to_none=True)
            x=torch.randn(3,2)+random.random()+float(np.random.random())
            loss=model(x).square().mean()
            scaler.scale(loss).backward()
            self.assertTrue(self.train.scaled_optimizer_step(opt,sch,scaler))
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        model,opt,sch,scaler=setup()
        for _ in range(5): step(model,opt,sch,scaler)
        scratch=ROOT/'work/test_local_resume'; scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            cp=self.train.save_checkpoint(Path(directory),5,model,opt,sch,scaler,1.,5)
            step(model,opt,sch,scaler)
            weights={k:v.clone() for k,v in model.state_dict().items()}
            expected_rng=(random.random(),float(np.random.random()),torch.rand(3))
            expected_schedule=sch.state_dict(); expected_scaler=scaler.state_dict()
            m2,o2,s2,c2=setup()
            state=self.train.restore_checkpoint(cp,m2,o2,s2,c2,torch.device('cpu'))
            self.assertEqual(state['completed_epoch'],5)
            self.assertEqual(s2.total_iters,1920)
            step(m2,o2,s2,c2)
            for k,v in m2.state_dict().items(): self.assertTrue(torch.equal(v,weights[k]))
            self.assertEqual(s2.state_dict(),expected_schedule)
            self.assertEqual(c2.state_dict(),expected_scaler)
            self.assertEqual(random.random(),expected_rng[0])
            self.assertEqual(float(np.random.random()),expected_rng[1])
            self.assertTrue(torch.equal(torch.rand(3),expected_rng[2]))
            (cp/'config.json').write_text('{"tampered":true}',encoding='utf-8')
            with self.assertRaises(ValueError): self.train.restore_checkpoint(cp,m2,o2,s2,c2,torch.device('cpu'))

    def test_amp_skip_does_not_advance_scheduler(self):
        import torch
        p=torch.nn.Parameter(torch.ones(1))
        opt=torch.optim.AdamW([p],lr=.01)
        sch=torch.optim.lr_scheduler.PolynomialLR(opt,total_iters=1920,power=.9)
        class SkippedScaler:
            scale=128.
            def get_scale(self): return self.scale
            def step(self,optimizer): pass
            def update(self): self.scale/=2
        before=sch.state_dict()
        self.assertFalse(self.train.scaled_optimizer_step(opt,sch,SkippedScaler()))
        self.assertEqual(before,sch.state_dict())

if __name__=='__main__': unittest.main()
