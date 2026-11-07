import copy
import tempfile
import unittest
from pathlib import Path
import torch
from epoch_engine import run_epochs

class EpochEngineTests(unittest.TestCase):
    def test_resume_and_config_guard(self):
        dataset=[torch.tensor([[float(i)]]) for i in range(1,6)]
        def step(model,x,epoch): return ((model(x)-2*x)**2).mean()
        def evaluate(model,epoch): return {'weight':float(model.weight.detach()[0,0])}
        cfg={'seed':42,'epochs':3,'accumulation':4,'lambda':.05}
        initial=torch.nn.Linear(1,1); initial_state=copy.deepcopy(initial.state_dict())
        def setup():
            model=torch.nn.Linear(1,1); model.load_state_dict(initial_state)
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            sched=torch.optim.lr_scheduler.PolynomialLR(opt,total_iters=6,power=.9)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            return model,opt,sched,scaler
        with tempfile.TemporaryDirectory() as tmp:
            a,ao,asched,asc=setup(); b,bo,bsched,bsc=setup()
            run_epochs(a,dataset,step,evaluate,ao,asched,asc,Path(tmp)/'a',cfg)
            run_epochs(b,dataset,step,evaluate,bo,bsched,bsc,Path(tmp)/'b',cfg,stop_after=1)
            state=run_epochs(b,dataset,step,evaluate,bo,bsched,bsc,Path(tmp)/'b',cfg,resume=True)
            for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y,rtol=0,atol=0)
            self.assertEqual(state['global_step'],6)
            self.assertEqual(asched.state_dict(),bsched.state_dict())
            self.assertEqual(ao.state_dict()['param_groups'],bo.state_dict()['param_groups'])
            with self.assertRaises(ValueError): run_epochs(b,dataset,step,evaluate,bo,bsched,bsc,Path(tmp)/'b',{**cfg,'lambda':.2},resume=True)

if __name__=='__main__': unittest.main()

class AmpSkipTests(unittest.TestCase):
    def test_scaled_inf_skips_optimizer_and_scheduler(self):
        model=torch.nn.Linear(1,1,bias=False)
        opt=torch.optim.SGD(model.parameters(),lr=.1)
        scheduler=torch.optim.lr_scheduler.PolynomialLR(opt,total_iters=3,power=.9)
        scaler=torch.amp.GradScaler('cpu',init_scale=8.)
        before=model.weight.detach().clone()
        scaler.scale(model(torch.ones(1,1)).sum()).backward()
        model.weight.grad.fill_(float('inf'))
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=not scaler.is_enabled())
        previous=scaler.get_scale(); scaler.step(opt); scaler.update()
        updated=scaler.get_scale()>=previous
        if updated: scheduler.step()
        self.assertFalse(updated); self.assertEqual(scheduler.last_epoch,0)
        torch.testing.assert_close(model.weight,before)
