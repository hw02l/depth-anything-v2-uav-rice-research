import unittest
from types import SimpleNamespace
import torch
from train_step import step_batch
from test_objective import frozen

class ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.conv=torch.nn.Conv2d(3,1,1)
    def forward(self,pixel_values):
        return SimpleNamespace(predicted_depth=torch.nn.functional.softplus(self.conv(pixel_values))[:,0]+1)

class TrainingStepTests(unittest.TestCase):
    def test_one_optimizer_step(self):
        torch.manual_seed(42); model=ToyNet(); opt=torch.optim.AdamW(model.parameters(),lr=.001)
        gt=torch.linspace(1,3,80*80).reshape(80,80)
        batch={'pixel_values':torch.randn(3,80,80),'depth':gt,'original_depth':gt,
          'geometry':{'original_hw':(80,80),'resized_hw':(80,80),'crop_yx':(0,0),'crop_hw':(80,80),'flipped':False},'image_relpath':'toy'}
        before=model.conv.weight.detach().clone()
        total,s,l,c=step_batch(model,batch,frozen.silog_loss,'cpu',1,.05)
        total.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        opt.step(); self.assertFalse(torch.equal(before,model.conv.weight))
        self.assertTrue(all(v>0 for v in c.values()))

if __name__=='__main__': unittest.main()
