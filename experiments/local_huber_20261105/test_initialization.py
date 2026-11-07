import json
import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from initialization import load_model_only,sha256

class InitializationTests(unittest.TestCase):
    def test_strict_model_only_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); a=torch.nn.Linear(3,2); b=torch.nn.Linear(3,2)
            save_file(a.state_dict(),str(p/'model.safetensors')); (p/'config.json').write_text('{}')
            m=sha256(p/'model.safetensors'); c=sha256(p/'config.json')
            result=load_model_only(b,p,m,c)
            for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y)
            self.assertFalse(result['trainer_state_loaded'])
            with self.assertRaises(ValueError): load_model_only(b,p,'0'*64,c)
            with self.assertRaises(RuntimeError): load_model_only(torch.nn.Linear(4,2),p,m,c)
