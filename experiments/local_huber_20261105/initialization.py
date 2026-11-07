"""Strict model-only initialization from the recorded epoch-20 checkpoint."""
import hashlib
from pathlib import Path
from safetensors.torch import load_file

EPOCH20_MODEL_SHA256='52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb'
EPOCH20_CONFIG_SHA256='3c6ab05881e6a76edbfd33ff393fdfb082582084e9262826bb1ac0fd2b1dac7c'

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def load_model_only(model, checkpoint_dir, expected_model_hash=EPOCH20_MODEL_SHA256,
                    expected_config_hash=EPOCH20_CONFIG_SHA256):
    checkpoint_dir=Path(checkpoint_dir)
    for name, expected in [('model.safetensors',expected_model_hash),('config.json',expected_config_hash)]:
        if sha256(checkpoint_dir/name)!=expected:
            raise ValueError(f'checkpoint hash mismatch: {name}')
    state=load_file(str(checkpoint_dir/'model.safetensors'),device='cpu')
    model.load_state_dict(state,strict=True)
    return {'model_sha256':expected_model_hash,'config_sha256':expected_config_hash,
            'trainer_state_loaded':False}
