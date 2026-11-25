"""scripts/half_bundle.py: only the weights change (floating tensors to bfloat16); a half span bundle is refused."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location('half_bundle', ROOT / 'scripts/half_bundle.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bundle(tmp_path, name, meta):
    src = tmp_path / 'src'
    src.mkdir()
    save_file({'w': torch.randn(4, 3), 'ids': torch.arange(5)}, str(src / 'model.safetensors'))
    (src / name).write_text(json.dumps(meta))
    (src / 'tokenizer.json').write_text('{"t": 1}')
    return src


@pytest.mark.parametrize('name', ['span_bundle.json', 'bundle.json'])
def test_only_floating_weights_become_bfloat16(tmp_path, name):
    hb = module()
    src = bundle(tmp_path, name, {'threshold': 0.5})
    out = tmp_path / 'out'
    hb.halve(src, out)
    before, after = load_file(str(src / 'model.safetensors')), load_file(str(out / 'model.safetensors'))
    assert after['w'].dtype == torch.bfloat16 and torch.equal(after['w'], before['w'].to(torch.bfloat16))
    assert torch.equal(after['ids'], before['ids'])
    assert (out / 'tokenizer.json').read_text() == '{"t": 1}'
    meta = json.loads((out / name).read_text())
    assert meta['threshold'] == 0.5 and meta['half_storage']['dtype'] == 'bfloat16'
    assert 'torch_dtype' not in meta  # loaders keep computing in float32


def test_a_span_bundle_that_loads_in_half_is_refused(tmp_path):
    hb = module()
    src = bundle(tmp_path, 'span_bundle.json', {'threshold': 0.5, 'torch_dtype': 'bfloat16'})
    with pytest.raises(ValueError):
        hb.halve(src, tmp_path / 'out')
