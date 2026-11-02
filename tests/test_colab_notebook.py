import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'notebooks/colab_train_span.ipynb').read_text())
CODE = [''.join(c['source']) for c in NOTEBOOK['cells'] if c['cell_type'] == 'code']


def settings():
    values = {}
    exec(CODE[0], values)
    return values


def test_code_cells_parse_and_carry_no_outputs():
    for cell, source in zip([c for c in NOTEBOOK['cells'] if c['cell_type'] == 'code'], CODE):
        ast.parse(source)
        assert cell['outputs'] == [] and cell['execution_count'] is None


def test_default_settings_point_at_existing_recipe_and_bundle():
    s = settings()
    assert (ROOT / s['BASE_CONFIG']).exists() and s['OVERRIDES'] == {}
    assert (ROOT / f"models/span-detector-{s['START_FROM']}/model.safetensors").exists()


def test_gpu_required_and_test_outputs_left_out_of_archive():
    source = '\n'.join(CODE)
    assert "raise RuntimeError('GPU가 없습니다" in source and "device='cuda'" in source and "'cpu'" not in source
    keep = next(line for line in source.splitlines() if line.startswith('KEEP = '))
    assert 'evaluation' not in keep and 'test-predictions' not in keep
