"""notebooks/colab_two_stage.ipynb: runs the remaining two-stage steps with one 'Run all' on whatever GPU Colab gives.

The verifier steps: 'verifier' (local stage, mmBERT-small) and 'verifier-qwen...' (server stage, Qwen + LoRA)."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'notebooks/colab_two_stage.ipynb').read_text(encoding='utf-8'))
CELLS = [c for c in NOTEBOOK['cells'] if c['cell_type'] == 'code']
CODE = [''.join(c['source']) for c in CELLS]


def settings():
    values = {}
    exec(CODE[0], values)
    return values


def test_code_cells_parse_and_carry_no_outputs():
    for cell, source in zip(CELLS, CODE):
        ast.parse(source)
        assert cell['outputs'] == [] and cell['execution_count'] is None


def test_every_default_step_has_its_config_and_test_stays_closed():
    s = settings()
    assert s['RUN_TEST'] is False and s['REPO_REF'] == 'main'
    for step in s['STEPS']:
        if step == 'verifier':
            name = 'verifier-creddata-mmbert-small'
        elif step.startswith('verifier-'):
            name = 'verifier-creddata-' + step.removeprefix('verifier-')
        else:
            name = f'token-span-creddata-{step}'
        assert (ROOT / f'configs/{name}.yaml').exists(), step
    # Each server verifier follows the span model of its size, so a GPU that runs one runs the other.
    assert [x for x in s['STEPS'] if x.startswith('verifier-')] == \
        ['verifier-' + x for x in s['STEPS'] if x.startswith('qwen')]
    known = ast.literal_eval(next(line.split(' = ', 1)[1] for line in '\n'.join(CODE).splitlines()
                                  if line.startswith('KNOWN = ')))
    assert known == set(s['STEPS'])


def test_gpu_required_bf16_steps_gated_and_scripts_reused():
    source = '\n'.join(CODE)
    assert "raise RuntimeError('GPU가 없습니다" in source
    assert ("NEEDS = {'qwen7b-bidir': (22, 8), 'qwen14b-bidir': (38, 8), 'verifier-qwen7b-bidir': (22, 8), "
            "'verifier-qwen14b-bidir': (38, 8)}") in source
    assert '"peft>=0.15,<1"' in source  # the verifier's marker rows need PEFT trainable tokens
    assert 'transformers==4.57.6' in source
    for command in ('scripts/colab_verifier.sh prepare', 'scripts/colab_verifier.sh train {m}',
                    'scripts/colab_verifier.sh system {m}', 'scripts/colab_verifier.sh test {m}',
                    'scripts/colab_token_span.sh {step}'):
        assert command in source
    assert all((ROOT / p).exists() for p in ('scripts/colab_verifier.sh', 'scripts/colab_token_span.sh'))
