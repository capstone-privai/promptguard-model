"""The Qwen step-0 pilot reuses the Qwen recipe: A and B differ only in data, and only length limits change."""
import importlib.util
from argparse import Namespace
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def config(name):
    return yaml.safe_load((ROOT / f'configs/{name}.yaml').read_text())


def script():
    spec = importlib.util.spec_from_file_location('qwen_span_script', ROOT / 'scripts/qwen_span.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_and_b_differ_only_in_data():
    a, b = config('qwen-span-pilot-step0-a'), config('qwen-span-pilot-step0-b')
    assert {k for k in a if a[k] != b[k]} == {'sft_dir', 'output_dir'}


def test_pilot_keeps_the_qwen_recipe_apart_from_length_limits():
    base, pilot = config('qwen-span-coder-0.5b'), config('qwen-span-pilot-step0-b')
    assert set(base) == set(pilot)
    assert {k for k in base if base[k] != pilot[k]} == {'sft_dir', 'output_dir', 'max_length', 'max_new_tokens'}
    assert pilot['max_length'] >= 10158 and pilot['max_new_tokens'] >= 1322  # longest B train row, longest val answer


def test_eval_sets_default_and_override():
    qwen = script()
    assert qwen.eval_sets(Namespace(eval=[])) == qwen.qdata.EVAL_SETS
    path = 'runs/data/span-pilot-step0-b/validation.jsonl'
    assert qwen.eval_sets(Namespace(eval=[f'validation={path}'])) == {'validation': path}
    with pytest.raises(SystemExit):
        qwen.eval_sets(Namespace(eval=['fresh_test=data/span-context-v9/test.jsonl']))
