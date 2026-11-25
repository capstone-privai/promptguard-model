"""Models retrained with NoseyParker-trajectory added: each differs from its CredData-trajectory run only in the data."""
import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

from privai_model.token_span.train import by_source

ROOT = Path(__file__).resolve().parents[2]


def config(name):
    return yaml.safe_load((ROOT / f'configs/{name}.yaml').read_text())


def changed(a, b):
    assert a.keys() == b.keys()
    return {k for k in a if a[k] != b[k]}


@pytest.mark.parametrize('model', ['mmbert', 'mmbert-base', 'qwen-bidir', 'qwen1.5b-bidir'])
def test_np_span_runs_differ_from_creddata_runs_only_in_data_and_output(model):
    old, new = config(f'token-span-creddata-{model}'), config(f'token-span-creddata-np-{model}')
    assert changed(old, new) == {'data_dir', 'output_dir'}
    assert new['data_dir'] == 'runs/data/span-creddata-np-traj-v1'
    assert new['output_dir'] == f'runs/token-span/creddata-np-{model}'


@pytest.mark.parametrize('model', ['mmbert-small', 'mmbert-base'])
def test_np_verifiers_differ_from_creddata_verifiers_only_in_data_output_and_version(model):
    old, new = config(f'verifier-creddata-{model}'), config(f'verifier-creddata-np-{model}')
    assert changed(old, new) == {'data_path', 'split_manifest', 'output_dir', 'model_version'}
    assert new['data_path'].startswith('runs/data/verifier-creddata-np-traj-v1/')
    assert new['split_manifest'].startswith('runs/data/verifier-creddata-np-traj-v1/')


def test_pilot_b_creddata_np_differs_from_b_creddata_only_in_data():
    old, new = config('span-pilot-step0-bcred'), config('span-pilot-step0-bcrednp')
    assert changed(old, new) == {'data_dir', 'output_dir', 'backbone_id'}
    assert new['data_dir'].endswith('-bcrednp')


def prep():
    spec = importlib.util.spec_from_file_location('prep_np', ROOT / 'scripts/prepare_creddata_np_traj.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(group, text, spans):
    return dict(schema_version='secret-span-v1', synthetic=True, id=f'noseyparker_traj:{group}:{text}', text=text,
                spans=spans, value_spans=spans, source='tool_output', language='mixed',
                family_id=f'noseyparker_traj_{group}', template_id=f'noseyparker_traj_{group}',
                origin=f'noseyparker_traj:{group}', value_family=f'noseyparker_traj_{group}',
                kind='positive' if spans else 'hard_negative')


def test_groups_sharing_a_template_with_a_secret_are_one_unit():
    p = prep()
    secret = lambda a, b: [dict(start=a, end=b, secret_type='SECRET')]
    rows = [row('r1', 'key = AAAA1111', secret(6, 14)), row('r2', 'key = BBBB2222', secret(6, 14)),  # same template
            row('r3', 'tok: CCCC', secret(5, 9)),
            row('r4', 'done', []), row('r5', 'done', [])]  # a shared secret-free text does not link groups
    unit = p.linked_groups(rows)
    assert unit['r1'] == unit['r2']
    assert len({unit['r1'], unit['r3'], unit['r4'], unit['r5']}) == 4


def test_np_validation_takes_units_up_to_its_share_and_large_units_stay_in_train():
    p = prep()
    gold = {f'u{i}': n for i, n in enumerate([300] + [10] * 100)}
    split = p.assign(gold, 'seed')
    assert split['u0'] == 'train'
    total = sum(gold.values())
    held = sum(n for u, n in gold.items() if split[u] == 'validation')
    assert p.SHARE * total <= held < p.SHARE * total + 10
    assert set(split.values()) == {'train', 'validation'}


def test_validation_counts_are_also_reported_per_source():
    rows = [dict(id='creddata_traj:a', spans=[dict(start=0, end=4)]), dict(id='noseyparker_traj:b', spans=[dict(start=2, end=6)])]
    predictions = {'creddata_traj:a': [dict(start=0, end=4, score=0.9), dict(start=5, end=8, score=0.95)],
                   'noseyparker_traj:b': [dict(start=2, end=6, score=0.4)]}
    counts = by_source(rows, predictions, 0.5)
    assert {n: (c['tp'], c['fp'], c['fn']) for n, c in counts.items()} == {'creddata_traj': (1, 1, 0),
                                                                            'noseyparker_traj': (0, 0, 1)}
    assert by_source(rows[:1], predictions, 0.5) is None  # one source: nothing extra


@pytest.mark.parametrize('script', ['colab_np_all.sh', 'colab_prepare_creddata_np_traj.sh', 'colab_token_span.sh',
                                    'colab_verifier.sh'])
def test_colab_scripts_parse(script):
    subprocess.run(['bash', '-n', str(ROOT / 'scripts' / script)], check=True)


@pytest.mark.parametrize('script', ['colab_token_span.sh', 'colab_verifier.sh'])
def test_test_scoring_refuses_np_models(script):
    # Both scripts stop before any GPU work on a Mac, so the guard is checked in the source.
    assert "case $m" in (text := (ROOT / 'scripts' / script).read_text()) or 'case $model' in text
    assert "NP models are scored on validation only; the NP test is the team's hold-out" in text
