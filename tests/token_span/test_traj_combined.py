"""Models retrained on Trajectory-train-combined v1 (the third training set): data split and run settings."""
import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def config(name):
    return yaml.safe_load((ROOT / f'configs/{name}.yaml').read_text())


def changed(a, b):
    assert a.keys() == b.keys()
    return {k for k in a if a[k] != b[k]}


def test_span_run_differs_from_the_creddata_run_only_in_data_and_output():
    old, new = config('token-span-creddata-mmbert-base'), config('token-span-traj-combined-mmbert-base')
    assert changed(old, new) == {'data_dir', 'output_dir'}
    assert new['data_dir'] == 'runs/data/span-traj-combined-v1'


def test_verifier_differs_from_the_creddata_verifier_only_in_data_output_and_version():
    old, new = config('verifier-creddata-mmbert-small'), config('verifier-traj-combined-mmbert-small')
    assert changed(old, new) == {'data_path', 'split_manifest', 'output_dir', 'model_version'}
    assert new['data_path'].startswith('runs/data/verifier-traj-combined-v1/')
    assert new['split_manifest'].startswith('runs/data/verifier-traj-combined-v1/')


def prep():
    spec = importlib.util.spec_from_file_location('prep_tc', ROOT / 'scripts/prepare_traj_combined.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(prefix, group, text, spans):
    return dict(schema_version='secret-span-v1', synthetic=True, id=f'{prefix}:{group}:{text}', text=text, spans=spans,
                value_spans=spans, source='tool_output', language='mixed', family_id=f'{prefix}_{group}',
                template_id=f'{prefix}_{group}', origin=f'{prefix}:{group}', value_family=f'{prefix}_{group}',
                kind='positive' if spans else 'hard_negative')


secret = lambda a, b: [dict(start=a, end=b, secret_type='SECRET')]


def test_earlier_groups_keep_their_split_and_linked_infra_groups_follow_them():
    p = prep()
    rows = [row('creddata_traj', 'c1', 'key = AAAA1111', secret(6, 14)),
            row('noseyparker_traj', 'n1', 'tok: BBBB', secret(5, 9)),
            row('infra_config_traj', 'i1', 'key = CCCC2222', secret(6, 14)),  # same template as c1
            row('infra_config_traj', 'i2', 'pw: DDDD', secret(4, 8)),
            row('infra_config_traj', 'i3', 'done', [])]
    earlier = {'creddata_traj_c1': 'test', 'noseyparker_traj_n1': 'validation'}
    split, unit, fixed = p.split_families(rows, earlier, 'seed')
    assert split['creddata_traj_c1'] == 'test' and split['noseyparker_traj_n1'] == 'validation'
    assert split['infra_config_traj_i1'] == 'test'  # takes the split of the CredData group it shares a template with
    assert split['infra_config_traj_i2'] in ('train', 'validation') and split['infra_config_traj_i3'] in ('train', 'validation')


def test_a_unit_across_two_earlier_splits_or_an_unknown_earlier_source_group_stops_the_build():
    p = prep()
    rows = [row('creddata_traj', 'c1', 'key = AAAA1111', secret(6, 14)), row('creddata_traj', 'c2', 'key = BBBB2222', secret(6, 14))]
    with pytest.raises(ValueError, match='two earlier splits'):
        p.split_families(rows, {'creddata_traj_c1': 'train', 'creddata_traj_c2': 'validation'}, 'seed')
    with pytest.raises(ValueError, match='not in the earlier split'):
        p.split_families(rows, {'creddata_traj_c1': 'train'}, 'seed')


def test_infra_validation_takes_units_up_to_its_share_and_large_units_stay_in_train():
    p = prep()
    gold = {f'u{i}': n for i, n in enumerate([100] + [10] * 300)}
    split = p.assign(gold, 'seed')
    assert split['u0'] == 'train'
    total = sum(gold.values())
    held = sum(n for u, n in gold.items() if split[u] == 'validation')
    assert p.SHARE * total <= held < p.SHARE * total + 10
    assert set(split.values()) == {'train', 'validation'}


@pytest.mark.parametrize('script', ['colab_tc_all.sh', 'colab_prepare_traj_combined.sh', 'colab_token_span.sh',
                                    'colab_verifier.sh'])
def test_colab_scripts_parse(script):
    subprocess.run(['bash', '-n', str(ROOT / 'scripts' / script)], check=True)


@pytest.mark.parametrize('script', ['colab_token_span.sh', 'colab_verifier.sh'])
def test_test_scoring_refuses_tc_models(script):
    text = (ROOT / 'scripts' / script).read_text()
    assert 'np-*|tc-*) echo' in text


def test_tc_names_map_to_the_new_configs():
    out = subprocess.run(['bash', '-c', 'eval "$(sed -n "/^run_name()/,/^config_of()/p" scripts/colab_token_span.sh)"; '
                          'config_of tc-mmbert-base; config_of np-mmbert-base'], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ['configs/token-span-traj-combined-mmbert-base.yaml', 'configs/token-span-creddata-np-mmbert-base.yaml']
    assert all((ROOT / c).exists() for c in out)
