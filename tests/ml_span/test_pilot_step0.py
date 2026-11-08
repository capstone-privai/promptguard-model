"""The step-0 pilot compares model A and model B, so their training settings must match apart from the data."""
from pathlib import Path
import pytest
import yaml
from privai_model.ml_span.splitter import assert_disjoint

ROOT = Path(__file__).resolve().parents[2]
DATA_KEYS = {'data_dir', 'output_dir', 'backbone_id'}


def recipe(model):
    return yaml.safe_load((ROOT / f'configs/span-pilot-step0-{model}.yaml').read_text())


def test_a_and_b_differ_only_in_data():
    a, b = recipe('a'), recipe('b')
    assert a.keys() == b.keys()
    assert {k for k in a if a[k] != b[k]} == DATA_KEYS
    assert a['data_dir'].endswith('-a') and b['data_dir'].endswith('-b')
    assert a['shared_key_kinds'] == ['value']


@pytest.mark.parametrize('variant', ['bhn', 'bv10'])
def test_follow_ups_differ_from_b_only_in_data(variant):
    b, other = recipe('b'), recipe(variant)
    assert b.keys() == other.keys()
    assert {k for k in b if b[k] != other[k]} == DATA_KEYS
    assert other['data_dir'].endswith(f'-{variant}')


def test_pilot_uses_the_v10_base_recipe():
    base = yaml.safe_load((ROOT / 'configs/span-mmbert-context-v10-base.yaml').read_text())
    a = recipe('a')
    assert {k: a[k] for k in base if k not in DATA_KEYS} == {k: base[k] for k in base if k not in DATA_KEYS}


def row(rid, value, family):
    text = f'token {value} end'
    span = dict(start=6, end=6 + len(value), secret_type='SECRET')
    return dict(schema_version='secret-span-v1', synthetic=True, id=rid, text=text, spans=[span], value_spans=[span],
                source='prompt', language='en', family_id=family, template_id=family, origin=family,
                value_family=family, kind='positive')


def test_shared_values_allowed_only_when_named():
    splits = dict(train=[row('a', 'common-pass-1', 'f1')], validation=[row('b', 'common-pass-1', 'f2')])
    splits['validation'][0]['text'] = 'other common-pass-1 end'
    splits['validation'][0]['spans'] = splits['validation'][0]['value_spans'] = [dict(start=6, end=19, secret_type='SECRET')]
    with pytest.raises(ValueError, match='leakage'):
        assert_disjoint(splits)
    assert_disjoint(splits, ('value',))
    splits['validation'][0]['family_id'] = 'f1'
    with pytest.raises(ValueError, match='leakage'):
        assert_disjoint(splits, ('value',))
