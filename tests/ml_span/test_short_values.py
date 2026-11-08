"""The runtime's opt-in length guard matches PromptGuard's ml-v9 adapter check."""
import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from privai_model.ml_span.bio import LABELS
from privai_model.ml_span.function_words import POLICY
from privai_model.ml_span.runtime import MLSpanDetector
from privai_model.ml_span.short_values import PROMPTGUARD_MIN_VALUE_LENGTH, filter_short_values


def finding(text, value):
    a = text.index(value)
    return dict(start=a, end=a + len(value), score=1., detector='ml', secret_type='SECRET')


@pytest.mark.parametrize('text,value', [
    ('pin 42 here', '42'), ('x', 'x'), ('code: "ab",', '"ab",'), ('id (abc) ok', '(abc)'), ('key=abcde', 'abcde'),
    ('value  ab  end', '  ab  '),
])
def test_short_values_dropped(text, value):
    assert filter_short_values(text, [finding(text, value)], PROMPTGUARD_MIN_VALUE_LENGTH) == []


@pytest.mark.parametrize('text,value', [
    ('key=abcdef', 'abcdef'), ('token "s3cr3t!"', '"s3cr3t!"'), ('pw: 12-34-5', '12-34-5'), ('ab cd ef', 'ab cd ef'),
])
def test_values_at_or_over_the_limit_kept(text, value):
    span = finding(text, value)
    assert filter_short_values(text, [span], PROMPTGUARD_MIN_VALUE_LENGTH) == [span]


def test_limit_is_the_promptguard_adapter_default():
    assert PROMPTGUARD_MIN_VALUE_LENGTH == 6


def scanning_detector(spans, min_value_length):
    detector = MLSpanDetector.__new__(MLSpanDetector)
    detector.threshold, detector.device, detector.tokenizer = .5, 'cpu', None
    detector.model = lambda **kw: SimpleNamespace(logits=torch.tensor([[[0., 20., 0.]]]))
    detector.max_length, detector.stride = 512, 128
    detector.candidate_filter, detector.min_value_length, detector.delimiter_policy = POLICY, min_value_length, None
    detector.boundary_refiner = SimpleNamespace(refine=lambda text, _: spans)
    return detector


def test_runtime_applies_guard_after_function_words():
    text = 'Written by ab: synthetic_value'
    by, short, long = finding(text, 'by'), finding(text, 'ab'), finding(text, 'synthetic_value')
    windows = [dict(inputs={'input_ids': [1]}, offsets=[(0, len(text))])]
    with patch('privai_model.ml_span.runtime.windows', return_value=windows):
        assert scanning_detector([by, short, long], 6).scan_text(text) == [long]
        assert scanning_detector([by, short, long], None).scan_text(text) == [short, long]


class FakeModel:
    config = SimpleNamespace(id2label=LABELS)

    def to(self, device):
        return self

    def eval(self):
        return self


def load(tmp_path, bundle_value, **kwargs):
    meta = dict(schema_version='ml-span-bundle-v1', max_length=512, stride=128, threshold=.5)
    if bundle_value is not None:
        meta['min_value_length'] = bundle_value
    (tmp_path / 'span_bundle.json').write_text(json.dumps(meta))
    with patch('privai_model.ml_span.runtime.AutoTokenizer.from_pretrained', return_value=SimpleNamespace(is_fast=True)), \
         patch('privai_model.ml_span.runtime.AutoModelForTokenClassification.from_pretrained', return_value=FakeModel()), \
         patch('privai_model.ml_span.runtime.windows', return_value=[]):
        return MLSpanDetector(tmp_path, **kwargs)


def test_bundle_value_is_read_and_caller_value_wins(tmp_path):
    assert load(tmp_path, None).min_value_length is None
    assert load(tmp_path, 6).min_value_length == 6
    assert load(tmp_path, None, min_value_length=6).min_value_length == 6
    assert load(tmp_path, 6, min_value_length=3).min_value_length == 3


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '6'])
def test_invalid_limit_rejected(tmp_path, value):
    with pytest.raises(RuntimeError, match='initialization failed'):
        load(tmp_path, None, min_value_length=value)
    with pytest.raises(RuntimeError, match='initialization failed'):
        load(tmp_path, value)
