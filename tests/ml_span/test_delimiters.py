"""Delimiter trims: fix token-merged quotes and separators without ever cutting a value that ends in a quote."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from privai_model.ml_span.delimiters import POLICY, POLICY_V2, trim_quote_delimiters
from privai_model.ml_span.runtime import MLSpanDetector


def span(text, value, at=0):
    a = text.index(value, at)
    return dict(start=a, end=a + len(value), score=1., detector='ml', secret_type='SECRET')


@pytest.mark.parametrize('text,predicted,expected', [
    ('pw="syn-value)"; next', 'syn-value)";', 'syn-value)'),
    ("pw='syn-value.'\nnext", "syn-value.'", 'syn-value.'),
    ('{"k":"c3luLXZhbHVl==","n":1}', 'c3luLXZhbHVl==",', 'c3luLXZhbHVl=='),
    ('{"k":"syn-value"","n":1}', 'syn-value""', 'syn-value"'),       # the value itself ends with a quote
    ('key: "syn\\"value\\""\nnext', 'syn\\"value\\""', 'syn\\"value\\"'),  # escaped quotes stay inside
    ('use `syn-value`.', '`syn-value', 'syn-value'),                  # swallowed opening quote
])
def test_closing_quote_trimmed(text, predicted, expected):
    out = trim_quote_delimiters(text, [span(text, predicted)])
    assert text[out[0]['start']:out[0]['end']] == expected


@pytest.mark.parametrize('text,value', [
    ('pw="syn-value"; next', 'syn-value'),        # already exact
    ('pw="syn"inner"x"', 'syn"inner'),            # a quote followed by more value is part of it
    ('pw = syn-value"; x', 'syn-value";'),        # no opening quote before the span: left alone
    ('{"k":"syn-value"","n":1}', 'syn-value"'),   # exact value that ends with the quote character
])
def test_left_alone(text, value):
    s = span(text, value)
    assert trim_quote_delimiters(text, [s]) == [s]


@pytest.mark.parametrize('text,predicted,expected', [
    ('pw="+syn-value"; next', 'syn-value";', '+syn-value'),          # `"+` was one token: start moves back
    ('{"k":"#syn~value.","n":1}', 'syn~value.",', '#syn~value.'),
    ('use `/syn/value`.', 'syn/value', '/syn/value'),
    ('queue:%9synvalue@host', ':%9synvalue', '%9synvalue'),          # `:%` was one token: separator dropped
    ('pw="#syn-value"; x', '="#syn-value";', '#syn-value'),          # separator and opening quote dropped
    ('pw="syn-value)"; next', 'syn-value)";', 'syn-value)'),         # v1 behaviour kept
])
def test_v2_moves_token_merged_start(text, predicted, expected):
    s = span(text, predicted)
    out = trim_quote_delimiters(text, [s], POLICY_V2)
    assert text[out[0]['start']:out[0]['end']] == expected


@pytest.mark.parametrize('text,value', [
    ("token: '=SynValue'", '=SynValue'),       # a value that starts with = after a quote
    ("a: k='=SynValue'", '=SynValue'),
    ('x="><syn-value"', '<syn-value'),         # > is not value punctuation
    ('pw="+syn-value"; next', '+syn-value'),  # already exact
    ('key=syn-value', 'syn-value'),
])
def test_v2_left_alone(text, value):
    s = span(text, value)
    assert trim_quote_delimiters(text, [s], POLICY_V2) == [s]


def test_unknown_policy_name_rejected():
    with pytest.raises(ValueError):
        trim_quote_delimiters('x', [], 'unknown')


def test_runtime_applies_policy_only_when_set():
    text = 'pw="syn-value)"; next'
    raw = span(text, 'syn-value)";')
    detector = MLSpanDetector.__new__(MLSpanDetector)
    detector.threshold, detector.device, detector.tokenizer = .5, 'cpu', None
    detector.model = lambda **kw: SimpleNamespace(logits=torch.tensor([[[0., 20., 0.]]]))
    detector.max_length, detector.stride = 512, 128
    detector.candidate_filter, detector.min_value_length, detector.delimiter_policy = None, None, POLICY
    detector.boundary_refiner = SimpleNamespace(refine=lambda text, _: [raw])
    windows = [dict(inputs={'input_ids': [1]}, offsets=[(0, len(text))])]
    with patch('privai_model.ml_span.runtime.windows', return_value=windows):
        out = detector.scan_text(text)
        assert text[out[0]['start']:out[0]['end']] == 'syn-value)'
        detector.delimiter_policy = None
        assert detector.scan_text(text) == [raw]


def test_unknown_policy_rejected():
    with pytest.raises(RuntimeError, match='initialization failed'):
        MLSpanDetector(delimiter_policy='unknown')
