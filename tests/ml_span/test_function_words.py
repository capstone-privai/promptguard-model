"""Regression checks for grammatical-word FP suppression and short literal retention."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from privai_model.ml_span.function_words import POLICY, filter_function_words
from privai_model.ml_span.runtime import MLSpanDetector, candidate_union


def finding(text, word):
    a = text.index(word)
    return dict(start=a, end=a+len(word), score=1., detector='ml', secret_type='SECRET')


@pytest.mark.parametrize('text,word', [
    ('Written by the author.', 'by'), ('The name of the service.', 'of'),
    ('Read and write files.', 'and'), ('Enable this or that.', 'or'),
    ('Small but useful.', 'but'), ('By the author.', 'By'),
    ('if enabled then run', 'if'), ('by', 'by'), ('of', 'of'),
])
def test_prose_false_positive_removed(text, word):
    assert filter_function_words(text, [finding(text, word)]) == []


@pytest.mark.parametrize('text,word', [
    ('password="by"', 'by'), ("secret: 'of'", 'of'), ('token=and', 'and'),
    ('password is of', 'of'), ('password is the word by', 'by'),
    ('비밀번호는 by', 'by'), ('토큰 값은 of', 'of'), ('The password is exactly by', 'by'),
    ('Authorization: Bearer or', 'or'), ('of is the actual password', 'of'),
    ('by는 비밀번호입니다.', 'by'), ('암호: of', 'of'), ('password:\n  by', 'by'),
    ('credential is `and`', 'and'), ('password = “of”', 'of'),
    ('abcbydef', 'by'), ('synthetic_of_123', 'of'), ('by-token', 'by'),
    ('of.txt', 'of'), ('by\u0301', 'by'), ('https://of@example.invalid', 'of'),
    ('password = by-secret-123', 'by-secret-123'), ('password = of-and', 'of-and'),
    ('plain synthetic_secret_value', 'synthetic_secret_value'),
])
def test_explicit_values_and_fragments_preserved(text, word):
    span = finding(text, word)
    assert filter_function_words(text, [span]) == [span]


def test_only_ml_candidate_removed_and_offsets_unchanged():
    text = 'Written by client: synthetic_value'
    bad, good = finding(text, 'by'), finding(text, 'synthetic_value')
    assert filter_function_words(text, [bad, good]) == [good]
    rule = dict(bad, detector='rule')
    union = candidate_union(text, [rule], filter_function_words(text, [bad, good]), .5)
    assert [(s['start'], s['end']) for s in union] == [(bad['start'], bad['end']), (good['start'], good['end'])]


def test_runtime_uses_filter_after_boundary_refinement():
    detector = MLSpanDetector.__new__(MLSpanDetector)
    detector.threshold = .5
    detector.device = 'cpu'
    detector.model = lambda **kw: SimpleNamespace(logits=torch.tensor([[[0., 20., 0.]]]))
    detector.tokenizer = None
    detector.max_length, detector.stride = 512, 128
    detector.candidate_filter = POLICY
    text = 'Written by the author.'
    span = finding(text, 'by')
    detector.boundary_refiner = SimpleNamespace(refine=lambda text, spans: [span])
    windows = [dict(inputs={'input_ids': [1]}, offsets=[(0, len(text))])]
    with patch('privai_model.ml_span.runtime.windows', return_value=windows):
        assert detector.scan_text(text) == []
        detector.candidate_filter = None
        assert detector.scan_text(text) == [span]


def test_unknown_filter_rejected():
    with pytest.raises(RuntimeError, match='initialization failed'):
        MLSpanDetector(candidate_filter='unknown')
