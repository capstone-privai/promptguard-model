import pytest
from privai_model.data.adapter import adapt_record
from privai_model.data.input_builder import build_input, encode_candidate, MARKERS


def test_builder(record):
    text = build_input(adapt_record(record)).text
    assert '[V]synthetic-token[/V]' in text
    for expected in ('[TYPE]', '[CHANNEL]', '[FILE]', '[TOOL]', '[DETECTOR_NAME]', '[BEFORE]', '[AFTER]'):
        assert expected in text


@pytest.mark.parametrize('window', [64,128,256,512,1024])
def test_long_context_preserves_candidate(record, tokenizer, window):
    record['context'] = {'before': 'x '*5000, 'after': 'tail '*5000}
    ex = adapt_record(record)
    encoded, info = encode_candidate(ex, tokenizer, window)
    ids = encoded['input_ids']
    start = ids.index(tokenizer.convert_tokens_to_ids('[V]'))
    end = ids.index(tokenizer.convert_tokens_to_ids('[/V]'))
    assert ids[start+1:end] == tokenizer.encode(ex.value, add_special_tokens=False)
    assert len(ids) <= window
    assert not info['candidate_truncated']


def test_oversized_candidate(record, tokenizer):
    value = 'x '*1000+'tail'
    record.update(value=value, line={'text':value,'value_start':0,'value_end':len(value)})
    encoded, info = encode_candidate(adapt_record(record), tokenizer, 64)
    ids = encoded['input_ids']
    assert info['candidate_truncated']
    assert len(ids) == 64
    assert all(tokenizer.convert_tokens_to_ids(m) in ids for m in MARKERS.values())


def test_tiny_window(record, tokenizer):
    with pytest.raises(ValueError):
        encode_candidate(adapt_record(record), tokenizer, 4)
