import pytest
from privai_model.data.synthetic import hard_cases
from privai_model.data.adapter import adapt_records
from privai_model.data.input_builder import encode_candidate


@pytest.mark.parametrize('window',[64,256,512,1024])
def test_long_fixture_fills_window_and_preserves_marker(tokenizer,window):
    examples=adapt_records(hard_cases(long_context=True))
    assert len(examples)==18 and len({e.candidate_id for e in examples})==18
    for ex in examples:
        tokens,info=encode_candidate(ex,tokenizer,window)
        assert len(tokens['input_ids'])==window
        assert tokenizer.convert_tokens_to_ids('[V]') in tokens['input_ids']
        assert tokenizer.convert_tokens_to_ids('[/V]') in tokens['input_ids']
