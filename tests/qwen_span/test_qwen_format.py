import json
import pytest
from privai_model.qwen_span import format as fmt
from privai_model.qwen_span.scoring import item_scores, token_char_ranges


def row(text, ranges):
    spans = [dict(start=a, end=b, secret_type='SECRET') for a, b in ranges]
    return dict(text=text, spans=spans)


@pytest.mark.parametrize('text,ranges', [
    ('token=abc abc', [(10, 13)]),                       # second of two identical values
    ('k=abc; j=abc', [(2, 5), (9, 12)]),                 # both identical values
    ('aaaa', [(1, 3)]),                                  # overlapping occurrences
    ('암호는 가상!123입니다.', [(4, 10)]),
    ('🔒 비밀\r\n가상é!', [(6, 10)]),
    ('-----BEGIN KEY-----\nAb9\n-----END KEY-----', [(20, 23)]),
    ('p="a\\"b"', [(3, 7)]),                             # JSON escaping inside the value
    ('nothing here', []),
    ('', []),
])
def test_target_roundtrip_is_exact(text, ranges):
    r = row(text, ranges)
    assert fmt.check_roundtrip(r) == set(ranges)
    answer = fmt.target(r)
    assert json.loads(answer)['secrets'] == fmt.target_items(text, r['spans'])


def test_occurrence_index_is_not_repaired():
    text = 'a=SECRET1 b=SECRET1'
    result = fmt.recover(text, '{"secrets":[{"value":"SECRET1","occurrence":3}]}')
    assert result['spans'] == [] and result['counts'][fmt.ITEM_BAD_OCCURRENCE] == 1
    result = fmt.recover(text, '{"secrets":[{"value":"SECRET1","occurrence":0}]}')
    assert result['spans'] == [] and result['counts'][fmt.ITEM_BAD_OCCURRENCE] == 1


def test_hallucinated_value_is_dropped_not_fuzzy_matched():
    result = fmt.recover('key=Ab9xYz', '{"secrets":[{"value":"Ab9xYZ","occurrence":1}]}')
    assert result['spans'] == [] and result['counts'][fmt.ITEM_HALLUCINATED] == 1


def test_duplicates_and_malformed_items_counted():
    raw = '{"secrets":[{"value":"abc","occurrence":1},{"value":"abc","occurrence":1},{"value":"abc"},{"value":"abc","occurrence":true}]}'
    result = fmt.recover('x=abc', raw)
    assert [(s['start'], s['end']) for s in result['spans']] == [(2, 5)]
    assert result['counts'][fmt.ITEM_DUPLICATE] == 1 and result['counts'][fmt.ITEM_MALFORMED] == 2
    assert result['canonical'] is False


@pytest.mark.parametrize('raw,error', [
    ('', 'invalid_json'), ('{"secrets":[', 'invalid_json'), ('```json\n{"secrets":[]}\n```', 'invalid_json'),
    ('[]', 'invalid_schema'), ('{"secrets":{}}', 'invalid_schema'), ('{"secrets":[],"x":1}', 'invalid_schema'),
])
def test_malformed_answers_fail_closed(raw, error):
    result = fmt.recover('x=abc', raw)
    assert result['error'] == error and result['spans'] == []


def test_canonical_item_ranges_point_at_items():
    text = 'a=111 b=222'
    answer = ' ' + fmt.target(row(text, [(2, 5), (8, 11)]))
    result = fmt.recover(text, answer)
    assert result['canonical']
    pieces = [json.loads(answer[a:b])['value'] for a, b in result['item_ranges']]
    assert pieces == ['111', '222']


def test_prompt_carries_text_only():
    m = fmt.messages('x=abc')
    assert [x['role'] for x in m] == ['system', 'user'] and m[1]['content'] == '<text>\nx=abc\n</text>'


class CharTokenizer:
    """Each token id is one character code; enough to exercise range attribution."""
    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(i) for i in ids)


def test_item_scores_attribute_token_probabilities():
    import math
    text = 'a=111 b=222'
    answer = fmt.target(row(text, [(2, 5), (8, 11)]))
    result = fmt.recover(text, answer)
    ids = [ord(c) for c in answer]
    ranges = token_char_ranges(CharTokenizer(), ids)
    assert ranges[0] == (0, 1) and ranges[-1] == (len(answer) - 1, len(answer))
    a, b = result['item_ranges'][1]
    logprobs = [math.log(0.5) if a <= i < b else 0.0 for i in range(len(ids))]
    first, second = item_scores(result, logprobs, ranges)
    assert first == pytest.approx(1.0) and second == pytest.approx(0.5)
