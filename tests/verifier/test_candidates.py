"""Verifier data: detector candidates (UTF-16) become candidate-v1 records labelled by gold overlap, in the span rows'
split; the system metrics compare union, strong-only and strong+verifier masking. All values are synthetic."""
import pytest

from privai_model.data.adapter import adapt_records
from privai_model.data.split import restore_split
from privai_model.verifier.candidates import (analysis, code_points, lines_around, match, records_for_row,
                                              split_manifest, system_metrics)


def utf16(text, cp):
    return len(text[:cp].encode('utf-16-le')) // 2


def test_code_points_undo_utf16_offsets_across_astral_characters():
    text = 'a😀b 비밀 c😀d'
    to_cp = code_points(text)
    for cp in range(len(text) + 1):
        assert to_cp(utf16(text, cp)) == cp
    with pytest.raises(ValueError):
        to_cp(2)  # inside the first emoji's surrogate pair


@pytest.mark.parametrize('span, expected', [((5, 10), 'exact'), ((3, 12), 'covers'), ((6, 9), 'inside'),
                                            ((8, 22), 'partial'), ((11, 14), 'none'), ((3, 25), 'covers')])
def test_match_kinds(span, expected):
    assert match(*span, [(5, 10), (20, 24)]) == expected


def test_lines_around_returns_the_value_lines_and_neighbours():
    text = 'one\ntwo\nkey=VALUE1 end\nfour\nfive'
    start = text.index('VALUE1')
    line_start, line_end, before, after = lines_around(text, start, start + 6, 1)
    assert text[line_start:line_end] == 'key=VALUE1 end'
    assert before == ['two'] and after == ['four']
    assert lines_around(text, start, start + 6, 0)[2:] == ([], [])


def row(rid, text, gold, family='g1'):
    spans = [dict(start=text.index(g), end=text.index(g) + len(g), secret_type='SECRET') for g in gold]
    return dict(id=rid, text=text, spans=spans, value_spans=spans, source='tool_output', family_id=family)


def candidate(text, value, strength='ambiguous', score=None):
    start = text.index(value)
    evidence = [dict(adapter='promptguard-rules', family='secret-assignment', strength='moderate')]
    if score is not None:
        evidence.append(dict(adapter='ml-span', family='ml', strength='ambiguous', score=score))
    return dict(start=utf16(text, start), end=utf16(text, start + len(value)), type='PASSWORD',
                rule_id='secret-assignment', detectors=['promptguard-rules'], strength=strength,
                legacy_protected=strength == 'strong', evidence=evidence)


TEXT = '😀 setup\npassword = SynthPass_91\nbuild_id = Build_7731x\ndone'


def records(split='train', family='g1', rid='r1'):
    r = row(rid, TEXT, ['SynthPass_91'], family)
    return records_for_row(r, split, [candidate(TEXT, 'SynthPass_91', 'strong'), candidate(TEXT, 'Build_7731x', score=0.4)])


def test_records_are_valid_candidate_v1_labelled_by_gold_overlap():
    recs = records()
    assert [r['label']['action'] for r in recs] == ['MASK', 'KEEP']
    assert [r['detector']['strength'] for r in recs] == ['strong', 'ambiguous']
    examples = adapt_records(recs)
    assert [e.value for e in examples] == ['SynthPass_91', 'Build_7731x']
    assert examples[1].line == 'build_id = Build_7731x' and examples[1].before == '😀 setup\npassword = SynthPass_91'
    assert recs[1]['detector']['score'] == 0.4 and recs[0]['detector']['score'] is None
    assert recs[0]['candidate_id'] == f"r1:{TEXT.index('SynthPass_91')}:{TEXT.index('SynthPass_91') + 12}"


def test_split_manifest_keeps_the_row_split_and_rejects_a_group_in_two_splits():
    recs = records('train', 'g1', 'r1') + records('validation', 'g2', 'r2') + records('test', 'g3', 'r3')
    manifest = split_manifest(recs)
    splits = restore_split(adapt_records(recs), manifest)
    assert {name: len(v) for name, v in splits.items()} == {'train': 2, 'validation': 2, 'test': 2}
    with pytest.raises(ValueError, match='two splits'):
        split_manifest(records('train', 'g1', 'r1') + records('test', 'g1', 'r9'))


def test_analysis_counts_precision_by_tier_and_gold_reach_without_values():
    recs = records()
    rows = {'train': [row('r1', TEXT, ['SynthPass_91']), row('r0', 'nothing here', [])], 'validation': []}
    report = analysis(recs, rows)['train']
    assert report['by_tier'] == {'strong': dict(candidates=1, mask=1, keep=0, precision=1.0),
                                 'ambiguous': dict(candidates=1, mask=0, keep=1, precision=0.0)}
    assert report['gold_reached']['any_candidate'] == dict(spans=1, share=1.0)
    assert report['gold_missed_by_every_candidate'] == 0
    assert 'SynthPass' not in repr(report) and 'Build_' not in repr(report)


def test_system_metrics_compare_union_strong_only_and_verifier():
    recs = records()
    start = TEXT.index('SynthPass_91')
    gold = {'r1': [(start, start + 12)]}
    union = system_metrics(recs, gold)
    assert (union['masked'], union['false_masks'], union['recall']) == (2, 1, 1.0)
    strong = system_metrics(recs, gold, scores={}, threshold=2.0)
    assert (strong['masked'], strong['false_masks'], strong['precision']) == (1, 0, 1.0)
    ambiguous_id = recs[1]['candidate_id']
    assert system_metrics(recs, gold, scores={ambiguous_id: 0.2}, threshold=0.5)['masked'] == 1
    assert system_metrics(recs, gold, scores={ambiguous_id: 0.7}, threshold=0.5)['masked'] == 2
