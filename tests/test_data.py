import copy
import gzip
import json
import pytest
from conftest import candidate
from privai_model.data.adapter import adapt_record, adapt_records, load_examples, CandidateValidationError
from privai_model.data.split import split_groups, restore_split


def test_gzip_filter_labels(tmp_path):
    records = [candidate(0), candidate(1, 'KEEP'), {'granularity': 'line'}]
    path = tmp_path/'fixture.jsonl.gz'
    with gzip.open(path, 'wt') as f:
        for r in records:
            f.write(json.dumps(r)+'\n')
    examples = load_examples(path)
    assert [e.label for e in examples] == [1,0]
    assert all(e.provisional for e in examples)


@pytest.mark.parametrize('change', [
    {'value_start': -1}, {'value_start': True}, {'value_end': 999}, {'value_start': 5}, {'value_end': 4}])
def test_span_invalid(record, change):
    record['line'].update(change)
    with pytest.raises(CandidateValidationError):
        adapt_record(record)


@pytest.mark.parametrize('field', ['candidate_id','group_id','value','line','label'])
def test_missing_fields(record, field):
    del record[field]
    with pytest.raises(CandidateValidationError):
        adapt_record(record)


def test_invalid_label(record):
    record['label']['action'] = 'UNKNOWN'
    with pytest.raises(CandidateValidationError):
        adapt_record(record)


def test_duplicate(record):
    with pytest.raises(CandidateValidationError):
        adapt_records([record, record])


def test_inference_without_label(record):
    del record['label']
    assert adapt_record(record, require_label=False).label is None


def test_error_redaction(record):
    record['line']['value_start'] = -1
    with pytest.raises(CandidateValidationError) as exc:
        adapt_record(record)
    assert record['value'] not in str(exc.value)


def test_group_split_reproducible():
    examples = adapt_records([candidate(i, 'MASK' if i%2 else 'KEEP', f'g{i//2}') for i in range(60)])
    a, m = split_groups(examples, seed=9)
    b, n = split_groups(list(reversed(examples)), seed=9)
    assert m == n
    assert not m['statistics']['group_overlap']
    groups = [{e.group_id for e in rows} for rows in a.values()]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert sum(len(v) for v in a.values()) == 60
    assert restore_split(examples, m) == a
    changed = list(examples[:-1])
    with pytest.raises(ValueError):
        restore_split(changed, m)


def test_corrupt_manifest():
    examples = adapt_records([candidate(i) for i in range(30)])
    _, manifest = split_groups(examples)
    manifest['candidate_ids']['test'].append(manifest['candidate_ids']['train'][0])
    with pytest.raises(ValueError):
        restore_split(examples, manifest)
