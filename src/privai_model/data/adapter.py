"""Only this boundary knows candidate-v1. Never include raw records in errors."""
from dataclasses import dataclass, field
from pathlib import Path
import gzip
import json


class CandidateValidationError(ValueError):
    pass


@dataclass(frozen=True, repr=False)
class ModelExample:
    candidate_id: str
    group_id: str
    value: str = field(repr=False)
    line: str = field(repr=False)
    start: int
    end: int
    before: str = field(repr=False)
    after: str = field(repr=False)
    type: str
    source: dict = field(repr=False)
    detector: dict = field(repr=False)
    label: int | None
    reason: str | None
    provisional: bool

    def __repr__(self):
        return 'ModelExample(<redacted>)'


def _context(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return '\n'.join(value)
    raise CandidateValidationError('context must be text or list of text')


def adapt_record(record, require_label=True):
    if not isinstance(record, dict):
        raise CandidateValidationError('record must be an object')
    if record.get('granularity') != 'value':
        return None
    try:
        if record.get('schema_version', 'candidate-v1') != 'candidate-v1':
            raise CandidateValidationError('unsupported schema_version')
        cid, gid = record['candidate_id'], record['group_id']
        if not all(isinstance(v, str) and v.strip() for v in (cid, gid)):
            raise CandidateValidationError('candidate_id/group_id required')
        value, line = record['value'], record['line']
        text, start, end = line['text'], line['value_start'], line['value_end']
        if not isinstance(value, str) or not value or not isinstance(text, str):
            raise CandidateValidationError('value/text must be nonempty text')
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise CandidateValidationError('invalid value span')
        if text[start:end] != value:
            raise CandidateValidationError('value span mismatch')
        label = record.get('label')
        if require_label and label is None:
            raise CandidateValidationError('training label required')
        y, reason, provisional = None, None, False
        if label is not None:
            if not isinstance(label, dict) or label.get('action') not in ('MASK', 'KEEP'):
                raise CandidateValidationError('invalid label.action')
            y = {'MASK': 1, 'KEEP': 0}[label['action']]
            reason = label.get('reason')
            provisional = label.get('provisional', False)
            if type(provisional) is not bool:
                raise CandidateValidationError('label.provisional must be boolean')
        source, detector = record.get('source', {}), record.get('detector', {})
        context = record.get('context', {})
        if not all(isinstance(x, dict) for x in (source, detector, context)):
            raise CandidateValidationError('source/detector/context must be objects')
        kind = record.get('type', 'UNKNOWN')
        if not isinstance(kind, str):
            raise CandidateValidationError('type must be text')
        return ModelExample(cid, gid, value, text, start, end,
                            _context(context.get('before')), _context(context.get('after')),
                            kind, dict(source), dict(detector), y, reason, provisional)
    except (KeyError, TypeError, IndexError):
        raise CandidateValidationError('missing or malformed candidate-v1 field') from None


def adapt_records(records, require_label=True):
    examples, seen = [], set()
    for index, record in enumerate(records, 1):
        try:
            ex = adapt_record(record, require_label)
            if ex is None:
                continue
            if ex.candidate_id in seen:
                raise CandidateValidationError('duplicate candidate_id')
            seen.add(ex.candidate_id)
            examples.append(ex)
        except CandidateValidationError as exc:
            raise CandidateValidationError(f'record {index}: {exc}') from None
    return examples


def iter_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(Path(path), 'rt', encoding='utf-8') as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except (ValueError, TypeError):
                raise CandidateValidationError(f'invalid JSON at line {index}') from None


def load_examples(path, require_label=True):
    return adapt_records(iter_jsonl(path), require_label)
