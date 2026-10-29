"""Generative span contract: the model copies values, offsets are recovered locally.

The model never emits character offsets. It emits each secret value together with
its 1-based occurrence index among all (possibly overlapping) exact matches in the
input. Recovery is deterministic and never "repairs" a wrong answer:

* a value that does not occur in the text is a hallucination and is dropped;
* an occurrence index that does not exist is a bad occurrence and is dropped,
  it is NOT moved to another match;
* malformed output yields no spans and is counted, so callers can fail closed.

No function here logs or embeds input text in errors.
"""
import json

SCHEMA_VERSION = 'qwen-span-json-v1'

SYSTEM_PROMPT = (
    'You are a secret-span detector for an AI coding agent. Find every secret value in the '
    'text: real credentials such as passwords, API keys, access tokens, private keys, '
    'connection-string passwords and session cookies. Public identifiers, placeholders, '
    'redacted values, documentation examples, hashes and test fixtures that are clearly '
    'marked as public are NOT secrets.\n'
    'Answer with JSON only, exactly in this form: '
    '{"secrets":[{"value":"<exact substring>","occurrence":<n>}]}\n'
    'Copy each value character for character, without surrounding quotes or key names. '
    '"occurrence" is the 1-based index of that exact substring among all of its '
    'occurrences in the text. List secrets in text order. If there are none, answer '
    '{"secrets":[]}'
)

ITEM_OK = 'ok'
ITEM_HALLUCINATED = 'hallucinated'
ITEM_BAD_OCCURRENCE = 'bad_occurrence'
ITEM_DUPLICATE = 'duplicate'
ITEM_MALFORMED = 'malformed_item'
ITEM_STATUSES = (ITEM_OK, ITEM_HALLUCINATED, ITEM_BAD_OCCURRENCE, ITEM_DUPLICATE, ITEM_MALFORMED)


def user_prompt(text):
    if not isinstance(text, str):
        raise ValueError('text must be a string')
    return '<text>\n' + text + '\n</text>'


def messages(text, answer=None):
    result = [dict(role='system', content=SYSTEM_PROMPT), dict(role='user', content=user_prompt(text))]
    if answer is not None:
        result.append(dict(role='assistant', content=answer))
    return result


def occurrences(text, value):
    """All start offsets of value in text, overlapping matches included."""
    if not value:
        return []
    result = []
    start = text.find(value)
    while start != -1:
        result.append(start)
        start = text.find(value, start + 1)
    return result


def serialize(items):
    """Canonical compact JSON plus each item's [start, end) range in the output string.

    The ranges let inference attribute generated-token probabilities to one item.
    """
    head = '{"secrets":['
    out = [head]
    cursor = len(head)
    ranges = []
    for i, item in enumerate(items):
        if i:
            out.append(',')
            cursor += 1
        piece = json.dumps(dict(value=item['value'], occurrence=item['occurrence']),
                           ensure_ascii=False, separators=(',', ':'))
        out.append(piece)
        ranges.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    out.append(']}')
    return ''.join(out), ranges


def target_items(text, spans):
    items = []
    for span in sorted(spans, key=lambda s: (s['start'], s['end'])):
        value = text[span['start']:span['end']]
        starts = occurrences(text, value)
        if span['start'] not in starts:
            raise ValueError('span is not addressable')
        items.append(dict(value=value, occurrence=starts.index(span['start']) + 1))
    return items


def target(row):
    """Training answer for one secret-span-v1 row (gold `spans`, not `value_spans`)."""
    return serialize(target_items(row['text'], row['spans']))[0]


def parse(raw):
    """Return (items, error). error is None, 'invalid_json' or 'invalid_schema'."""
    if not isinstance(raw, str):
        return None, 'invalid_json'
    try:
        data = json.loads(raw.strip())
    except (ValueError, RecursionError):
        return None, 'invalid_json'
    if not isinstance(data, dict) or set(data) != {'secrets'} or not isinstance(data['secrets'], list):
        return None, 'invalid_schema'
    return data['secrets'], None


def recover(text, raw):
    """Map a model answer to exact character spans without searching for substitutes."""
    items, error = parse(raw)
    result = dict(spans=[], error=error, items=[], counts={k: 0 for k in ITEM_STATUSES},
                  canonical=False, item_ranges=None)
    if error is not None:
        return result
    seen = set()
    well_formed = []
    for item in items:
        if (not isinstance(item, dict) or set(item) != {'value', 'occurrence'}
                or not isinstance(item['value'], str) or type(item['occurrence']) is not int):
            status, span = ITEM_MALFORMED, None
        else:
            well_formed.append(item)
            starts = occurrences(text, item['value'])
            if not starts:
                status, span = ITEM_HALLUCINATED, None
            elif not 1 <= item['occurrence'] <= len(starts):
                status, span = ITEM_BAD_OCCURRENCE, None
            else:
                a = starts[item['occurrence'] - 1]
                span = (a, a + len(item['value']))
                status = ITEM_DUPLICATE if span in seen else ITEM_OK
        result['counts'][status] += 1
        result['items'].append(dict(status=status, span=span))
        if status == ITEM_OK:
            seen.add(span)
            result['spans'].append(dict(start=span[0], end=span[1], secret_type='SECRET'))
    if len(well_formed) == len(items):
        canonical, ranges = serialize(items)
        if canonical == raw.strip():
            lead = len(raw) - len(raw.lstrip())
            result['canonical'] = True
            # Ranges index the raw answer string (leading whitespace included).
            result['item_ranges'] = [(a + lead, b + lead) for a, b in ranges]
    result['spans'].sort(key=lambda s: (s['start'], s['end']))
    return result


def check_roundtrip(row):
    """Gold spans must survive target -> recover exactly; returns recovered (start, end) set."""
    recovered = recover(row['text'], target(row))
    if recovered['error'] is not None or not recovered['canonical']:
        raise ValueError('target does not parse canonically')
    return {(s['start'], s['end']) for s in recovered['spans']}
