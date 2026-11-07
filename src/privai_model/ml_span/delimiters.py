"""Trim the closing quote of a quoted value when an ML span runs past it.

mmBERT's tokenizer merges punctuation runs such as `)";` or `.",` into one token, so a token-level
span cannot stop between a value's last character and its closing quote: the model either drops the
last character or swallows the quote. This opt-in policy cuts a span at the quote that matches the
one just before the span, when only that quote and trailing delimiters (`;`, `,`, `)`, `]`, `}`)
follow it inside the span. Escaped quotes (`\\"`) and a doubled quote (a value that itself ends with
the quote character) are left alone, and so is a span that does not sit right after an opening quote.
An opening quote swallowed at the start is dropped the same way.

quote-delimiters-v2 also moves the start. The tokenizer glues a value's leading punctuation to what
comes before it (`"+`, `="#`, `:%`), so the span starts one token late (after `"+`) or early (at `:%`).
v2 extends a start back over up to three such characters when an opening quote precedes them, and drops
a leading `=`/`:` (and a quote after it) that directly follows a key name. Neither rule changes a gold
span in data/ or runs/data.
"""
POLICY = 'quote-delimiters-v1'
POLICY_V2 = 'quote-delimiters-v2'
POLICIES = (POLICY, POLICY_V2)
QUOTES = '"\'`'
TRAILING = ';,)]}'
LEADING = '+/#%_-!^*~@$&?.='
SEPARATORS = '=:'


def _start(text, a, b):
    k = a
    while k > 0 and text[k - 1] in LEADING and a - k < 3:
        k -= 1
    if k < a and k > 0 and text[k - 1] in QUOTES:
        return k
    if a > 0 and (text[a - 1].isalnum() or text[a - 1] == '_') and text[a] in SEPARATORS:
        k = a
        while k < b - 1 and text[k] in SEPARATORS:
            k += 1
        if k < b - 1 and text[k] in QUOTES:
            k += 1
        return k
    return a


def _trim(text, a, b):
    q = text[a - 1] if a > 0 and text[a - 1] in QUOTES else None
    if q is None and b - a > 1 and text[a] in QUOTES and b < len(text) and text[b] == text[a] and text[a] not in text[a + 1:b]:
        return a + 1, b
    if q is None:
        return a, b
    k = a
    while k < b:
        if text[k] == q and text[k - 1] != '\\':
            if k + 1 < len(text) and text[k + 1] == q:
                k += 1  # doubled: the value ends with the quote character, the next one closes it
            if k > a and all(ch == q or ch in TRAILING for ch in text[k:b]):
                return a, k
            return a, b
        k += 1
    return a, b


def trim_quote_delimiters(text, spans, policy=POLICY):
    if policy not in POLICIES:
        raise ValueError('unknown delimiter policy')
    result = []
    for span in spans:
        a = _start(text, span['start'], span['end']) if policy == POLICY_V2 else span['start']
        a, b = _trim(text, a, span['end'])
        result.append(span if (a, b) == (span['start'], span['end']) else dict(span, start=a, end=b))
    return result
