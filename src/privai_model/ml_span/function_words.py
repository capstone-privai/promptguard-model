"""Suppress standalone grammatical-word candidates without trimming secret values.

This is an opt-in ML candidate policy, not a rule-detector allowlist. Quoted,
assigned and explicitly credential-labelled values are retained, even if short.
Unlabelled secrets consisting solely of a listed word remain an unavoidable risk.
"""
import re
import unicodedata

POLICY = 'contextual-function-words-v1'
FUNCTION_WORDS = frozenset('''a an the and or but nor so yet for by of to in on at
with from as via into onto upon over under than if else because although though
while unless since until whether either neither both'''.split())
_CREDENTIAL = r'(?:password|passwd|pwd|passphrase|secret|token|credential|api[ _-]?key|access[ _-]?key|authorization|bearer|비밀번호|암호|비밀값|토큰|인증키)'
_LABEL_BEFORE = re.compile(
    _CREDENTIAL + r'(?:은|는|이|가)?(?:\s+(?:value|is|was|equals|exactly|literally|the|actual|literal|word|값(?:은|는)?))*\s*[:=]?\s*$', re.I)
_LABEL_AFTER = re.compile(
    r'^\s*(?:(?:is|was)\s+(?:(?:the|a|an|my|our|actual)\s+)*' + _CREDENTIAL
    + r'|(?:은|는|이|가)\s*(?:실제\s*)?' + _CREDENTIAL + r')', re.I)


def _joined(text, start, end):
    """A token fragment inside an identifier/credential must never be suppressed."""
    for i in (start - 1, end):
        if not 0 <= i < len(text):
            continue
        c = text[i]
        if c.isalnum() or unicodedata.category(c).startswith('M') or c in '_-+/=@%\\':
            return True
        if c == '.' and 0 <= (j := i - 1 if i < start else i + 1) < len(text) and text[j].isalnum():
            return True
    return False


def filter_function_words(text, spans):
    result = []
    for span in spans:
        a, b = span['start'], span['end']
        value = text[a:b]
        if value.casefold() not in FUNCTION_WORDS or _joined(text, a, b):
            result.append(span)
            continue
        # Preserve quoted/backticked literals and uncertain delimiter fragments.
        before, after = text[max(0, a - 128):a], text[b:b + 128]
        quoted = (bool(before.rstrip()) and before.rstrip()[-1] in '\"\'`“‘') or (bool(after.lstrip()) and after.lstrip()[0] in '\"\'`”’')
        assigned = re.search(r'[:=]\s*$', before) is not None
        if quoted or assigned or _LABEL_BEFORE.search(before) or _LABEL_AFTER.search(after):
            result.append(span)
    return result
