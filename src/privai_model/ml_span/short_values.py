"""Drop ML candidates whose value is too short to be a credential.

This is the check PromptGuard's ml-v9 adapter already applies (`min_value_length`, default 6), so a
caller of `MLSpanDetector` outside PromptGuard gets the same guard. Surrounding whitespace, quotes
and punctuation are stripped first, so `"ab",` counts as two characters. It is an opt-in ML candidate
policy: rule candidates are not filtered. Short labelled secrets (four- or five-character weak
passwords) are dropped as well and must come from rule detectors.
"""
PROMPTGUARD_MIN_VALUE_LENGTH = 6
STRIP = '"\'`.,;:()[]{}'


def check_min_value_length(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError('min_value_length must be a positive integer')


def filter_short_values(text, spans, min_length):
    return [s for s in spans if len(text[s['start']:s['end']].strip().strip(STRIP)) >= min_length]
