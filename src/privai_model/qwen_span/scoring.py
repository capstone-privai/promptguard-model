"""Attribute generated-token probabilities to answer items (torch-free)."""
import math


def token_char_ranges(tokenizer, token_ids):
    """Character range of each generated token inside the decoded answer.

    Byte-level BPE can split one Hangul/emoji character over several tokens; prefix
    decoding then briefly shows U+FFFD. Ranges are clamped monotone, which is enough
    to attribute token probabilities to answer items.
    """
    ranges, previous = [], 0
    for i in range(len(token_ids)):
        prefix = tokenizer.decode(token_ids[:i + 1], skip_special_tokens=True)
        end = len(prefix.rstrip('�'))
        end = max(previous, end)
        ranges.append((previous, end))
        previous = end
    return ranges


def item_scores(recovered, logprobs, ranges):
    if not logprobs:
        return []
    sequence = math.exp(sum(logprobs) / len(logprobs))
    if not recovered['canonical']:
        return [sequence] * len(recovered['items'])
    scores = []
    for a, b in recovered['item_ranges']:
        selected = [lp for lp, (c, d) in zip(logprobs, ranges) if max(a, c) < min(b, d)]
        scores.append(math.exp(sum(selected) / len(selected)) if selected else sequence)
    return scores
