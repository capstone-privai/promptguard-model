"""Marker ownership and candidate-centered token budgets live here."""
from dataclasses import dataclass

MARKERS = {'open': '[V]', 'close': '[/V]', 'cut': '[V_TRUNCATED]'}


@dataclass(repr=False)
class InputParts:
    metadata: str
    left: str
    value: str
    right: str

    @property
    def text(self):
        return self.metadata + self.left + MARKERS['open'] + self.value + MARKERS['close'] + self.right


def build_input(ex):
    metadata = f'[TYPE] {ex.type}\n'
    for tag, key in [('CHANNEL','channel'), ('FILE','file_path'), ('TOOL','tool_name'), ('AGENT','agent')]:
        metadata += f'[{tag}] {ex.source.get(key) or "UNKNOWN"}\n'
    for key in ('name', 'rule_id', 'raw_type', 'score'):
        metadata += f'[DETECTOR_{key.upper()}] {ex.detector.get(key, "UNKNOWN")}\n'
    return InputParts(metadata, '[BEFORE]\n'+ex.before+'\n[LINE]\n'+ex.line[:ex.start],
                      ex.value, ex.line[ex.end:]+'\n[AFTER]\n'+ex.after)


def register_markers(tokenizer):
    return tokenizer.add_special_tokens({'additional_special_tokens': list(MARKERS.values())})


def encode_candidate(ex, tokenizer, window_size):
    parts = build_input(ex)
    # Segments may exceed the backbone limit; only the bounded final IDs reach it.
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False, verbose=False)
    op, cl, cut = (enc(MARKERS[k]) for k in ('open','close','cut'))
    if any(len(x) != 1 for x in (op, cl, cut)):
        raise ValueError('register marker special tokens before encoding')
    budget = window_size - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < len(op)+len(cl)+3:
        raise ValueError('window too small to preserve candidate markers and content')
    value, left, right, meta = enc(parts.value), enc(parts.left), enc(parts.right), enc(parts.metadata)
    value_budget = budget-len(op)-len(cl)
    truncated = len(value) > value_budget
    if truncated:
        keep = value_budget-len(cut)
        front = (keep+1)//2
        value = value[:front]+cut+value[-(keep-front):] if keep-front else value[:front]+cut
    anchor = op+value+cl
    remaining = budget-len(anchor)
    # Reserve at most 1/4 of remaining budget for metadata, then nearest context.
    m = min(len(meta), remaining//4)
    remaining -= m
    l = min(len(left), (remaining+1)//2)
    r = min(len(right), remaining-l)
    l = min(len(left), remaining-r)
    extra = remaining-l-r
    m = min(len(meta), m+extra)
    ids = meta[:m]+(left[-l:] if l else [])+anchor+right[:r]
    # Fast generic tokenizers may ignore backend post-processors in prepare_for_model.
    # Obtain the true single-sequence special-token template through __call__.
    template = tokenizer(MARKERS['open'], add_special_tokens=True,
                         return_attention_mask=False, return_token_type_ids=False)['input_ids']
    if template.count(op[0]) != 1:
        raise ValueError('unsupported tokenizer single-sequence template')
    pivot = template.index(op[0])
    final_ids = template[:pivot] + ids + template[pivot+1:]
    encoded = {'input_ids': final_ids, 'attention_mask': [1]*len(final_ids)}
    if len(encoded['input_ids']) > window_size:
        raise ValueError('tokenizer special-token budget mismatch')
    return dict(encoded), {'candidate_truncated': truncated, 'tokens': len(encoded['input_ids'])}
