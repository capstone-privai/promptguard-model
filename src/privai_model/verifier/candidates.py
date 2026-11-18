"""Back-end verifier data: PromptGuard detector candidates on span rows, labelled by the gold spans, as candidate-v1.

The PromptGuard detector (promptguard-claude-demoV0, src/detector) scans each span row as PromptGuard scans it and
returns `candidates`: every span any adapter found, merged where they overlap, each with a strength. A candidate is
`strong` when the majority vote kept it or a rule gave strong evidence, otherwise `ambiguous`: the low-confidence
candidates the detector can hand to a back-end verifier.

A candidate is MASK when it overlaps a gold span (masking it hides secret material), KEEP otherwise. The records are
candidate-v1, so the existing candidate classifier (src/privai_model/training) trains on them unchanged; the split
manifest keeps the span rows' split (by group), so a verifier never trains on a group its span model was tested on.

Records hold values and context: they are written under runs/ and never committed. The analysis holds counts only.
"""
from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import accumulate

from privai_model.data.adapter import adapt_records
from privai_model.data.split import NAMES, fingerprint, statistics

SPLITS = NAMES  # train, validation, test
TIERS = ('strong', 'ambiguous')


def code_points(text):
    """UTF-16 offset -> code-point offset for `text` (the detector reports UTF-16, span rows use code points)."""
    bounds = [0, *accumulate(2 if ord(ch) > 0xFFFF else 1 for ch in text)]

    def convert(offset):
        i = bisect_left(bounds, offset)
        if i == len(bounds) or bounds[i] != offset:
            raise ValueError('UTF-16 offset inside a surrogate pair')
        return i
    return convert


def match(start, end, gold):
    """How a candidate relates to the gold spans: exact, covers, inside, partial or none."""
    hits = [(a, b) for a, b in gold if max(start, a) < min(end, b)]
    if not hits:
        return 'none'
    if (start, end) in hits:
        return 'exact'
    if all(start <= a and b <= end for a, b in hits):
        return 'covers'
    if len(hits) == 1 and hits[0][0] <= start and end <= hits[0][1]:
        return 'inside'
    return 'partial'


def lines_around(text, start, end, n):
    """The lines holding [start, end) and up to n lines before and after them."""
    line_start = text.rfind('\n', 0, start) + 1
    line_end = text.find('\n', end)
    line_end = len(text) if line_end < 0 else line_end
    before = text[:line_start].split('\n')[:-1][-n:] if n else []
    after = text[line_end + 1:].split('\n')[:n] if n and line_end < len(text) else []
    return line_start, line_end, before, after


def evidence_summary(candidate):
    """Adapter/family pairs, the strongest evidence strength and the best model score, without values."""
    evidence = candidate.get('evidence', [])
    families = sorted({f"{e['adapter']}/{e['family']}" for e in evidence})
    scores = [e['score'] for e in evidence if e.get('score') is not None]
    return families, (max(scores) if scores else None)


def records_for_row(row, split, candidates, context_lines=3, dataset='creddata-traj-v1'):
    """candidate-v1 records for one span row from its detector candidates (UTF-16 offsets)."""
    text = row['text']
    to_cp = code_points(text)
    gold = [(s['start'], s['end']) for s in row['spans']]
    output = []
    for c in candidates:
        start, end = to_cp(c['start']), to_cp(c['end'])
        line_start, line_end, before, after = lines_around(text, start, end, context_lines)
        relation = match(start, end, gold)
        families, score = evidence_summary(c)
        output.append({
            'schema_version': 'candidate-v1', 'candidate_id': f"{row['id']}:{start}:{end}", 'group_id': row['family_id'],
            'granularity': 'value', 'value': text[start:end],
            'line': {'text': text[line_start:line_end], 'value_start': start - line_start, 'value_end': end - line_start},
            'context': {'before': before, 'after': after}, 'type': c['type'],
            'source': {'dataset': dataset, 'channel': row['source'], 'file_path': None, 'row_id': row['id'], 'split': split,
                       'start': start, 'end': end},
            'detector': {'name': '+'.join(c['detectors']), 'rule_id': c['rule_id'], 'raw_type': c['type'],
                         'score': score, 'strength': c['strength'], 'legacy_protected': c['legacy_protected'],
                         'evidence': families},
            'label': {'action': 'KEEP' if relation == 'none' else 'MASK', 'provisional': False,
                      'reason': f'gold overlap: {relation}'},
        })
    return output


def scan_rows(detector, rows, batch_size=64):
    """Detector candidates per row, batched through the detector's own scan_many (the production path)."""
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        results = detector.scan_many([(r['text'], r['source']) for r in chunk])
        for row, result in zip(chunk, results):
            if result['errors']:
                raise RuntimeError(f"detector errors on a row: {result['errors']}")  # codes only, never text
            yield row, result['candidates']


def split_manifest(records):
    """split-v1 manifest that keeps each record in its span row's split (restore_split checks it)."""
    examples = adapt_records(records)
    by_split = {name: [e for e in examples if e.source['split'] == name] for name in SPLITS}
    groups = {}
    for name, rows in by_split.items():
        for e in rows:
            if groups.setdefault(e.group_id, name) != name:
                raise ValueError('a group appears in two splits')
    return {'schema_version': 'split-v1', 'seed': None, 'ratios': None, 'rule': 'the span rows\' split (by group)',
            'dataset_fingerprint': fingerprint(examples), 'group_assignment': groups,
            'candidate_ids': {k: [e.candidate_id for e in v] for k, v in by_split.items()},
            'statistics': statistics(by_split)}


def rate(mask, total):
    return round(mask / total, 4) if total else None


def analysis(records, rows_by_split, splits=('train', 'validation')):
    """False-positive analysis, counts only: precision by tier, adapter set, rule and evidence family, and how much of
    the gold any candidate reaches. Test is left out: it is a hold-out."""
    report = {}
    for split in splits:
        recs = [r for r in records if r['source']['split'] == split]
        rows = rows_by_split[split]
        table = lambda key: _table(recs, key)
        gold = {(r['id'], s['start'], s['end']) for r in rows for s in r['spans']}
        gold_by_row = defaultdict(list)
        for rid, a, b in gold:
            gold_by_row[rid].append((a, b))
        reached = defaultdict(set)
        for r in recs:
            src = r['source']
            for a, b in gold_by_row.get(src['row_id'], ()):
                if max(a, src['start']) < min(b, src['end']):
                    reached['any'].add((src['row_id'], a, b))
                    reached[r['detector']['strength']].add((src['row_id'], a, b))
        report[split] = {
            'rows': len(rows), 'gold_spans': len(gold), 'candidates': len(recs),
            'by_tier': table(lambda r: r['detector']['strength']),
            'by_tier_and_adapters': table(lambda r: f"{r['detector']['strength']} | {r['detector']['name']}"),
            'by_tier_and_rule': table(lambda r: f"{r['detector']['strength']} | {r['detector']['rule_id']}"),
            'by_evidence_family': _family_table(recs),
            'by_match': dict(Counter(r['label']['reason'].split(': ')[1] for r in recs)),
            'gold_reached': {k: dict(spans=len(v), share=rate(len(v), len(gold))) for k, v in
                             (('any_candidate', reached['any']), ('strong', reached['strong']),
                              ('ambiguous_only', reached['ambiguous'] - reached['strong']))},
            'gold_missed_by_every_candidate': len(gold - reached['any']),
        }
    return report


def _table(records, key):
    counts = defaultdict(Counter)
    for r in records:
        counts[key(r)][r['label']['action']] += 1
    rows = {k: dict(candidates=c['MASK'] + c['KEEP'], mask=c['MASK'], keep=c['KEEP'],
                    precision=rate(c['MASK'], c['MASK'] + c['KEEP'])) for k, c in counts.items()}
    return dict(sorted(rows.items(), key=lambda kv: -kv[1]['candidates']))


def _family_table(records):
    counts = defaultdict(Counter)
    for r in records:
        for family in r['detector']['evidence']:
            counts[f"{r['detector']['strength']} | {family}"][r['label']['action']] += 1
    rows = {k: dict(candidates=c['MASK'] + c['KEEP'], mask=c['MASK'], keep=c['KEEP'],
                    precision=rate(c['MASK'], c['MASK'] + c['KEEP'])) for k, c in counts.items()}
    return dict(sorted(rows.items(), key=lambda kv: -kv[1]['candidates']))


def system_metrics(records, gold_by_row, scores=None, threshold=None):
    """What PromptGuard would mask, at span level: overlap recall of the gold spans and precision of masked candidates.

    Masked set: every candidate (scores None, today's union policy), or strong candidates plus ambiguous ones whose
    verifier score is at least `threshold` (a missing score keeps the candidate masked: fail closed)."""
    masked = [r for r in records if scores is None or r['detector']['strength'] == 'strong'
              or scores.get(r['candidate_id'], 1.0) >= threshold]
    hit = set()
    for r in masked:
        src = r['source']
        for a, b in gold_by_row.get(src['row_id'], ()):
            if max(a, src['start']) < min(b, src['end']):
                hit.add((src['row_id'], a, b))
    gold = sum(len(v) for v in gold_by_row.values())
    true_masks = sum(r['label']['action'] == 'MASK' for r in masked)
    precision, recall = rate(true_masks, len(masked)), rate(len(hit), gold)
    f2 = round(5 * precision * recall / (4 * precision + recall), 4) if precision and recall else None
    return dict(masked=len(masked), false_masks=len(masked) - true_masks, gold_spans=gold, gold_reached=len(hit),
                precision=precision, recall=recall, f2=f2)
