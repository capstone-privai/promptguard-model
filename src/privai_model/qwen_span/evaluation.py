"""Exact-span scoring for generative predictions and per-record comparison with v9.

Pure Python (no torch) so it is testable anywhere. Inputs are prediction records:
{id, spans:[{start,end,score}], error, counts, latency_ms}. Reports contain counts,
IDs and offsets only, never text or values.
"""
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from . import format as fmt


def measures(gold, predicted):
    tp = len(gold & predicted); fp = len(predicted - gold); fn = len(gold - predicted)
    return dict(tp=tp, fp=fp, fn=fn,
                precision=tp / len(predicted) if predicted else None,
                recall=tp / len(gold) if gold else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                f2=5 * tp / (5 * tp + fp + 4 * fn) if 5 * tp + fp + 4 * fn else None)


def gold_keys(rows):
    return {(r['id'], s['start'], s['end']) for r in rows for s in r['spans']}


def predicted_keys(rows, predictions, threshold):
    return {(r['id'], s['start'], s['end']) for r in rows
            for s in predictions[r['id']]['spans'] if s['score'] >= threshold}


def latency(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(unit='ms/input', calls=len(values), mean=statistics.mean(values), p50=statistics.median(values),
                p95=ordered[min(len(values) - 1, math.ceil(.95 * len(values)) - 1)])


def check_coverage(rows, predictions):
    ids = [r['id'] for r in rows]
    if not rows or len(set(ids)) != len(ids) or set(predictions) != set(ids):
        raise ValueError('complete unique prediction coverage required')


def score(rows, predictions, threshold=0.0):
    check_coverage(rows, predictions)
    gold = gold_keys(rows)
    predicted = predicted_keys(rows, predictions, threshold)
    errors = Counter(p['error'] for p in predictions.values() if p['error'] is not None)
    items = Counter()
    for p in predictions.values():
        items.update(p.get('counts', {}))
    negatives = {r['id'] for r in rows if not r['spans']}
    result = dict(records=len(rows), gold_spans=len(gold), threshold=threshold, exact=measures(gold, predicted),
                  invalid_outputs=dict(errors), invalid_output_rate=sum(errors.values()) / len(rows),
                  answer_items=dict(items),
                  hallucinated_items=items.get(fmt.ITEM_HALLUCINATED, 0),
                  bad_occurrence_items=items.get(fmt.ITEM_BAD_OCCURRENCE, 0),
                  negative_rows_with_findings=len({k[0] for k in predicted} & negatives),
                  overlap_recall=overlap_recall(rows, predictions, threshold),
                  latency=latency([p['latency_ms'] for p in predictions.values() if 'latency_ms' in p]))
    return result


def overlap_recall(rows, predictions, threshold):
    gold = hit = 0
    for r in rows:
        pred = [s for s in predictions[r['id']]['spans'] if s['score'] >= threshold]
        for g in r['spans']:
            gold += 1
            hit += any(max(g['start'], p['start']) < min(g['end'], p['end']) for p in pred)
    return hit / gold if gold else None


def breakdown(rows, predictions, threshold, field):
    result = {}
    for value in sorted({r.get(field, 'legacy') for r in rows}):
        subset = [r for r in rows if r.get(field, 'legacy') == value]
        ids = {r['id'] for r in subset}
        result[value] = measures(gold_keys(subset), predicted_keys(subset, {k: v for k, v in predictions.items() if k in ids}, threshold))
    return result


def errors(rows, predictions, threshold):
    result = []
    for r in rows:
        p = predictions[r['id']]
        gold = {(s['start'], s['end']) for s in r['spans']}
        pred = {(s['start'], s['end']) for s in p['spans'] if s['score'] >= threshold}
        if gold == pred and p['error'] is None:
            continue
        result.append(dict(id=r['id'], family=r['family_id'], source=r['source'], kind=r['kind'], error=p['error'],
            missing=[dict(start=a, end=b, overlapping_prediction=any(max(a, c) < min(b, d) for c, d in pred)) for a, b in sorted(gold - pred)],
            false_positive=[dict(start=a, end=b, overlapping_gold=any(max(a, c) < min(b, d) for c, d in gold)) for a, b in sorted(pred - gold)]))
    return result


def v9_predictions(rows, comparison_dataset, checkpoint='candidate'):
    """Rebuild v9's thresholded exact predictions from its saved comparison report.

    The report stores, per record, gold spans the detector missed and its false
    positives; every other gold span was predicted exactly. Counts are cross-checked
    against the stored aggregate so a mismatched dataset cannot pass silently.
    """
    entry = comparison_dataset[checkpoint]
    by_id = {e['id']: e for e in entry['errors']}
    known = {r['id'] for r in rows}
    if not set(by_id) <= known:
        raise ValueError('v9 report refers to records outside this dataset')
    predicted = set()
    for r in rows:
        gold = {(s['start'], s['end']) for s in r['spans']}
        e = by_id.get(r['id'])
        missing = {(m['start'], m['end']) for m in e['missing']} if e else set()
        fps = {(f['start'], f['end']) for f in e['false_positive']} if e else set()
        if not missing <= gold:
            raise ValueError('v9 report does not match dataset gold spans')
        predicted |= {(r['id'], a, b) for a, b in (gold - missing) | fps}
    observed = measures(gold_keys(rows), predicted)
    stored = entry['ml_only']
    if (observed['tp'], observed['fp'], observed['fn']) != (stored['tp'], stored['fp'], stored['fn']):
        raise ValueError('rebuilt v9 predictions disagree with stored counts')
    return predicted


def compare_with_v9(rows, predictions, threshold, v9):
    gold = gold_keys(rows)
    qwen = predicted_keys(rows, predictions, threshold)
    return dict(
        v9=measures(gold, v9), qwen=measures(gold, qwen), ml_union_v9_or_qwen=measures(gold, v9 | qwen),
        both_tp=len(gold & qwen & v9), qwen_only_tp=len((gold & qwen) - v9), v9_only_tp=len((gold & v9) - qwen),
        neither_found=len(gold - qwen - v9), qwen_new_fp=len((qwen - gold) - v9), v9_fp_avoided_by_qwen=len((v9 - gold) - qwen),
        shared_fp=len((qwen & v9) - gold))


def select_threshold(rows, predictions, minimum_precision=None):
    """Validation-only: maximize exact F2 over observed score cutoffs (precision tie-break).

    With `minimum_precision`, only cutoffs meeting it are eligible; if none do, the
    best-F2 cutoff is returned with eligible=False so the failure is visible.
    """
    check_coverage(rows, predictions)
    gold = gold_keys(rows)
    cutoffs = sorted({0.0} | {s['score'] for p in predictions.values() for s in p['spans']})
    table = []
    for t in cutoffs:
        m = measures(gold, predicted_keys(rows, predictions, t))
        table.append(dict(threshold=t, **m))
    def key(m): return (m['f2'] or 0, m['precision'] or 0, m['threshold'])
    eligible = [m for m in table if minimum_precision is None or (m['precision'] or 0) >= minimum_precision]
    best = max(eligible or table, key=key)
    return dict(selected=best, eligible=bool(eligible), minimum_precision=minimum_precision, cutoffs_evaluated=len(table))


def read_predictions(path):
    result = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            record = json.loads(line)
            if record['id'] in result:
                raise ValueError('duplicate prediction id')
            result[record['id']] = record
    return result
