"""Model-independent ID-joined evaluation. Scores are P(MASK); hard labels are 0/1."""
import math
import numbers
from collections import defaultdict
from sklearn.metrics import average_precision_score, precision_recall_curve, auc


def align_scores(examples, predictions):
    truth = {e.candidate_id: e for e in examples}
    if len(truth) != len(examples) or any(e.label not in (0, 1) for e in examples):
        raise ValueError('unique labeled examples required')
    found = {}
    for row in predictions:
        cid = row.get('candidate_id')
        if not isinstance(cid, str) or cid not in truth or cid in found:
            raise ValueError('prediction ID missing, duplicate, or unexpected')
        keys = [k for k in ('score', 'p_mask', 'prediction') if k in row]
        if len(keys) != 1:
            raise ValueError('exactly one of score/p_mask/prediction required')
        value = row[keys[0]]
        if keys[0] == 'prediction':
            value = {'MASK': 1, 'KEEP': 0}.get(value, value)
            if value not in (0, 1):
                raise ValueError('prediction must be MASK/KEEP or 1/0')
        if not isinstance(value, numbers.Real) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('score must be finite and in [0,1]')
        found[cid] = float(value)
    if set(found) != set(truth):
        raise ValueError('prediction coverage does not match examples')
    return [e.label for e in examples], [found[e.candidate_id] for e in examples]


def choose_threshold(labels, scores, target):
    """Highest threshold attaining target recall; ties kept together (>=). O(n log n)."""
    if not 0 < target <= 1 or len(labels) != len(scores) or not labels:
        raise ValueError('invalid threshold selection inputs')
    if any(y not in (0, 1) for y in labels) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('invalid labels/scores')
    positives = sorted((s for y, s in zip(labels, scores) if y == 1), reverse=True)
    if not positives:
        raise ValueError('validation has no MASK examples; recall threshold undefined')
    index = max(0, math.ceil(target*len(positives))-1)
    return float(positives[index])


def metrics(labels, scores, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('invalid threshold')
    predictions = [s >= threshold for s in scores]
    tp = sum(y == 1 and p for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p for y, p in zip(labels, predictions))
    fn = sum(y == 1 and not p for y, p in zip(labels, predictions))
    tn = sum(y == 0 and not p for y, p in zip(labels, predictions))
    ap = pr_auc = None
    if sum(labels):
        ap = float(average_precision_score(labels, scores))
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = float(auc(recall, precision))
    return {'records': len(labels), 'threshold': threshold, 'recall': tp/(tp+fn) if tp+fn else None,
            'precision': tp/(tp+fp) if tp+fp else None, 'false_positive_count': fp,
            'false_negative_count': fn, 'true_positive_count': tp, 'true_negative_count': tn,
            'average_precision': ap, 'pr_auc_trapezoid': pr_auc}


def evaluate(examples, predictions, threshold):
    labels, scores = align_scores(examples, predictions)
    result = metrics(labels, scores, threshold)
    result['provisional_records'] = sum(e.provisional for e in examples)
    for field, keyfn in [('type', lambda e: e.type), ('channel', lambda e: e.source.get('channel') or 'UNKNOWN'),
                         ('dataset', lambda e: e.source.get('dataset') or 'UNKNOWN')]:
        groups = defaultdict(list)
        for i, ex in enumerate(examples):
            groups[str(keyfn(ex))].append(i)
        result['by_'+field] = {key: metrics([labels[i] for i in ids], [scores[i] for i in ids], threshold)
                              for key, ids in sorted(groups.items())}
    return result


def select_thresholds(validation_examples, validation_predictions, targets=(0.95, 0.99)):
    labels, scores = align_scores(validation_examples, validation_predictions)
    return {str(target): choose_threshold(labels, scores, target) for target in targets}


def validate_prediction_v1(rows, expected_ids):
    expected = list(expected_ids)
    if len(expected) != len(set(expected)):
        raise ValueError('duplicate expected candidate_id')
    expected_set = set(expected)
    seen = set()
    for row in rows:
        cid = row.get('candidate_id')
        if not isinstance(cid, str) or cid in seen or cid not in expected_set:
            raise ValueError('invalid prediction candidate_id coverage')
        seen.add(cid)
        if row.get('schema_version') != 'prediction-v1' or not row.get('model_version'):
            raise ValueError('invalid prediction schema/model version')
        p, t, latency = (row.get(k) for k in ('p_mask','threshold','latency_ms'))
        if any(not isinstance(x, numbers.Real) or not math.isfinite(x) for x in (p,t,latency)):
            raise ValueError('prediction values must be finite numbers')
        if not 0 <= p <= 1 or not 0 <= t <= 1 or latency < 0:
            raise ValueError('prediction values out of range')
        if row.get('action') != ('MASK' if p >= t else 'KEEP'):
            raise ValueError('prediction action mismatch')
    if seen != set(expected):
        raise ValueError('missing prediction candidate_id')
