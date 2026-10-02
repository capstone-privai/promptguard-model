"""Deterministic group-only split; sorted IDs remove dependence on input order."""
from collections import Counter
import hashlib
import json
import math
import random

NAMES = ('train', 'validation', 'test')


def fingerprint(examples):
    digest = hashlib.sha256()
    for ex in sorted(examples, key=lambda x: x.candidate_id):
        # Hash content to detect changes; never write credential contents to manifests.
        payload = vars(ex)
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def statistics(splits):
    groups = {k: {e.group_id for e in v} for k, v in splits.items()}
    overlap = any(groups[a] & groups[b] for i, a in enumerate(NAMES) for b in NAMES[i+1:])
    return {'group_overlap': overlap, 'splits': {
        name: {'records': len(rows), 'groups': len(groups[name]),
               'labels': {'MASK': sum(e.label == 1 for e in rows), 'KEEP': sum(e.label == 0 for e in rows)},
               'types': dict(Counter(e.type for e in rows))}
        for name, rows in splits.items()}}


def split_groups(examples, seed=42, ratios=(0.8, 0.1, 0.1)):
    if len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1):
        raise ValueError('three positive split ratios summing to 1 required')
    if len({e.candidate_id for e in examples}) != len(examples):
        raise ValueError('duplicate candidate_id')
    groups = sorted({e.group_id for e in examples})
    if len(groups) < 3:
        raise ValueError('at least three groups required')
    random.Random(seed).shuffle(groups)
    n = len(groups)
    ntrain = min(n-2, max(1, int(n*ratios[0])))
    nval = min(n-ntrain-1, max(1, int(n*ratios[1])))
    assignment = {g: NAMES[0 if i < ntrain else 1 if i < ntrain+nval else 2] for i, g in enumerate(groups)}
    splits = {name: [] for name in NAMES}
    for ex in sorted(examples, key=lambda x: x.candidate_id):
        splits[assignment[ex.group_id]].append(ex)
    manifest = {'schema_version': 'split-v1', 'seed': seed, 'ratios': list(ratios),
                'dataset_fingerprint': fingerprint(examples), 'group_assignment': assignment,
                'candidate_ids': {k: [e.candidate_id for e in v] for k, v in splits.items()},
                'statistics': statistics(splits)}
    return splits, manifest


def restore_split(examples, manifest):
    if fingerprint(examples) != manifest['dataset_fingerprint']:
        raise ValueError('dataset does not match split manifest')
    by_id = {e.candidate_id: e for e in examples}
    ids = [cid for name in NAMES for cid in manifest['candidate_ids'][name]]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError('invalid manifest candidate coverage')
    splits = {k: [by_id[cid] for cid in manifest['candidate_ids'][k]] for k in NAMES}
    if statistics(splits)['group_overlap']:
        raise ValueError('manifest group overlap')
    return splits
