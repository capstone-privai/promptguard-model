"""Context v10: retrain on every existing training split, evaluate on a new independent hold-out.

v9 warm-started v8 at learning rate 1e-6 on 2,056 rows (a bounded replay). The repository
holds 6,304 distinct training rows across all earlier splits plus the Qwen augmentations,
none sharing a leakage key with any evaluation set. v10 trains on all of them.

Every earlier test set has now been looked at, so the v10 test is authored here, before
any v10 training: new contextual families and non-value reference forms that appear in
no earlier file. Validation stays the v9 validation split so selection is comparable.
"""
import importlib.util
import json
from collections import Counter
from pathlib import Path
from privai_model.io import write_json, write_jsonl
from privai_model.ml_span.context_data import generate_context
from privai_model.ml_span.schema import load, group_keys, digest
from privai_model.ml_span.splitter import assert_disjoint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/span-context-v10'
TRAIN = ['span-synthetic-v1', 'span-refinement-v1', 'span-code-refinement-v1', 'span-refinement-v3',
         'span-context-v4', 'span-context-v5', 'span-context-v6', 'span-context-v7', 'span-context-v8',
         'span-context-v9', 'span-qwen-augment-v1', 'span-qwen-augment-v2']
EVALUATION = [f'{d}/{s}.jsonl' for d in ['span-synthetic-v1', 'span-refinement-v1', 'span-code-refinement-v1',
              'span-refinement-v3', 'span-context-v4', 'span-context-v5', 'span-context-v6', 'span-context-v7',
              'span-context-v8', 'span-context-v9'] for s in ['validation', 'test']] + [
              'span-negative-challenge-v1/test.jsonl', 'span-qwen-holdout-v1/test.jsonl', 'span-qwen-holdout-v2/test.jsonl']

# Contextual pairs: the same value is a secret in one row and public in the other.
FAMILIES = [
 ('v10_test_suffix_review', 'test', 'prompt', 'en',
  'Reviewed the deployment notes. The string «S» is the service login secret, keep it private. Public change number «P».',
  'Reviewed the deployment notes. The string «S» is the public changelog anchor, safe to share. Public change number «P».'),
 ('v10_test_ko_suffix', 'test', 'prompt', 'ko',
  '«S» 이 값은 운영 DB 접속 비밀번호라서 외부에 공유하면 안 됩니다. 공개 티켓 «P».',
  '«S» 이 값은 공개 릴리스 태그라서 외부에 공유해도 됩니다. 공개 티켓 «P».'),
 ('multiline_v10_test_paste', 'test', 'file_read', 'en',
  'Pasting what the vault UI showed for the signing material:\n«S»\n(do not commit). Public request: «P»',
  'Pasting what the docs page showed for the sample checksum list:\n«S»\n(fine to commit). Public request: «P»'),
 ('v10_test_chat_two', 'test', 'prompt', 'en',
  'old api secret was «S», the rotated api secret is «T»; ticket «P» tracks it',
  'old public build id was «S», the new public build id is «T»; ticket «P» tracks it'),
 ('v10_test_csv', 'test', 'tool_output', 'en',
  'service,credential,trace\nbilling,«S»,«P»',
  'service,public_reference,trace\nbilling,«S»,«P»'),
 ('long_v10_test_ci_log', 'test', 'log', 'en',
  '##[group]Run deploy\nauth secret resolved for registry login: «S»\npublic run id «P»\n##[endgroup]',
  '##[group]Run deploy\npublic registry image digest label: «S»\npublic run id «P»\n##[endgroup]'),
 ('v10_test_decoy_first', 'test', 'prompt', 'mixed',
  'Public order number «P» was attached; the merchant API secret is «S».',
  'Public order number «P» was attached; the merchant public catalog id is «S».'),
 ('v10_test_xml', 'test', 'configuration', 'en',
  '<credentials>\n  <publicId>«P»</publicId>\n  <secretKey>«S»</secretKey>\n</credentials>',
  '<catalog>\n  <publicId>«P»</publicId>\n  <catalogKey>«S»</catalogKey>\n</catalog>'),
]

# Non-value reference forms absent from every earlier dataset (the 40 Qwen forms included).
FORMS = {
    'terraform_var': lambda N, n: f'var.{n}',
    'cfn_ref': lambda N, n: f'!Ref {N}',
    'cfn_dynamic': lambda N, n: f'{{{{resolve:secretsmanager:app/{n}:SecretString}}}}',
    'serverless_ssm': lambda N, n: f'${{ssm:/app/{n}}}',
    'concourse_var': lambda N, n: f'(({n}))',
    'configparser': lambda N, n: f'%({n})s',
    'laravel_env': lambda N, n: f"env('{N}')",
    'docker_secret_file': lambda N, n: f'/run/secrets/{n}',
    'pulumi_config': lambda N, n: f'config.require_secret("{n}")',
    'ansible_lookup': lambda N, n: f"{{{{ lookup('hashi_vault', 'secret=kv/{n}') }}}}",
}
CONTEXTS = [
    ('hcl_block', 'configuration', 'en', 'resource "db" "main" {{\n  {k} = {r}\n  tags = {{ trace = "{i}" }}\n}}'),
    ('ini_section', 'configuration', 'en', '[database]\n{k} = {r}\nrelease = {i}'),
    ('php_array', 'code', 'en', "'{k}' => {r}, 'request' => '{i}',"),
    ('chat_note', 'prompt', 'en', 'heads up: set {k} to {r} before the deploy (run {i})'),
    ('ko_review', 'prompt', 'ko', '리뷰 코멘트: {k} 항목이 {r} 로 되어 있습니다. 배포 번호 {i}.'),
    ('yaml_list', 'configuration', 'en', '- name: {K}\n  value: {r}\n- name: TRACE\n  value: {i}'),
]
KEYS = [('BROKER_PASSWORD', 'broker_password'), ('LICENSE_KEY', 'license_key'), ('MASTER_TOKEN', 'master_token'),
        ('ENCRYPTION_SECRET', 'encryption_secret')]


def qwen_builder():
    spec = importlib.util.spec_from_file_location('qaug2', ROOT / 'scripts/prepare_qwen_augment_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def union_train():
    by_text = {}
    for name in TRAIN:
        for r in load(ROOT / 'data' / name / 'train.jsonl'):
            first = by_text.setdefault(r['text'], r)
            if sorted((s['start'], s['end']) for s in first['spans']) != sorted((s['start'], s['end']) for s in r['spans']):
                raise ValueError('same training text with different gold spans')
    return list(by_text.values())


def composition(rows):
    return dict(records=len(rows), positive=sum(bool(r['spans']) for r in rows), negative=sum(not r['spans'] for r in rows),
                gold_spans=sum(len(r['spans']) for r in rows), families=len({r['family_id'] for r in rows}),
                sources=dict(Counter(r['source'] for r in rows)))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    train = union_train()
    validation = load(ROOT / 'data/span-context-v9/validation.jsonl')
    evaluation = {path: load(ROOT / 'data' / path) for path in EVALUATION}
    exclusions = json.loads((ROOT / 'reports/ml_span/public-audit-exclusions.json').read_text())
    pairs, generated = generate_context(24, 20261110, exclusions, families=FAMILIES, variable_multiline=True)
    qaug = qwen_builder()
    qaug.FORMS = FORMS
    used = {r['text'][s['start']:s['end']] for rows in [train, *evaluation.values()] for r in rows for s in r['value_spans']}
    used |= {r['text'][s['start']:s['end']] for r in pairs['test'] for s in r['value_spans']}
    references = qaug.build('ctx10ref', sorted(FORMS), CONTEXTS, KEYS, 20261111, per_cell=1, positives_per_context=10,
                            forbidden=used)
    test = pairs['test'] + references
    assert_disjoint({'train': train, 'validation': validation, 'test': test})
    assert_disjoint({'train': train, 'historical_eval': [r for rows in evaluation.values() for r in rows]})
    # The test must share nothing with any earlier evaluation set either (value_family aside).
    earlier = {k for rows in evaluation.values() for r in rows for k in group_keys(r) if k[0] != 'value_family'}
    if any(k in earlier for r in test for k in group_keys(r) if k[0] != 'value_family'):
        raise ValueError('v10 test shares a key with an earlier evaluation set')
    OUT.mkdir(parents=True)
    splits = dict(train=train, validation=validation, test=test)
    for name, rows in splits.items():
        write_jsonl(OUT / f'{name}.jsonl', rows)
    write_json(OUT / 'manifest.json', dict(
        train_sources={name: len(load(ROOT / 'data' / name / 'train.jsonl')) for name in TRAIN},
        validation_source='data/span-context-v9/validation.jsonl',
        test_contextual=generated, test_reference_forms=sorted(FORMS),
        composition={name: composition(rows) for name, rows in splits.items()},
        fingerprint=digest(json.dumps(splits, sort_keys=True)),
        notice=('Train is the de-duplicated union of every earlier training split. Validation is unchanged from v9. '
                'Test is new, authored before v10 training, and shares no leakage key with any train or earlier '
                'evaluation row. Same author as the training data, so it is not a general benchmark.')))
    print(json.dumps({name: composition(rows) for name, rows in splits.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
