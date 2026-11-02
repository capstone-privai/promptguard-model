"""Hold-out v11: an independent test written after the v10 test was looked at.

v10 selected base on validation, but warm was singled out for a second check after its
v10 test results were seen, so the v10 test is development from here on. This hold-out is
authored before warm, base or v9 is run on it: new contextual families, new non-value
reference forms, contexts and key names that appear in no earlier file. It shares no
leakage key (value_family aside) with any training or evaluation row in the repository.
"""
import importlib.util
import json
from collections import Counter
from pathlib import Path
from privai_model.io import write_json, write_jsonl
from privai_model.ml_span.context_data import generate_context
from privai_model.ml_span.schema import load, group_keys, digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/span-holdout-v11'

# Contextual pairs: the same value is a secret in one row and public in the other.
FAMILIES = [
 ('v11_hold_sql_update', 'test', 'code', 'en',
  "UPDATE service_accounts SET password = '«S»' WHERE request_id = '«P»';",
  "UPDATE service_accounts SET public_alias = '«S»' WHERE request_id = '«P»';"),
 ('v11_hold_ko_handoff', 'test', 'prompt', 'ko',
  '방금 발급받은 결제 API 비밀키 «S» 공유드려요, 외부 유출 금지입니다. 요청 번호 «P»',
  '방금 생성된 공개 주문 코드 «S» 공유드려요, 고객에게 보내도 됩니다. 요청 번호 «P»'),
 ('multiline_v11_hold_export', 'test', 'file_read', 'en',
  'Private key material copied from the HSM export:\n«S»\nchange request «P»',
  'Public test vector copied from the RFC appendix:\n«S»\nchange request «P»'),
 ('v11_hold_email', 'test', 'prompt', 'en',
  'Subject: staging access\n\nThe temporary admin password is «S». Rotate it after use. Ticket «P».',
  'Subject: staging access\n\nThe temporary public demo code is «S». Share it freely. Ticket «P».'),
 ('v11_hold_tsv', 'test', 'tool_output', 'en',
  'name\tsecret_value\ttrace\npayments\t«S»\t«P»',
  'name\tbuild_label\ttrace\npayments\t«S»\t«P»'),
 ('long_v11_hold_pod_log', 'test', 'log', 'en',
  'kubectl logs: injected bearer token for the worker service account: «S»\npod uid «P»',
  'kubectl logs: injected public image tag for the worker service account: «S»\npod uid «P»'),
 ('v11_hold_public_first_two', 'test', 'prompt', 'mixed',
  'Public invoice «P» is fine to share, but the webhook signing secret «S» and the backup signing secret «T» are not.',
  'Public invoice «P» is fine to share, and so are the webhook public id «S» and the backup public id «T».'),
 ('v11_hold_md_list', 'test', 'markdown', 'en',
  '- staging DB password: `«S»`\n- public build: `«P»`',
  '- staging DB public schema hash: `«S»`\n- public build: `«P»`'),
]

# Non-value reference forms absent from every earlier dataset (the Qwen and v10 forms included).
FORMS = {
    'deno_env': lambda N, n: f'Deno.env.get("{N}")',
    'elixir_env': lambda N, n: f'System.fetch_env!("{N}")',
    'perl_env': lambda N, n: f'$ENV{{{N}}}',
    'swift_env': lambda N, n: f'ProcessInfo.processInfo.environment["{N}"]',
    'gradle_property': lambda N, n: f'project.findProperty("{n}")',
    'argo_parameter': lambda N, n: f'{{{{workflow.parameters.{n}}}}}',
    'vault_agent': lambda N, n: f'{{{{ with secret "secret/data/app" }}}}{{{{ .Data.data.{n} }}}}{{{{ end }}}}',
    'kubectl_jsonpath': lambda N, n: f"$(kubectl get secret app -o jsonpath='{{.data.{n}}}')",
    'puppet_lookup': lambda N, n: f"lookup('app::{n}')",
    'chef_data_bag': lambda N, n: f"data_bag_item('secrets', '{n}')['value']",
}
CONTEXTS = [
    ('tsv_line', 'tool_output', 'en', '{k}\t{r}\ttrace={i}'),
    ('sql_insert', 'code', 'en', "INSERT INTO settings (name, value, trace) VALUES ('{k}', '{r}', '{i}');"),
    ('makefile', 'configuration', 'en', '{K} ?= {r}\nBUILD_ID := {i}'),
    ('ko_slack', 'prompt', 'ko', '[#배포] {k} 은 {r} 로 바꿔 두었어요 (작업 {i})'),
    ('email_note', 'prompt', 'en', 'Hi team,\nplease set {k} to {r} on staging.\nRef: {i}\nThanks'),
    ('curl_header', 'shell', 'en', 'curl -H "X-{K}: {r}" https://api.example.test/v1?run={i}'),
]
KEYS = [('PAYMENT_API_KEY', 'payment_api_key'), ('LDAP_BIND_PASSWORD', 'ldap_bind_password'),
        ('OAUTH_CLIENT_SECRET', 'oauth_client_secret'), ('SESSION_SECRET', 'session_secret')]


def qwen_builder():
    spec = importlib.util.spec_from_file_location('qaug2', ROOT / 'scripts/prepare_qwen_augment_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def composition(rows):
    return dict(records=len(rows), positive=sum(bool(r['spans']) for r in rows), negative=sum(not r['spans'] for r in rows),
                gold_spans=sum(len(r['spans']) for r in rows), families=len({r['family_id'] for r in rows}),
                sources=dict(Counter(r['source'] for r in rows)))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    existing = {str(p.relative_to(ROOT)): load(p) for p in sorted((ROOT / 'data').glob('span-*/*.jsonl'))}
    exclusions = json.loads((ROOT / 'reports/ml_span/public-audit-exclusions.json').read_text())
    pairs, generated = generate_context(24, 20261111, exclusions, families=FAMILIES, variable_multiline=True)
    qaug = qwen_builder()
    qaug.FORMS = FORMS
    used = {r['text'][s['start']:s['end']] for rows in existing.values() for r in rows for s in r['value_spans']}
    used |= {r['text'][s['start']:s['end']] for r in pairs['test'] for s in r['value_spans']}
    references = qaug.build('hold11ref', sorted(FORMS), CONTEXTS, KEYS, 20261112, per_cell=1, positives_per_context=10,
                            forbidden=used)
    test = pairs['test'] + references
    if len({r['id'] for r in test}) != len(test):
        raise ValueError('duplicate ids')
    earlier = {}
    for name, rows in existing.items():
        for r in rows:
            for key in group_keys(r):
                if key[0] != 'value_family':
                    earlier.setdefault(key, name)
    hits = Counter(earlier[k] for r in test for k in group_keys(r) if k in earlier)
    if hits:
        raise ValueError(f'hold-out shares keys with earlier files: {dict(hits)}')
    OUT.mkdir(parents=True)
    write_jsonl(OUT / 'test.jsonl', test)
    write_json(OUT / 'manifest.json', dict(
        checked_against=sorted(existing), test_contextual=generated, test_reference_forms=sorted(FORMS),
        composition=composition(test), fingerprint=digest(json.dumps(test, sort_keys=True)),
        notice=('Independent hold-out authored after the v10 test was seen and before any model was run on it. '
                'Shares no leakage key (value_family aside) with any earlier file. Same author as the training '
                'data, so it is not a general benchmark.')))
    print(json.dumps(composition(test), ensure_ascii=False))


if __name__ == '__main__':
    main()
