"""Qwen v3 remedy data: combinatorial reference/placeholder negatives with contrast positives.

v2 still emitted non-value syntax (`${{ secrets.X }}`, `${X:?msg}`, `[FILTERED]`, a lookup
key name) as secrets and flagged public IDs next to credential words. One-template-per-
family augmentation did not generalize, so this generator composes many reference forms
with many line contexts.

Reference forms are split by a fixed seed BEFORE any v3 result is seen: about a third of
the forms appear only in the hold-out, so the hold-out measures generalization to syntax
the model never saw. Contexts, key names and value seeds are also disjoint between the two
outputs, and both are checked against every existing set for shared keys.
"""
import random
import string
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from privai_model.io import write_jsonl, write_json  # noqa: E402
from privai_model.ml_span.schema import validate, load, group_keys  # noqa: E402
from privai_model.qwen_span import data as qdata  # noqa: E402

# A reference form turns an env-style NAME and a lowercase name into non-value syntax.
FORMS = {
    'sh_dollar': lambda N, n: f'${N}',
    'sh_brace': lambda N, n: f'${{{N}}}',
    'sh_default': lambda N, n: f'${{{N}:-}}',
    'sh_required': lambda N, n: f'${{{N}:?{n} is required}}',
    'sh_cat_secret': lambda N, n: f'$(cat /run/secrets/{n})',
    'win_percent': lambda N, n: f'%{N}%',
    'pwsh_env': lambda N, n: f'$env:{N}',
    'py_environ': lambda N, n: f'os.environ["{N}"]',
    'py_getenv': lambda N, n: f'os.getenv("{N}")',
    'node_env': lambda N, n: f'process.env.{N}',
    'java_getenv': lambda N, n: f'System.getenv("{N}")',
    'ruby_fetch': lambda N, n: f'ENV.fetch("{N}")',
    'rust_var': lambda N, n: f'std::env::var("{N}")?',
    'go_getenv': lambda N, n: f'os.Getenv("{N}")',
    'dotnet_config': lambda N, n: f'Configuration["Secrets:{N}"]',
    'get_secret_call': lambda N, n: f'get_secret("{n}")',
    'secret_manager_call': lambda N, n: f'client.access_secret_version(name="{n}")',
    'keyring_call': lambda N, n: f'keyring.get_password("app", "{n}")',
    'gha_secret': lambda N, n: f'${{{{ secrets.{N} }}}}',
    'jinja_secret': lambda N, n: f'{{{{ secrets.{n} }}}}',
    'helm_values': lambda N, n: f'{{{{ .Values.auth.{n} }}}}',
    'erb_env': lambda N, n: f'<%= ENV["{N}"] %>',
    'octopus_var': lambda N, n: f'#{{{N}}}',
    'vault_uri': lambda N, n: f'vault://kv/app/{n}',
    'aws_arn': lambda N, n: f'arn:aws:secretsmanager:eu-west-1:000000000000:secret:app/{n}',
    'gcp_secret_path': lambda N, n: f'projects/demo/secrets/{n}/versions/latest',
    'onepassword_ref': lambda N, n: f'op://Engineering/{n}/password',
    'azure_kv_ref': lambda N, n: f'@Microsoft.KeyVault(SecretUri=https://demo.vault.azure.net/secrets/{n}/)',
    'angle_placeholder': lambda N, n: f'<YOUR_{N}>',
    'replace_me': lambda N, n: 'REPLACE_ME',
    'changeme': lambda N, n: 'changeme',
    'your_key_here': lambda N, n: f'your-{n}-here',
    'x_mask': lambda N, n: 'xxxxxxxxxxxxxxxx',
    'star_mask': lambda N, n: '****************',
    'redacted': lambda N, n: '[REDACTED]',
    'filtered': lambda N, n: '[FILTERED]',
    'hidden': lambda N, n: '<hidden>',
    'bullet_mask': lambda N, n: '••••••••',
    'empty_quotes': lambda N, n: '""',
    'null_literal': lambda N, n: 'null',
}

# Line contexts: {k} key, {r} right-hand side (reference or literal), {i} public id value.
TRAIN_CONTEXTS = [
    ('env_file', 'configuration', 'en', '{K}={r}\nREQUEST_ID={i}'),
    ('py_assign', 'code', 'en', '{k} = {r}  # trace {i}'),
    ('yaml_map', 'configuration', 'en', 'auth:\n  {k}: {r}\n  build: {i}'),
    ('json_obj', 'tool_output', 'en', '{{"{k}": "{r}", "request_id": "{i}"}}'),
    ('shell_export', 'shell', 'en', 'export {K}="{r}"  # job {i}'),
    ('log_line', 'log', 'en', 'INFO configured {k}={r} correlation={i}'),
    ('ko_prose', 'prompt', 'ko', '{k} 값은 {r} 로 설정되어 있고 공개 요청 번호는 {i} 입니다.'),
    ('en_prose', 'prompt', 'en', 'The {k} is set to {r}; the public ticket is {i}.'),
]
HOLDOUT_CONTEXTS = [
    ('toml_table', 'configuration', 'en', '[auth]\n{k} = "{r}"\nrevision = "{i}"'),
    ('ts_const', 'code', 'en', 'const {k} = {r};\nconst spanId = "{i}";'),
    ('dockerfile_env', 'configuration', 'en', 'ENV {K}={r}\nLABEL build={i}'),
    ('md_table', 'markdown', 'en', '| setting | value |\n|---|---|\n| {k} | {r} |\n| trace | {i} |'),
    ('ko_chat', 'prompt', 'ko', '설정 파일에서 {k} 는 {r} 이고, 배포 추적 ID는 {i} 이에요.'),
    ('cli_flag', 'shell', 'en', 'deploy --{k} {r} --run-id {i}'),
]
TRAIN_KEYS = [('API_KEY', 'api_key'), ('DB_PASSWORD', 'db_password'), ('CLIENT_SECRET', 'client_secret'),
              ('AUTH_TOKEN', 'auth_token'), ('SMTP_PASSWORD', 'smtp_password'), ('WEBHOOK_SECRET', 'webhook_secret')]
HOLDOUT_KEYS = [('SERVICE_TOKEN', 'service_token'), ('REDIS_PASSWORD', 'redis_password'), ('SIGNING_KEY', 'signing_key'),
                ('ACCESS_SECRET', 'access_secret')]


def random_value(rng):
    style = rng.randrange(4)
    if style == 0:
        return ''.join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(24, 40)))
    if style == 1:
        return ''.join(rng.choices(string.ascii_letters + string.digits + '-_', k=rng.randint(28, 44)))
    if style == 2:
        return ''.join(rng.choices(string.ascii_letters + string.digits + '+/', k=rng.randrange(24, 40, 4))) + '=='
    return ''.join(rng.choices('0123456789abcdef', k=rng.choice([32, 40])))


def render(template, fields):
    """Fill {K} {k} {r} {i}; return text and offsets of {r} and {i}."""
    text, where, i = '', {}, 0
    while i < len(template):
        if template.startswith('{{', i):
            text += '{'; i += 2; continue
        if template.startswith('}}', i):
            text += '}'; i += 2; continue
        if template[i] == '{':
            j = template.index('}', i)
            name = template[i + 1:j]
            where[name] = (len(text), len(text) + len(fields[name]))
            text += fields[name]
            i = j + 1
        else:
            text += template[i]; i += 1
    return text, where


def row(prefix, family, source, language, kind, text, gold, registered, index):
    spans = [dict(start=a, end=b, secret_type='SECRET') for a, b in gold]
    values = [dict(start=a, end=b, secret_type='SECRET') for a, b in sorted(registered)]
    r = dict(schema_version='secret-span-v1', synthetic=True, id=f'{prefix}:{family}:{index}', text=text, spans=spans,
             value_spans=values, source=source, language=language, family_id=f'{prefix}_{family}',
             template_id=f'{prefix}_{family}', origin=f'{prefix}:{family}', value_family=f'{prefix}_random', kind=kind)
    validate(r)
    return r


def build(prefix, forms, contexts, keys, seed, per_cell, positives_per_context, forbidden=frozenset()):
    rng = random.Random(seed)
    seen = set(forbidden)

    def fresh():
        while True:
            value = random_value(rng)
            if value not in seen:
                seen.add(value)
                return value
    rows = []
    for ctx, source, language, template in contexts:
        for form in forms:
            for n in range(per_cell):
                K, k = rng.choice(keys)
                public = fresh()
                text, where = render(template, dict(K=K, k=k, r=FORMS[form](K, k), i=public))
                # Only the public id is a registered value; the reference is not a value at all.
                rows.append(row(prefix, f'neg_{ctx}_{form}', source, language, 'hard_negative', text, [], [where['i']], n))
        for n in range(positives_per_context):
            K, k = rng.choice(keys)
            secret, public = fresh(), fresh()
            text, where = render(template, dict(K=K, k=k, r=secret, i=public))
            rows.append(row(prefix, f'pos_{ctx}', source, language, 'positive', text, [where['r']], [where['r'], where['i']], n))
    return rows


def shared(rows, others):
    keys = {}
    for name, other in others.items():
        for r in other:
            for key in group_keys(r):
                if key[0] != 'value_family':
                    keys.setdefault(key, name)
    hits = Counter()
    for r in rows:
        for key in group_keys(r):
            if key in keys:
                hits[f'{keys[key]}/{key[0]}'] += 1
    return dict(hits)


def composition(rows):
    return dict(records=len(rows), positive=sum(bool(r['spans']) for r in rows), negative=sum(not r['spans'] for r in rows),
                gold_spans=sum(len(r['spans']) for r in rows), families=len({r['family_id'] for r in rows}),
                sources=dict(Counter(r['source'] for r in rows)))


def main():
    augment_dir, holdout_dir = ROOT / 'data/span-qwen-augment-v2', ROOT / 'data/span-qwen-holdout-v2'
    if augment_dir.exists() or holdout_dir.exists():
        raise FileExistsError('outputs must be new')
    existing = {'v9_train': load(ROOT / qdata.TRAIN_DIR / 'train.jsonl'),
                'qwen_augment_v1': load(ROOT / 'data/span-qwen-augment-v1/train.jsonl')}
    existing.update({name: load(ROOT / path) for name, path in qdata.EVAL_SETS.items()})
    # Values already used anywhere are never drawn again, so value keys stay disjoint.
    used = {r['text'][s['start']:s['end']] for rows in existing.values() for r in rows for s in r['value_spans']}
    names = sorted(FORMS)
    random.Random(20261101).shuffle(names)
    holdout_forms, train_forms = sorted(names[:13]), sorted(names[13:])
    augment = build('qaug2', train_forms, TRAIN_CONTEXTS, TRAIN_KEYS, 20261102, per_cell=1, positives_per_context=27,
                    forbidden=used)
    holdout = build('qhold2', holdout_forms, HOLDOUT_CONTEXTS, HOLDOUT_KEYS, 20261103, per_cell=1, positives_per_context=13,
                    forbidden=used | {r['text'][s['start']:s['end']] for r in augment for s in r['value_spans']})
    checks = dict(augment_vs_existing=shared(augment, existing), holdout_vs_existing=shared(holdout, existing),
                  holdout_vs_augment=shared(holdout, {'augment_v2': augment}))
    if any(checks.values()):
        raise ValueError(f'shared keys found: {checks}')
    augment_dir.mkdir(parents=True)
    holdout_dir.mkdir(parents=True)
    write_jsonl(augment_dir / 'train.jsonl', augment)
    write_jsonl(holdout_dir / 'test.jsonl', holdout)
    split = dict(seed=20261101, train_forms=train_forms, holdout_only_forms=holdout_forms)
    write_json(augment_dir / 'manifest.json', dict(role='Qwen v3 train-only augmentation', form_split=split,
               composition=composition(augment), shared_key_checks=checks))
    write_json(holdout_dir / 'manifest.json', dict(role=('Independent hold-out for Qwen v3. Built before v3 training; never '
               'used for training, epoch or cutoff selection. Reference forms, contexts, key names and seeds are disjoint '
               'from the augmentation. Same author as the augmentation and targeted at the remedied failure type, so it '
               'is not a general benchmark.'), form_split=split, composition=composition(holdout), shared_key_checks=checks))
    print('holdout-only forms', holdout_forms)
    print('augment', composition(augment))
    print('holdout', composition(holdout))
    print('shared keys', checks)


if __name__ == '__main__':
    main()
