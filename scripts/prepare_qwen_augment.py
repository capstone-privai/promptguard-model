"""Qwen v2 remedy data: a train-only augmentation and a separate independent hold-out.

Written after the v1 adapter fired on every negative_challenge row (references such as
${SERVICE_TOKEN} or os.environ[...] taken as secrets) and picked the public decoy in the
validation review family. The two outputs use disjoint template wording, disjoint value
seeds and disjoint family IDs; both are checked against v9 train and every evaluation set
for shared group/text/template/value keys before anything is written.

The hold-out is never used for training, epoch or cutoff selection. Rows hold synthetic
random values only.
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

# «S» = secret value (gold span), «D» = public/non-secret value (registered, not gold).
# (family, source, language, template)
AUGMENT_NEGATIVE = [
    ('getenv_lookup', 'code', 'en', 'api_key = os.getenv("STRIPE_KEY")  # request «D»'),
    ('bearer_expansion', 'shell', 'en', 'curl -sH "Authorization: Bearer $GITHUB_TOKEN" -H "X-Correlation-Id: «D»" https://api.example.invalid/v1'),
    ('vault_ref', 'configuration', 'en', 'password_ref: vault:secret/data/app#password\njob_id: «D»'),
    ('ssm_lookup', 'code', 'en', 'token = ssm.get_parameter(Name="/app/token", WithDecryption=True)["Parameter"]["Value"]\nrun_id = "«D»"'),
    ('helm_value', 'configuration', 'en', 'client_secret: "{{ .Values.clientSecret }}"\nrelease_tag: «D»'),
    ('ts_schema', 'code', 'en', 'type Credentials = { apiKey: string; secret: string };  // schema rev «D»'),
    ('star_masked', 'log', 'en', 'login ok user=svc password=*** token=<hidden> session_ref=«D»'),
    ('placeholder_angle', 'configuration', 'en', 'SLACK_BOT_TOKEN=<paste-your-token>\nBUILD_SHA=«D»'),
    ('ko_vault', 'prompt', 'ko', 'API 키는 배포 시 Vault에서 주입되므로 코드에 없습니다. 요청 ID는 «D» 입니다.'),
    ('dotnet_config', 'code', 'en', 'var secret = Configuration["Auth:ClientSecret"]; var correlation = "«D»";'),
]
AUGMENT_POSITIVE = [
    ('getenv_literal', 'code', 'en', 'api_key = os.getenv("STRIPE_KEY", "«S»")  # request «D»'),
    ('bearer_literal', 'shell', 'en', 'curl -sH "Authorization: Bearer «S»" -H "X-Correlation-Id: «D»" https://api.example.invalid/v1'),
    ('vault_literal', 'configuration', 'en', 'password: «S»  # TODO move to vault\njob_id: «D»'),
    ('ko_literal', 'prompt', 'ko', '테스트용이라며 API 키 «S» 를 그대로 붙여 넣었습니다. 요청 ID는 «D» 입니다.'),
    ('dotnet_literal', 'code', 'en', 'var secret = "«S»"; var correlation = "«D»";'),
]
# Two random values, one secret and one public, in both orders.
AUGMENT_DECOY = [
    ('decoy_rotate', 'log', 'en', 'rotation needed for leaked key «S»; public build number «D» is fine to share.'),
    ('decoy_rotate_rev', 'log', 'en', 'public build number «D» is fine to share; rotation needed for leaked key «S».'),
    ('decoy_ko', 'prompt', 'ko', '«D» 는 공개 요청 ID이고, 실제로 노출된 비밀 키는 «S» 입니다.'),
    ('decoy_ko_rev', 'prompt', 'ko', '실제로 노출된 비밀 키는 «S» 이고, «D» 는 공개 요청 ID입니다.'),
    ('decoy_table', 'markdown', 'en', '| field | value |\n|---|---|\n| trace (public) | «D» |\n| session secret | «S» |'),
]
AUGMENT_DECOY_NEGATIVE = [
    ('decoy_both_public', 'log', 'en', 'public build number «D» and public mirror id «D» are both safe to share.'),
    ('decoy_both_public_ko', 'prompt', 'ko', '«D» 와 «D» 는 둘 다 공개 문서 번호입니다.'),
]

HOLDOUT_NEGATIVE = [
    ('node_env', 'code', 'en', 'const key = process.env.PAYMENT_API_KEY;\nconst traceId = "«D»";'),
    ('aws_cli_subst', 'shell', 'en', 'export DB_PASSWORD="$(aws secretsmanager get-secret-value --secret-id prod/db --query SecretString)"  # run «D»'),
    ('gha_secret', 'configuration', 'en', 'env:\n  DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}\n  RUN_REF: «D»'),
    ('k8s_ref', 'configuration', 'en', 'valueFrom:\n  secretKeyRef:\n    name: api-creds\n    key: token\n# rollout «D»'),
    ('compose_default', 'configuration', 'en', 'POSTGRES_PASSWORD: ${DB_PASS:?missing}\nIMAGE_DIGEST_SHORT: «D»'),
    ('keyring_call', 'code', 'en', 'token = keyring.get_password("svc", "bot")  # ticket «D»'),
    ('ko_envvar', 'prompt', 'ko', '토큰은 환경변수 GH_TOKEN 에서 읽습니다. 배포 번호: «D»'),
    ('redacted_json', 'tool_output', 'en', '{"password": "[FILTERED]", "api_key": "••••••", "request": "«D»"}'),
]
HOLDOUT_POSITIVE = [
    ('node_literal', 'code', 'en', 'const key = "«S»"; // FIXME read from process.env\nconst traceId = "«D»";'),
    ('compose_literal', 'configuration', 'en', 'POSTGRES_PASSWORD: «S»\nIMAGE_DIGEST_SHORT: «D»'),
    ('ko_pasted', 'prompt', 'ko', '이 토큰 «S» 로 호출하면 401이 나요. 배포 번호: «D»'),
    ('json_literal', 'tool_output', 'en', '{"password": "«S»", "api_key": "••••••", "request": "«D»"}'),
]
HOLDOUT_DECOY = [
    ('decoy_incident', 'log', 'en', 'incident: credential «S» was pushed to a public repo; ticket «D» tracks it.'),
    ('decoy_incident_rev', 'log', 'en', 'ticket «D» tracks it: credential «S» was pushed to a public repo.'),
    ('decoy_ko_audit', 'prompt', 'ko', '감사 결과 공개 추적번호 «D» 와 함께 운영 접근 토큰 «S» 가 유출됐습니다.'),
    ('decoy_yaml', 'configuration', 'en', 'correlation_id: «D»  # public\napi_secret: «S»'),
]
HOLDOUT_DECOY_NEGATIVE = [
    ('decoy_public_pair', 'log', 'en', 'ticket «D» tracks it; public artifact «D» is attached.'),
    ('decoy_public_pair_ko', 'prompt', 'ko', '공개 추적번호 «D» 와 공개 빌드 번호 «D» 를 확인했습니다.'),
]


def random_value(rng):
    style = rng.randrange(4)
    if style == 0:
        return ''.join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(24, 40)))
    if style == 1:
        return ''.join(rng.choices(string.ascii_letters + string.digits + '-_', k=rng.randint(28, 44)))
    if style == 2:
        body = ''.join(rng.choices(string.ascii_letters + string.digits + '+/', k=rng.randrange(24, 40, 4)))
        return body + '=='
    return ''.join(rng.choices('0123456789abcdef', k=rng.choice([32, 40])))


def render(template, rng):
    text, spans, values = '', [], []
    parts = template.replace('«S»', '\0S\0').replace('«D»', '\0D\0').split('\0')
    for part in parts:
        if part in ('S', 'D'):
            value = random_value(rng)
            values.append(dict(start=len(text), end=len(text) + len(value), secret_type='SECRET'))
            if part == 'S':
                spans.append(values[-1])
            text += value
        else:
            text += part
    return text, spans, values


def build(prefix, origin, groups, per_family, seed):
    rng = random.Random(seed)
    rows = []
    for kind, templates in groups:
        for family, source, language, template in templates:
            for i in range(per_family):
                text, spans, values = render(template, rng)
                row = dict(schema_version='secret-span-v1', synthetic=True, id=f'{prefix}:{family}:{i}', text=text,
                           spans=spans, value_spans=values, source=source, language=language,
                           family_id=f'{prefix}_{family}', template_id=f'{prefix}_{family}',
                           origin=f'{origin}:{family}', value_family=f'{prefix}_random', kind=kind)
                validate(row)
                rows.append(row)
    return rows


def shared(rows, others):
    keys = {}
    for name, other in others.items():
        for row in other:
            for key in group_keys(row):
                if key[0] != 'value_family':
                    keys.setdefault(key, name)
    hits = Counter()
    for row in rows:
        for key in group_keys(row):
            if key in keys:
                hits[(keys[key], key[0])] += 1
    return {f'{a}/{b}': n for (a, b), n in hits.items()}


def composition(rows):
    return dict(records=len(rows), positive=sum(bool(r['spans']) for r in rows), negative=sum(not r['spans'] for r in rows),
                gold_spans=sum(len(r['spans']) for r in rows), families=len({r['family_id'] for r in rows}),
                kinds=dict(Counter(r['kind'] for r in rows)), sources=dict(Counter(r['source'] for r in rows)))


def main():
    augment_dir, holdout_dir = ROOT / 'data/span-qwen-augment-v1', ROOT / 'data/span-qwen-holdout-v1'
    if augment_dir.exists() or holdout_dir.exists():
        raise FileExistsError('outputs must be new')
    augment = build('qaug', 'qwen-augment-v1', [('hard_negative', AUGMENT_NEGATIVE), ('positive', AUGMENT_POSITIVE),
                    ('positive', AUGMENT_DECOY), ('hard_negative', AUGMENT_DECOY_NEGATIVE)], 12, 20261030)
    holdout = build('qhold', 'qwen-holdout-v1', [('hard_negative', HOLDOUT_NEGATIVE), ('positive', HOLDOUT_POSITIVE),
                    ('positive', HOLDOUT_DECOY), ('hard_negative', HOLDOUT_DECOY_NEGATIVE)], 8, 20261031)
    existing = {'v9_train': load(ROOT / qdata.TRAIN_DIR / 'train.jsonl')}
    existing.update({name: load(ROOT / path) for name, path in qdata.EVAL_SETS.items()})
    checks = dict(augment_vs_existing=shared(augment, existing),
                  holdout_vs_existing=shared(holdout, existing),
                  holdout_vs_augment=shared(holdout, {'augment': augment}))
    # value_family is a shared label within each output by design; every other key must be disjoint.
    if any(checks.values()):
        raise ValueError(f'shared keys found: {checks}')
    augment_dir.mkdir(parents=True)
    holdout_dir.mkdir(parents=True)
    write_jsonl(augment_dir / 'train.jsonl', augment)
    write_jsonl(holdout_dir / 'test.jsonl', holdout)
    note = ('Designed after inspecting v1 failures on validation (review decoy family) and negative_challenge '
            '(reference-not-value). Template wording, family IDs and value seeds are disjoint from the hold-out '
            'and from all existing sets (checked: group/text/canonical-template/value keys).')
    write_json(augment_dir / 'manifest.json', dict(role='Qwen v2 train-only augmentation', seed=20261030, per_family=12,
               note=note, composition=composition(augment), shared_key_checks=checks))
    write_json(holdout_dir / 'manifest.json', dict(role=('Independent hold-out for the Qwen v2 remedy. Never used for '
               'training, epoch or cutoff selection. Written by the same author as the augmentation, with different '
               'templates and seeds; it targets the remedied failure types, so it is not a general benchmark.'),
               seed=20261031, per_family=8, composition=composition(holdout), shared_key_checks=checks))
    print('augment', composition(augment))
    print('holdout', composition(holdout))
    print('shared keys', checks)


if __name__ == '__main__':
    main()
