import pytest
from privai_model.ml_span.schema import load
from privai_model.qwen_span import data as qdata
from privai_model.qwen_span.postfilter import filter_spans, non_value_rule, lookup_argument

TRAIN_FILES = ['data/span-context-v9/train.jsonl', 'data/span-qwen-augment-v1/train.jsonl', 'data/span-qwen-augment-v2/train.jsonl']


@pytest.mark.parametrize('value', ['${API_KEY}', '$TOKEN', '${DB_PASS:?missing}', '%SECRET%', '$env:KEY', '#{TOKEN}',
                                   '${{ secrets.DEPLOY }}', '{{ .Values.secret }}', '<%= ENV["K"] %>', '$(cat /run/secrets/x)',
                                   'os.environ["K"]', 'keyring.get_password("app", "x")', 'client.access_secret_version(name="x")', 'os.getenv("K")', 'process.env.K', 'std::env::var("K")?',
                                   'vault://kv/app/x', 'arn:aws:secretsmanager:eu-west-1:0:secret:x', 'op://Eng/x/password',
                                   '********', '[REDACTED]', '[FILTERED]', '<hidden>', '••••', 'xxxxxxxx', '""', 'null',
                                   '<YOUR_API_KEY>', 'REPLACE_ME', 'changeme', 'your-api-key-here'])
def test_non_value_syntax_is_dropped(value):
    assert non_value_rule(value) is not None


@pytest.mark.parametrize('value', ['ghp_9fQ2mLr7TbW4kPa8sVn3cYh6dJx1', 'p@ss$word{2024}', 'AKIAZ7Q3EXAMPLE9KEY', 'a${b}c9X8',
                                   'sk-live-4f9a8b7c6d5e', 'Pa55(word)', 'f(x)9Lk2', 'Zx9fQ2mLr7TbW4kPa8sVn3cYh6dJ=='])
def test_literal_values_are_kept(value):
    assert non_value_rule(value) is None


def test_lookup_argument_needs_call_and_name_shape():
    text = 'auth = getSecret("worker-password")\nclient = Client("Zx9fQ2mLr7TbW4kPa8")'
    a = text.index('worker'); b = a + len('worker-password')
    c = text.index('Zx9'); d = c + len('Zx9fQ2mLr7TbW4kPa8')
    assert lookup_argument(text, a, b) and not lookup_argument(text, c, d)
    kept, dropped = filter_spans(text, [dict(start=a, end=b), dict(start=c, end=d)])
    assert kept == [dict(start=c, end=d)] and dropped == {'lookup_argument': 1}


def test_filter_never_drops_a_gold_span():
    paths = TRAIN_FILES + list(qdata.EVAL_SETS.values())
    for path in paths:
        for row in load(path):
            kept, dropped = filter_spans(row['text'], row['spans'])
            assert not dropped, (row['id'], dropped)
