import importlib.util
import json
from pathlib import Path
import pytest
from privai_model.qwen_span.wrapper import DEFAULT_BUNDLE, QwenSecretScanner, ScanError, load_bundle

ROOT = Path(__file__).resolve().parents[2]


class FakeDetector:
    """Returns fixed spans for a value; mimics QwenSpanDetector.scan_detail's shape."""

    def __init__(self, spans=(), error=None, raises=False, tokens=10):
        self.spans, self.error, self.raises, self.tokens, self.calls = list(spans), error, raises, tokens, 0

    def prompt_length(self, text):
        return self.tokens

    def scan_detail(self, text):
        self.calls += 1
        if self.raises:
            raise RuntimeError('boom')
        return dict(spans=[] if self.error else self.spans, error=self.error, counts={'ok': len(self.spans)},
                    canonical=None, generated_tokens=5, latency_ms=1.0)


def span(a, b, score=0.99):
    return dict(start=a, end=b, secret_type='SECRET', detector='qwen', score=score)


def scanner(**kw):
    return QwenSecretScanner(detector=FakeDetector(**kw))


def test_bundle_is_published_with_validation_cutoff():
    meta = load_bundle(DEFAULT_BUNDLE)
    selection = json.loads((ROOT / 'reports/qwen_span/selection-v2.json').read_text())
    assert meta['threshold'] == selection['selected']['threshold']
    assert meta['revision'] and not meta['production_ready']
    assert (DEFAULT_BUNDLE / 'adapter_config.json').exists()


def test_v3_bundle_matches_its_selection():
    meta = load_bundle(ROOT / 'models/qwen-span-v3')
    selection = json.loads((ROOT / 'reports/qwen_span/selection-v3.json').read_text())
    assert meta['threshold'] == selection['selected']['threshold'] and not meta.get('postfilter')
    assert meta['revision'] == load_bundle(DEFAULT_BUNDLE)['revision'] and not meta['production_ready']


def test_scan_applies_bundle_threshold():
    s = scanner(spans=[span(0, 4, 0.999), span(6, 9, 0.5)])
    r = s.scan('abcd, xyz')
    assert r.status == 'ok' and [(f.start, f.end) for f in r.findings] == [(0, 4)] and not r.clean


def test_threshold_override_and_range():
    s = QwenSecretScanner(threshold=0.0, detector=FakeDetector(spans=[span(6, 9, 0.5)]))
    assert len(s.scan('abcd, xyz').findings) == 1
    with pytest.raises(ValueError):
        QwenSecretScanner(threshold=1.5, detector=FakeDetector())


def test_blank_text_skips_model():
    s = scanner()
    assert s.scan('  \n').clean and s.detector.calls == 0
    with pytest.raises(TypeError):
        s.scan(None)


@pytest.mark.parametrize('kw,error', [(dict(error='invalid_json'), 'invalid_json'), (dict(error='truncated'), 'truncated'),
                                      (dict(raises=True), 'inference_failed'), (dict(tokens=10**6), 'input_too_long')])
def test_failures_are_never_clean(kw, error):
    s = scanner(**kw)
    r = s.scan('some text')
    assert r.status == 'error' and r.error == error and not r.clean and r.findings == []
    m = s.mask('some text')
    assert m.status == 'blocked' and m.text is None
    with pytest.raises(ScanError):
        s.mask('some text', on_error='raise')


def test_mask_replaces_from_the_end_and_merges_overlaps():
    text = 'a=SECRET1 b=SECRET2'
    s = scanner(spans=[span(2, 9), span(12, 19), span(14, 19)])
    m = s.mask(text, placeholder='***')
    assert m.status == 'masked' and m.text == 'a=*** b=***' and len(m.findings) == 3


def test_mask_unicode_offsets_are_code_points():
    text = '토큰 🔑 abc'
    s = scanner(spans=[span(5, 8)])
    assert s.mask(text).text == '토큰 🔑 [SECRET]'


def test_bundle_postfilter_drops_non_value_syntax(tmp_path):
    meta = dict(load_bundle(DEFAULT_BUNDLE), postfilter='non-value-syntax-v1')
    (tmp_path / 'qwen_bundle.json').write_text(json.dumps(meta))
    text = 'TOKEN=${API_TOKEN} key=Zx9fQ2mLr7TbW4kPa8'
    s = QwenSecretScanner(tmp_path, detector=FakeDetector(spans=[span(6, 18), span(23, 41)]))
    r = s.scan(text)
    assert [(f.start, f.end) for f in r.findings] == [(23, 41)] and r.filtered == {'shell_variable': 1}
    with pytest.raises(ValueError):
        (tmp_path / 'qwen_bundle.json').write_text(json.dumps(dict(meta, postfilter='other')))
        QwenSecretScanner(tmp_path, detector=FakeDetector())


def test_clean_mask_returns_text_unchanged():
    assert scanner().mask('nothing here').to_dict() == dict(status='clean', text='nothing here', findings=[], error=None)


def load_cli():
    spec = importlib.util.spec_from_file_location('qwen_scan', ROOT / 'scripts/qwen_scan.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_jsonl_exit_codes_and_no_values(tmp_path, capsys):
    cli = load_cli()
    inputs = tmp_path / 'in.jsonl'
    inputs.write_text('\n'.join(json.dumps(dict(id=i, text=t)) for i, t in [('a', 'k=VALUE123'), ('b', '   ')]) + '\n')
    code = cli.main(['--jsonl', str(inputs)], scanner=scanner(spans=[span(2, 10)]))
    out = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert code == 1 and [o['id'] for o in out] == ['a', 'b']
    assert out[0]['findings'][0]['start'] == 2 and 'VALUE123' not in json.dumps(out)


def test_cli_mask_and_failure_exit_code(capsys):
    cli = load_cli()
    assert cli.main(['--text', 'k=VALUE123', '--mask'], scanner=scanner(spans=[span(2, 10)])) == 1
    assert json.loads(capsys.readouterr().out)['masked_text'] == 'k=[SECRET]'
    assert cli.main(['--text', 'x', '--mask'], scanner=scanner(error='invalid_json')) == 2
    out = json.loads(capsys.readouterr().out)
    assert out['mask_status'] == 'blocked' and out['masked_text'] is None
