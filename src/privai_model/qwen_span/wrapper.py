"""One entry point around the Qwen span detector: load once, check input, scan, mask.

Fail-closed: a malformed or truncated answer, an oversized input or a model error gives
status='error' and never a clean verdict, and mask() then blocks instead of handing back
text that may still hold a secret. Results carry offsets and scores, never values.
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

DEFAULT_BUNDLE = Path(__file__).resolve().parents[3] / 'models' / 'qwen-span-v2'
BUNDLE_SCHEMA = 'qwen-span-bundle-v1'
POSTFILTERS = (None, 'non-value-syntax-v1')


class ScanError(RuntimeError):
    """Raised by mask(on_error='raise') when the text could not be scanned reliably."""


@dataclass(frozen=True)
class Finding:
    start: int
    end: int
    score: float


@dataclass
class ScanResult:
    status: str                      # 'ok' or 'error'
    findings: list = field(default_factory=list)
    error: str = None                # invalid_json, truncated, input_too_long, inference_failed
    latency_ms: float = 0.0
    counts: dict = field(default_factory=dict)
    filtered: dict = field(default_factory=dict)   # post-filter rule -> spans dropped

    @property
    def clean(self):
        """True only when the scan succeeded and found nothing."""
        return self.status == 'ok' and not self.findings

    def to_dict(self):
        return asdict(self)


@dataclass
class MaskResult:
    status: str                      # 'clean', 'masked' or 'blocked'
    text: str = None                 # None when blocked
    findings: list = field(default_factory=list)
    error: str = None

    def to_dict(self):
        return asdict(self)


def load_bundle(path):
    meta = json.loads((Path(path) / 'qwen_bundle.json').read_text())
    if meta.get('schema_version') != BUNDLE_SCHEMA:
        raise ValueError(f'unsupported bundle schema in {path}')
    return meta


def merge(findings):
    """Overlapping or touching findings become one masked interval."""
    merged = []
    for f in sorted(findings, key=lambda f: (f.start, f.end)):
        if merged and f.start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], f.end)
        else:
            merged.append([f.start, f.end])
    return merged


class QwenSecretScanner:
    """Load the base model and adapter once, then call scan()/mask() per text.

    `detector` injects any object with scan_detail(text) and prompt_length(text); tests use
    it to run without torch.
    """

    def __init__(self, bundle=DEFAULT_BUNDLE, *, adapter=None, threshold=None, device='auto',
                 local_files_only=False, detector=None):
        self.meta = load_bundle(bundle)
        self.threshold = self.meta['threshold'] if threshold is None else float(threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError('threshold must be in [0, 1]')
        self.max_input_tokens = self.meta['max_input_tokens']
        self.postfilter = self.meta.get('postfilter')
        if self.postfilter not in POSTFILTERS:
            raise ValueError(f'unknown postfilter {self.postfilter!r}')
        if detector is None:
            from .infer import QwenSpanDetector
            detector = QwenSpanDetector(self.meta['base_model'], str(adapter or bundle), device, self.meta['dtype'],
                                        self.meta['revision'], local_files_only, self.meta['max_new_tokens'])
        self.detector = detector

    def scan(self, text):
        if not isinstance(text, str):
            raise TypeError('text must be a string')
        if not text.strip():
            return ScanResult('ok')
        if self.detector.prompt_length(text) > self.max_input_tokens:
            return ScanResult('error', error='input_too_long')
        try:
            detail = self.detector.scan_detail(text)
        except Exception:
            return ScanResult('error', error='inference_failed')
        if detail['error'] is not None:
            return ScanResult('error', error=detail['error'], latency_ms=detail['latency_ms'], counts=detail['counts'])
        spans, filtered = detail['spans'], {}
        if self.postfilter:
            from .postfilter import filter_spans
            spans, filtered = filter_spans(text, spans)
        findings = [Finding(s['start'], s['end'], s['score']) for s in spans if s['score'] >= self.threshold]
        return ScanResult('ok', findings, latency_ms=detail['latency_ms'], counts=detail['counts'], filtered=filtered)

    def scan_many(self, texts):
        return [self.scan(t) for t in texts]

    def mask(self, text, placeholder='[SECRET]', on_error='block'):
        if on_error not in ('block', 'raise'):
            raise ValueError("on_error must be 'block' or 'raise'")
        result = self.scan(text)
        if result.status != 'ok':
            if on_error == 'raise':
                raise ScanError(result.error)
            return MaskResult('blocked', None, [], result.error)
        if not result.findings:
            return MaskResult('clean', text)
        masked = text
        for start, end in reversed(merge(result.findings)):
            masked = masked[:start] + placeholder + masked[end:]
        return MaskResult('masked', masked, result.findings)
