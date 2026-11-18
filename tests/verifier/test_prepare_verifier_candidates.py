"""scripts/prepare_verifier_candidates.py reads strength with the majority policy unless the environment says otherwise.

detector.json's candidate_policy "union" protects every candidate, so every candidate came out strong and the
verifier had no ambiguous data. A stand-in pgdetector reports the policy the script set before loading the detector.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FAKE_DETECTOR = '''
import os

class Seen(Exception):
    pass

def load_detector():
    raise Seen(os.environ.get('PG_DETECTOR_CANDIDATE_POLICY'))
'''


@pytest.mark.parametrize('given, expected', [(None, 'majority'), ('union', 'union')])
def test_strength_policy_defaults_to_majority_and_keeps_an_explicit_choice(tmp_path, monkeypatch, given, expected):
    package = tmp_path / 'demo' / 'src' / 'detector' / 'pgdetector'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text(FAKE_DETECTOR, encoding='utf-8')
    for name in ('PG_DETECTOR_CANDIDATE_POLICY', 'PG_DETECTOR_MODE'):
        monkeypatch.setenv(name, 'x')  # recorded, so teardown removes what the script sets
        monkeypatch.delenv(name)
    if given:
        monkeypatch.setenv('PG_DETECTOR_CANDIDATE_POLICY', given)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.delitem(sys.modules, 'pgdetector', raising=False)
    monkeypatch.setattr(sys, 'argv', ['prepare_verifier_candidates.py', '--detector', str(tmp_path / 'demo'),
                                      '--data', str(tmp_path / 'data'), '--out', str(tmp_path / 'out')])
    spec = importlib.util.spec_from_file_location('prepare_verifier_candidates',
                                                  ROOT / 'scripts' / 'prepare_verifier_candidates.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    with pytest.raises(Exception) as seen:
        script.main()
    assert type(seen.value).__name__ == 'Seen' and seen.value.args == (expected,)
    assert not (tmp_path / 'out').exists()
