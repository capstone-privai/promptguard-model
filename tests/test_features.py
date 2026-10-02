import math
import random
import string
import pytest
from privai_model.modeling.features import FeatureSpec, character_statistics, extract_features

VALUES = ['your-api-key-here','ghp_fake123456','AKIAIOSFODNN7EXAMPLE',
          '550e8400-e29b-41d4-a716-446655440000','a'*40,'0123456789abcdef'*8192,
          'password','abc123!',''.join(random.Random(3).choices(string.printable,k=512)),
          '', 'a','한글비밀번호🙂123 A\t', 'github_pat_fake','sk-fake']


@pytest.mark.parametrize('value',VALUES,ids=[f'edge-{i}' for i in range(len(VALUES))])
def test_edge_features(value):
    stats=character_statistics(value)
    features=extract_features(value)
    assert len(features)==FeatureSpec().dimension==14
    assert all(math.isfinite(v) and 0<=v<=1 for v in features)
    assert stats['length']==len(value) and stats['entropy']>=0
    names=('uppercase_ratio','lowercase_ratio','digit_ratio','special_ratio','whitespace_ratio','uncased_letter_ratio')
    assert sum(stats[k] for k in names)==pytest.approx(1 if value else 0)
    if not value:
        assert features==[0.]*14


def test_entropy_and_classes():
    assert character_statistics('aaaa')['entropy']==0
    assert character_statistics('abcd')['entropy']==2
    s=character_statistics('Aa1! 한')
    assert s['class_diversity']==6
    assert s['special_ratio']==pytest.approx(1/6)
    assert s['uncased_letter_ratio']==pytest.approx(1/6)


@pytest.mark.parametrize('prefix',['sk-','ghp_','github_pat_','AKIA'])
def test_prefix(prefix):
    s=character_statistics(prefix+'SYNTHETIC')
    assert s['known_prefix']==1 and sum(s['prefix_matches'])==1
    assert character_statistics(' '+prefix)['known_prefix']==0


def test_feature_spec_roundtrip_and_custom():
    spec=FeatureSpec(('custom_',),128)
    assert FeatureSpec.from_dict(spec.to_dict())==spec
    assert len(extract_features('custom_a',spec))==11
    assert extract_features('x'*1000,spec)[0]==1
    data=spec.to_dict(); data['dimension']=999
    with pytest.raises(ValueError): FeatureSpec.from_dict(data)
    with pytest.raises(ValueError): FeatureSpec(('',))
