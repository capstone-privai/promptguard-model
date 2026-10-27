import importlib.util
import json
from pathlib import Path
import pytest
import torch
from unittest.mock import patch
from privai_model.ml_span.schema import validate,validate_spans,group_keys
from privai_model.ml_span.generator import generate,render
from privai_model.ml_span.splitter import split,assert_disjoint
from privai_model.ml_span.bio import align,decode,windows,aggregate
from privai_model.ml_span.runtime import candidate_union,mask_intervals,MLSpanDetector
from privai_model.ml_span.evaluation import evaluate


def span(a,b,score=1.): return dict(start=a,end=b,secret_type='SECRET',detector='ml',score=score)

@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    import span_smoke
    root=tmp_path_factory.mktemp('tiny-span')
    return span_smoke.tiny_model(root),root

@pytest.mark.parametrize('text,ranges',[
 ('암호는 가상!123입니다.',[(4,10)]),('Password: synthetic!',[(10,20)]),
 ('혼합 secret=Ab9!가상',[(10,16)]),('a=one; b=two',[(2,5),(9,12)]),
 ('abc abc',[(0,3),(4,7)]),('🔒 비밀\r\n가상é!',[(6,11)]),
 ('private\nabc\ndef\nend',[(8,15)]),('',[]),('no secret',[])])
def test_unicode_roundtrip(tiny,text,ranges):
    tokenizer,_=tiny
    gold=[span(a,b) for a,b in ranges]
    offsets=[(i,i+1) for i in range(len(text))]
    labels=align(text,offsets,gold)
    result=decode(text,offsets,labels)
    assert [(s['start'],s['end']) for s in result]==ranges
    for s in result: assert text[s['start']:s['end']]


def test_invalid_bio_and_empty():
    assert decode('abc',[(0,1),(1,2),(2,3)],[2,2,0])==[span(0,2)]
    assert decode('abc',[(0,1)],[0])==[]
    with pytest.raises(ValueError): decode('abc',[(0,1)],[4])
    with pytest.raises(ValueError): validate_spans('x',[span(0,2)])
    with pytest.raises(ValueError): validate_spans('x',[span(0,1,float('nan'))],True)


def test_sliding_window_secret_crosses_boundary(tiny):
    tok,_=tiny;text='z'*40+'abcDEFghiJKLmnop'+'x'*80;gold=[span(40,56)]
    predictions=[]
    for w in windows(tok,text,32,12):
        labels=align(text,w['offsets'],gold,w['special'])
        probs=[[float(i==max(0,y)) for i in range(3)] for y in labels]
        predictions.append((w['offsets'],probs))
    result=aggregate(text,predictions)
    assert [(s['start'],s['end']) for s in result]==[(40,56)]
    assert len(result)==1
    with pytest.raises(ValueError): windows(tok,text,32,32)


def test_generator_and_leakage():
    rows=generate(2);splits,manifest=split(rows);assert_disjoint(splits)
    assert manifest['statistics']['group_overlap'] is False
    assert {r['family_id'] for r in rows if not r['spans']} >= {'negative_uuid','negative_commit','negative_hash','negative_placeholder','negative_documentation'}
    assert len(rows)==len({r['text'] for r in rows})
    for r in rows: validate(r)
    bad=dict(rows[0],spans=[span(-1,2)])
    with pytest.raises(ValueError): validate(bad)
    with pytest.raises(ValueError): split([])
    with pytest.raises(ValueError): assert_disjoint({'train':[rows[0]],'test':[rows[1]]})
    # Cross-family reused value transitively joins components even when IDs differ.
    a,b=rows[0],rows[2]
    clone=dict(b,id='cross-family',text=a['text'],spans=a['spans'],value_spans=a['value_spans'])
    ss,_=split(rows+[clone]);assert_disjoint(ss)


def test_union_and_metric_denominators():
    row=generate(1)[0];text=row['text'];g=row['spans'][0]
    ml=[dict(g,score=.9,detector='ml'),span(0,2,.8)]
    rule=[dict(g,detector='rule')]
    assert len(candidate_union(text,rule,ml,.5))==2
    result=evaluate([row],{row['id']:[]},{row['id']:ml},.5)
    assert result['rule_miss_recall']==1 and result['ml_added_true_positives']==1
    assert result['ml_induced_false_positives']==1
    assert evaluate([row],{row['id']:rule},{row['id']:ml})['rule_miss_recall'] is None
    assert mask_intervals('abcdef',[span(0,3),span(2,5)])==[{'start':0,'end':5}]
    with pytest.raises(ValueError): evaluate([row],{}, {})


def test_runtime_errors_and_reuse(tiny):
    from transformers import AutoModelForTokenClassification
    tok,path=tiny
    model=AutoModelForTokenClassification.from_pretrained(path)
    detector=MLSpanDetector(tokenizer=tok,model=model,max_length=32,stride=8)
    identity=id(detector.model)
    assert detector.scan_text('')==[]
    for s in detector.scan_text('password=AbC123!'): validate_spans('password=AbC123!',[s],True)
    assert id(detector.model)==identity
    with patch.object(model,'forward',side_effect=ValueError('RAW_SECRET_DO_NOT_LOG')):
        with pytest.raises(RuntimeError,match='inference failed') as exc: detector.scan_text('abc')
        assert 'RAW_SECRET' not in str(exc.value)
    with pytest.raises(RuntimeError,match='initialization failed'): MLSpanDetector('/missing-model')


def test_boundary_representation_is_reported(tiny):
    from privai_model.ml_span.training import encoded_rows
    tok,_=tiny;rows=generate(1)[:2]
    encoded,stats=encoded_rows(rows,tok,128,32)
    assert stats['boundary_or_window_unrepresentable']==0
    assert all(r['labels'][0]==-100 and r['labels'][-1]==-100 for r in encoded)


def test_real_mmbert_unicode_offsets():
    from transformers import AutoTokenizer
    path=Path('runs/backbones/mmbert-small')
    if not path.exists(): pytest.skip('downloaded real tokenizer unavailable')
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    text='🔒 접속 인증: AbCD!가상123\r\n다시: AbCD!가상123'
    value='AbCD!가상123';starts=[text.index(value),text.rindex(value)]
    gold=[span(a,a+len(value)) for a in starts]
    predictions=[]
    for w in windows(tok,text,24,8):
        labels=align(text,w['offsets'],gold,w['special'])
        predictions.append((w['offsets'],[[float(i==max(0,y)) for i in range(3)] for y in labels]))
    output=aggregate(text,predictions)
    assert [text[s['start']:s['end']] for s in output]==[value,value]


def test_rule_proxy_maps_unicode_and_crlf():
    pytest.importorskip('credsweeper')
    from privai_model.ml_span.rules import RuleSpanDetector
    text='한글 🔒\r\npassword="xY39!sQ72pR84"\r\n'
    findings=RuleSpanDetector().scan_text(text)
    assert any(text[s['start']:s['end']]=='xY39!sQ72pR84' for s in findings)


def test_span_smoke_end_to_end(tmp_path):
    pytest.importorskip('credsweeper')
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    import span_smoke
    result=span_smoke.run(tmp_path/'smoke')
    model_dir=tmp_path/'smoke/run/model'
    detector=MLSpanDetector(model_dir)
    text='암호: abcXYZ123!'
    assert detector.scan_text(text)==MLSpanDetector(model_dir).scan_text(text)
    expected_records = len((tmp_path/'smoke/data/test.jsonl').read_text().splitlines())
    assert result['records'] == expected_records > 0 and result['parameters'] > 0
    assert (model_dir/'model.safetensors').exists()


def test_saturated_float32_probability_rounding():
    output=aggregate('secret', [([(0,6)],[[1e-9,0.99999994,8e-8]])])
    assert len(output)==1 and 0<=output[0]['score']<=1
    with pytest.raises(ValueError): aggregate('x',[([(0,1)],[[0.,float('nan'),0.]])])


def test_exact_miss_distinguished_from_covered_and_partial():
    row=generate(1)[0];g=row['spans'][0]
    wide=[span(g['start']-1,g['end']+1)]
    result=evaluate([row],{row['id']:wide},{row['id']:[dict(g,score=1.)]})
    assert result['rule_miss_count']==1
    assert result['rule_no_overlap_miss_count']==0
    assert result['rule_coverage_miss_count']==0
    partial=[span(g['start'],g['end']-1)]
    result=evaluate([row],{row['id']:partial},{row['id']:[dict(g,score=1.)]})
    assert result['rule_coverage_miss_count']==1 and result['rule_no_overlap_miss_count']==0
