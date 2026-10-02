import copy
from types import SimpleNamespace
import pytest
from privai_model.baselines.credsweeper import CredSweeperBaseline,covers_span
from privai_model.data.synthetic import hard_cases
from privai_model.data.adapter import adapt_records
from privai_model.evaluation.evaluator import align_scores,evaluate


def test_span_mapping_does_not_use_other_candidate_same_line():
    finding=SimpleNamespace(line_num=2,value_start=10,value_end=20)
    assert covers_span(finding,[0,30],40,50)
    assert not covers_span(finding,[0,30],51,60)
    assert not covers_span(finding,[0,30],39,50)
    assert not covers_span(SimpleNamespace(line_num=0,value_start=1,value_end=9),[0],1,9)


def test_real_credsweeper_and_common_evaluator():
    pytest.importorskip('credsweeper')
    scanner=CredSweeperBaseline(use_ml=True)
    records=hard_cases()
    predictions=scanner.predict(records)
    assert len(predictions)==len(records)
    assert {r['candidate_id'] for r in predictions}=={r['candidate_id'] for r in records}
    assert {r['prediction'] for r in predictions}=={'MASK','KEEP'}
    y,s=align_scores(adapt_records(records),list(reversed(predictions)))
    assert len(y)==len(s)==18
    assert evaluate(adapt_records(records),predictions,.5)['records']==18
    assert scanner.predict([])==[]
    with pytest.raises(ValueError): scanner.predict([records[0],records[0]])
    assert scanner.predict(records[:2])==predictions[:2] # no finding state leakage


def test_baseline_does_not_leak_detected_sibling():
    pytest.importorskip('credsweeper')
    scanner=CredSweeperBaseline(use_ml=False)
    value='ghp_xY9mL8kR7pQ6wE5tU4iO3aS2dF1gH0jK9lZ8'
    line='github_token="'+value+'"; request_id="abc"'
    a=copy.deepcopy(hard_cases()[0]);a.update(value=value,line={'text':line,'value_start':14,'value_end':14+len(value)})
    # Use exact Python character offsets, including quotes.
    a['line']['value_start']=line.index(value);a['line']['value_end']=line.index(value)+len(value)
    b=copy.deepcopy(a);b.update(candidate_id='sibling',value='abc',line={'text':line,'value_start':line.index('abc'),'value_end':line.index('abc')+3})
    rows=scanner.predict([a,b])
    assert rows[0]['prediction']=='MASK' and rows[1]['prediction']=='KEEP'
