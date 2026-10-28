import importlib.util
from pathlib import Path
from privai_model.ml_span.context_data import generate_context


def calibration():
    spec=importlib.util.spec_from_file_location('calibrate_span_precision',Path(__file__).resolve().parents[2]/'scripts/calibrate_span_precision.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.choose_threshold


def test_calibration_rejects_precision_gain_that_loses_reference_recall():
    data,_=generate_context(1)
    rows=data['train'][:4]
    reference={r['id']:[dict(s,score=1.) for s in r['spans']] for r in rows}
    candidate={k:[dict(s) for s in v] for k,v in reference.items()}
    candidate[rows[0]['id']][0]['score']=.97
    candidate[rows[2]['id']][0]['score']=.6
    candidate[rows[1]['id']]=[dict(rows[1]['value_spans'][0],score=.8)]
    result=calibration()(rows,reference,candidate,.5,[.5,.9])
    assert result['selected'] is None
    assert result['choices'][1]['metric']['precision']==1.
    assert result['choices'][1]['metric']['recall']==.5


def test_calibration_can_remove_false_positive_without_dropping_recall():
    data,_=generate_context(1);rows=data['train'][:2]
    reference={r['id']:[dict(r['value_spans'][0],score=1.)] for r in rows}
    candidate={rows[0]['id']:[dict(rows[0]['spans'][0],score=.97)],
               rows[1]['id']:[dict(rows[1]['value_spans'][0],score=.8)]}
    result=calibration()(rows,reference,candidate,.5,[.5,.9])
    assert result['selected']['threshold']==.9
    assert result['selected']['metric']['tp']==1
    assert result['selected']['metric']['fp']==0


def test_observed_cutoff_can_find_valid_point_between_grid_values():
    data,_=generate_context(1);rows=data['train'][:2]
    reference={rows[0]['id']:[dict(rows[0]['spans'][0],score=1.)],rows[1]['id']:[]}
    candidate={rows[0]['id']:[dict(rows[0]['spans'][0],score=.93)],
               rows[1]['id']:[dict(rows[1]['value_spans'][0],score=.91)]}
    assert calibration()(rows,reference,candidate,.5,[.9,.95])['selected'] is None
    observed=sorted({s['score'] for spans in candidate.values() for s in spans})
    result=calibration()(rows,reference,candidate,.5,observed)
    assert result['selected']['threshold']==.93
    assert result['selected']['metric']['precision']==1.
    assert result['selected']['metric']['recall']==1.
