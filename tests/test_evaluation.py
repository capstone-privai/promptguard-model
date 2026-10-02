import pytest
from conftest import candidate
from privai_model.data.adapter import adapt_records
from privai_model.evaluation.evaluator import align_scores, choose_threshold, metrics, evaluate, validate_prediction_v1


def test_threshold_highest_valid_and_ties():
    labels = [1]*100+[0,0]
    scores = [i/100 for i in range(100)]+[0.8,0.0]
    assert choose_threshold(labels,scores,.95) == .05
    assert choose_threshold(labels,scores,.99) == .01
    assert choose_threshold([1,1,0],[.5,.5,.5],.95) == .5
    assert metrics(labels,scores,.05)['recall'] == .95


def test_no_positive_threshold():
    with pytest.raises(ValueError):
        choose_threshold([0,0],[.1,.2],.95)


def test_id_join_and_metrics():
    examples = adapt_records([candidate(1),candidate(2,'KEEP')])
    predictions = [{'candidate_id':'synthetic:2','score':.8},{'candidate_id':'synthetic:1','score':.9}]
    assert align_scores(examples,predictions) == ([1,0],[.9,.8])
    report = evaluate(examples,predictions,.85)
    assert report['recall'] == 1 and report['precision'] == 1
    assert report['false_positive_count'] == report['false_negative_count'] == 0
    assert report['average_precision'] == 1


@pytest.mark.parametrize('rows', [[], [{'candidate_id':'synthetic:1','score':.9}]*2,
    [{'candidate_id':'bad','score':.9}], [{'candidate_id':'synthetic:1','score':float('nan')}],
    [{'candidate_id':'synthetic:1','score':1.1}]])
def test_invalid_predictions(rows):
    with pytest.raises(ValueError):
        align_scores(adapt_records([candidate(1)]),rows)


def test_hard_predictions():
    examples = adapt_records([candidate(1),candidate(2,'KEEP')])
    assert align_scores(examples,[{'candidate_id':'synthetic:1','prediction':'MASK'},
                                  {'candidate_id':'synthetic:2','prediction':0}]) == ([1,0],[1.,0.])


def test_prediction_v1_coverage():
    row = {'schema_version':'prediction-v1','candidate_id':'a','model_version':'test',
           'p_mask':.5,'threshold':.5,'action':'MASK','latency_ms':1.}
    validate_prediction_v1([row],['a'])
    for rows, ids in [([row,row],['a']),([] ,['a']),([row],['a','b']),([dict(row,action='KEEP')],['a'])]:
        with pytest.raises(ValueError):
            validate_prediction_v1(rows,ids)
