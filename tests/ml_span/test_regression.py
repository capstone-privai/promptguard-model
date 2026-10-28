from privai_model.ml_span.regression import check_regressions


def test_aggregate_gain_does_not_hide_family_regression():
    report={'validation_decision':{'selected':'candidate'},'datasets':{'old_development':{'candidate':{
        'ml_only':{'tp':200,'fp':0},'by_family_id':{'audit':{'ml_only':{'tp':5,'fp':0}}}}}}}
    gates=[{'name':'audit','dataset':'old_development','family':'audit','bounds':{'tp':{'min':22},'fp':{'max':1}}}]
    assert not check_regressions(report,gates)['eligible']
    report['datasets']['old_development']['candidate']['by_family_id']['audit']['ml_only']['tp']=22
    assert check_regressions(report,gates)['eligible']
    report['validation_decision']['selected']='previous'
    result=check_regressions(report,gates)
    assert result['regression_passed'] and not result['eligible']


def test_missing_and_undefined_results_fail_closed():
    gates=[{'name':'missing','dataset':'missing_development','bounds':{'recall':{'min':.5}}}]
    report={'validation_decision':{'selected':'candidate'},'datasets':{}}
    assert not check_regressions(report,gates)['regression_passed']
    for value in [None,float('nan'),float('inf'),True]:
        report['datasets']['missing_development']={'candidate':{'ml_only':{'recall':value}}}
        assert not check_regressions(report,gates)['eligible']


def test_malformed_gate_cannot_silently_pass():
    import pytest
    report={'validation_decision':{'selected':'candidate'}}
    for bounds in [{}, {'tp':{}}, {'tp':{'minimum':1}}, {'tp':{'min':float('nan')}},
                   {'tp':{'min':2,'max':1}}, {'tp':{'min':True}}]:
        with pytest.raises(ValueError):
            check_regressions(report,[{'name':'bad','dataset':'dev','bounds':bounds}])
    with pytest.raises(ValueError):
        check_regressions(report,[])
