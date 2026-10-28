"""Explicit development gates complement aggregate validation selection.

These checks cannot establish production readiness or independent test accuracy.
Missing datasets, families and undefined metrics fail closed.
"""
import math


def check_regressions(report, gates, model='candidate'):
    checks=[]
    for gate in gates:
        name=gate['name']
        if not isinstance(gate.get('bounds'),dict) or not gate['bounds']:
            raise ValueError('nonempty metric bounds required')
        for bounds in gate['bounds'].values():
            if not isinstance(bounds,dict) or not bounds or set(bounds)-{'min','max'}:
                raise ValueError('min/max bounds required')
            if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in bounds.values()):
                raise ValueError('finite numeric bounds required')
            if bounds.get('min',-math.inf)>bounds.get('max',math.inf):
                raise ValueError('minimum exceeds maximum')
        node=report.get('datasets',{}).get(gate['dataset'],{}).get(model,{})
        if gate.get('family'):
            node=node.get('by_family_id',{}).get(gate['family'],{})
        metric=node.get('ml_only',{})
        observed={key:metric.get(key) for key in gate['bounds']}
        failures=[]
        for key,bounds in gate['bounds'].items():
            value=observed[key]
            if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
                failures.append(key+': missing or undefined')
                continue
            if 'min' in bounds and value<bounds['min']:failures.append(key+': below minimum')
            if 'max' in bounds and value>bounds['max']:failures.append(key+': above maximum')
        checks.append(dict(name=name,dataset=gate['dataset'],family=gate.get('family'),
                           bounds=gate['bounds'],observed=observed,passed=not failures,failures=failures))
    if not checks:raise ValueError('at least one regression gate required')
    validation_selected=report.get('validation_decision',{}).get('selected')==model
    return dict(model=model,validation_selected=validation_selected,
                regression_passed=all(c['passed'] for c in checks),
                eligible=validation_selected and all(c['passed'] for c in checks),checks=checks,
                notice='Development regression gate, not production certification. Test results are not used for these bounds.')
