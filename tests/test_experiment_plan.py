import importlib.util
from pathlib import Path
import yaml
from privai_model.io import write_jsonl
from privai_model.data.synthetic import hard_cases


def test_experiment_plan_shared_split_no_test(tmp_path):
    spec=importlib.util.spec_from_file_location('plan',Path(__file__).parents[1]/'scripts/prepare_experiments.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=tmp_path/'fixture.jsonl';write_jsonl(data,hard_cases())
    plans=module.prepare(data,tmp_path/'plan',windows=[64,256])
    assert len(plans)==16
    configs=[yaml.safe_load(Path(p['config']).read_text()) for p in plans]
    assert len({c['split_manifest'] for c in configs})==1
    assert all(c['evaluate_test'] is False for c in configs)
    assert all(not Path(c['output_dir']).exists() for c in configs)
