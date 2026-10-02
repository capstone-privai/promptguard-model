"""Offline integration: the same train/evaluate/load API used by Ettin."""
import importlib.util
from pathlib import Path
import json
import torch
from privai_model.training.trainer import train
from privai_model.inference.predict import load_model, predict_candidates
from privai_model.data.adapter import iter_jsonl, load_examples
from privai_model.data.split import restore_split
from privai_model.evaluation.evaluator import select_thresholds, evaluate


def test_offline_modernbert_end_to_end(tmp_path):
    spec = importlib.util.spec_from_file_location('smoke', Path(__file__).parents[1]/'scripts/smoke.py')
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        config = smoke.prepare(tmp_path/'smoke')
        report = train(config)
        root = Path(config['output_dir'])
        bundle = load_model(root/'model',device='cpu')
        records = [r for r in iter_jsonl(config['data_path']) if r['granularity']=='value']
        rows = predict_candidates(bundle,records[:3],batch_size=2)
        assert len(rows) == 3
        assert bundle.threshold == report['thresholds']['0.99']
        manifest = json.loads((root/'split_manifest.json').read_text())
        splits = restore_split(load_examples(config['data_path']),manifest)
        val = list(iter_jsonl(root/'validation_predictions.jsonl'))
        test = list(iter_jsonl(root/'test_predictions.jsonl'))
        thresholds = select_thresholds(splits['validation'],val)
        assert thresholds == report['thresholds']
        for target, threshold in thresholds.items():
            assert evaluate(splits['test'],list(reversed(test)),threshold) == report['test'][target]
        # Reload uses identical model/tokenizer and yields identical scores on CPU.
        again = predict_candidates(load_model(root/'model',device='cpu'),records[:3],batch_size=2)
        assert [r['p_mask'] for r in rows] == [r['p_mask'] for r in again]
    finally:
        torch.set_num_threads(threads)
