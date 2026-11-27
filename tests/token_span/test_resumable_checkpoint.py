"""A full fine-tune that outlives a Colab runtime: checkpoint staged locally then swapped onto Drive, best epoch kept."""
import json
from pathlib import Path

import torch

from privai_model.token_span import train as ts


def parts():
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    return model, optimizer, scheduler, scaler


def test_a_staged_checkpoint_replaces_the_old_one_and_leaves_no_staging_or_part(tmp_path):
    model, optimizer, scheduler, scaler = parts()
    drive, staging = tmp_path / 'drive' / 'last', tmp_path / 'local' / 'stage'
    ts.save_checkpoint(drive, model, optimizer, scheduler, scaler, dict(epoch=1), staging)
    ts.save_checkpoint(drive, model, optimizer, scheduler, scaler, dict(epoch=2), staging)
    assert sorted(p.name for p in drive.iterdir()) == ['checkpoint.pt', 'state.json']
    assert json.loads((drive / 'state.json').read_text()) == dict(epoch=2)
    assert not staging.exists() and not drive.with_name('last.part').exists()
    saved = torch.load(drive / 'checkpoint.pt', weights_only=False)
    assert set(saved) == {'trainable', 'optimizer', 'scheduler', 'scaler', 'rng'}


def test_an_unstaged_checkpoint_works_as_before(tmp_path):
    model, optimizer, scheduler, scaler = parts()
    ts.save_checkpoint(tmp_path / 'last', model, optimizer, scheduler, scaler, dict(epoch=1))
    assert sorted(p.name for p in (tmp_path / 'last').iterdir()) == ['checkpoint.pt', 'state.json']
    assert not (tmp_path / 'last.tmp').exists()


class Saves:
    def __init__(self, name):
        self.name = name

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / f'{self.name}.bin').write_text(self.name)


def test_best_dir_holds_only_the_latest_best_epoch(tmp_path):
    best = tmp_path / 'best'
    ts.save_best(Saves('model1'), Saves('tok'), best, 1)
    ts.save_best(Saves('model2'), Saves('tok'), best, 2)
    assert json.loads((best / 'epoch.json').read_text()) == dict(epoch=2)
    assert sorted(p.name for p in best.iterdir()) == ['epoch.json', 'model2.bin', 'tok.bin']
    assert not (tmp_path / 'best.part').exists()


def test_the_best_epoch_follows_the_selection_rule():
    h = lambda e, f2, p: dict(epoch=e, selected=dict(f2=f2, precision=p))
    assert ts.chosen_epoch([h(1, 0.847, 0.82), h(2, 0.856, 0.832), h(3, 0.844, 0.818)])['epoch'] == 2
    assert ts.chosen_epoch([h(1, 0.85, 0.8), h(2, 0.85, 0.9)])['epoch'] == 2       # precision breaks an F2 tie
    assert ts.chosen_epoch([h(1, 0.85, 0.8), h(2, 0.85, 0.8)])['epoch'] == 1       # then the earlier epoch


def test_a_run_resumed_on_a_new_runtime_matches_an_uninterrupted_run(tmp_path):
    """Two epochs, then a 'new runtime' (empty output_dir) resumes from the Drive checkpoint for the third. The history
    equals a three-epoch run without the break, and best_dir holds the chosen epoch."""
    from privai_model.io import write_jsonl
    from tests.token_span.test_token_span import span_rows, tiny_checkpoint
    data = tmp_path / 'data'
    data.mkdir()
    write_jsonl(data / 'train.jsonl', span_rows(12, 't'))
    write_jsonl(data / 'validation.jsonl', span_rows(6, 'v'))
    base = dict(model_name=str(tiny_checkpoint(tmp_path)), revision=None, local_files_only=True, bidirectional=True,
                attn_implementation='sdpa', lora=None, data_dir=str(data), device='cpu', seed=1, patience=3, batch_size=4,
                learning_rate=1e-2, class_weights=[1.0, 2.0, 2.0], max_length=16, stride=4, eval_batch_size=8,
                log_every=1000)
    whole = ts.train(dict(base, epochs=3, output_dir=str(tmp_path / 'whole')), log=lambda _: None)
    drive = tmp_path / 'drive'
    resumable = dict(base, checkpoint_dir=str(drive / 'last'), checkpoint_staging=str(tmp_path / 'stage'),
                     best_dir=str(drive / 'best'))
    ts.train(dict(resumable, epochs=2, output_dir=str(tmp_path / 'runtime1')), log=lambda _: None)
    resumed = ts.train(dict(resumable, epochs=3, output_dir=str(tmp_path / 'runtime2')), resume=True, log=lambda _: None)
    strip = lambda s: [{k: h[k] for k in ('epoch', 'loss', 'selected')} for h in s['history']]
    assert strip(resumed) == strip(whole) and resumed['selected'] == whole['selected']
    assert sorted(p.name for p in (tmp_path / 'runtime2').glob('epoch-*')) == ['epoch-3']
    assert json.loads((drive / 'best' / 'epoch.json').read_text())['epoch'] == resumed['selected']['epoch']
