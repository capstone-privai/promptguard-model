"""One trainer for BIO token-classification span detectors: mmBERT (full fine-tune) or Qwen (LoRA + head).

Both models read the same windows (ml_span.bio: max_length tokens, `stride` overlap), predict O / B-SECRET /
I-SECRET per token, and turn overlapping windows into spans the same way (bio.aggregate: averaged probabilities,
span score = mean secret probability). The ml_span modules are imported, not changed.

Epoch and cutoff are chosen on validation only (exact-span F2, precision tie-break, every observed span score is a
cutoff candidate). The test split is never read here.

Qwen is a causal decoder: with Qwen2ForTokenClassification each token sees only the text before it. That is the
model as Hugging Face defines it; it is not made bidirectional here.

On CUDA the forward pass runs in float16 autocast with a gradient scaler (the trainable weights stay float32).
`checkpoint_dir` keeps a resumable checkpoint (trainable weights, optimizer, scheduler, scaler, position in the
epoch) so a Colab runtime reset loses at most `save_every` steps.
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from privai_model.io import write_json
from privai_model.ml_span.bio import LABELS, aggregate, align, windows
from privai_model.ml_span.evaluation import measures
from privai_model.ml_span.schema import load


def resolve_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable')
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(config):
    kwargs = dict(revision=config.get('revision'), local_files_only=config.get('local_files_only', False))
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], **kwargs)
    if not tokenizer.is_fast:
        raise ValueError('fast tokenizer required for character offsets')
    if tokenizer.pad_token_id is None:
        raise ValueError('tokenizer has no pad token')
    return tokenizer


def load_model(config, adapter=None):
    """The base model with a fresh 3-label head; LoRA on top when config['lora'] is set.
    `adapter` loads trained weights saved by save_weights (a LoRA adapter folder or a full model folder)."""
    kwargs = dict(revision=config.get('revision'), local_files_only=config.get('local_files_only', False))
    cfg = AutoConfig.from_pretrained(config['model_name'], **kwargs)
    cfg.num_labels = 3
    cfg.id2label = dict(LABELS)
    cfg.label2id = {v: k for k, v in LABELS.items()}
    if cfg.model_type == 'modernbert':
        cfg.reference_compile = False
    extra = {}
    if config.get('attn_implementation'):
        extra['attn_implementation'] = config['attn_implementation']
    lora = config.get('lora')
    if adapter is not None and not lora:  # a fully fine-tuned model is saved whole
        return AutoModelForTokenClassification.from_pretrained(adapter, **extra)
    model = AutoModelForTokenClassification.from_pretrained(config['model_name'], config=cfg, torch_dtype=torch.float32,
                                                            **kwargs, **extra)
    if config.get('gradient_checkpointing'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
    if lora:
        from peft import LoraConfig, PeftModel, get_peft_model
        if adapter is not None:
            return PeftModel.from_pretrained(model, adapter)
        model = get_peft_model(model, LoraConfig(task_type='TOKEN_CLS', r=lora['r'], lora_alpha=lora['alpha'],
                                                 lora_dropout=lora['dropout'], target_modules=lora['target_modules'],
                                                 modules_to_save=lora.get('modules_to_save'), bias='none'))
        if config.get('gradient_checkpointing'):
            model.enable_input_require_grads()
    return model


def encode(rows, tokenizer, max_length, stride):
    """Training windows as compact arrays (ids int32, labels int8 with -100 for ignored tokens)."""
    ids, labels = [], []
    for row in rows:
        for w in windows(tokenizer, row['text'], max_length, stride):
            ids.append(np.asarray(w['inputs']['input_ids'], dtype=np.int32))
            labels.append(np.asarray(align(row['text'], w['offsets'], row['spans'], w['special']), dtype=np.int8))
    return ids, labels


def epoch_batches(lengths, batch_size, seed, epoch):
    """Length-grouped batches in an order fixed by (seed, epoch), so a resumed epoch replays the same batches."""
    rng = random.Random(seed * 1000 + epoch)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    order.sort(key=lambda i: lengths[i])
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


def collate(index, ids, labels, pad_id, device):
    width = max(len(ids[i]) for i in index)
    input_ids = torch.full((len(index), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(index), width), dtype=torch.long)
    target = torch.full((len(index), width), -100, dtype=torch.long)
    for row, i in enumerate(index):
        n = len(ids[i])
        input_ids[row, :n] = torch.from_numpy(ids[i].astype(np.int64))
        mask[row, :n] = 1
        target[row, :n] = torch.from_numpy(labels[i].astype(np.int64))
    return input_ids.to(device), mask.to(device), target.to(device)


@torch.inference_mode()
def predict(model, tokenizer, rows, max_length, stride, device, fp16=False, batch_size=16):
    """{row id: spans with scores, cutoff 0}. Windows of all rows are batched by length, then aggregated per row."""
    model.eval()
    jobs = []
    for r, row in enumerate(rows):
        if row['text']:
            for w in windows(tokenizer, row['text'], max_length, stride):
                jobs.append((r, w))
    jobs.sort(key=lambda j: len(j[1]['inputs']['input_ids']))
    probs = [None] * len(jobs)
    pad_id = tokenizer.pad_token_id
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        width = max(len(w['inputs']['input_ids']) for _, w in chunk)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for k, (_, w) in enumerate(chunk):
            n = len(w['inputs']['input_ids'])
            input_ids[k, :n] = torch.tensor(w['inputs']['input_ids'])
            mask[k, :n] = 1
        with torch.autocast('cuda', dtype=torch.float16, enabled=fp16 and device.type == 'cuda'):
            logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        p = logits.float().softmax(-1).cpu()
        for k, (_, w) in enumerate(chunk):
            probs[start + k] = p[k, :len(w['inputs']['input_ids'])].tolist()
    per_row = {}
    for (r, w), p in zip(jobs, probs):
        per_row.setdefault(r, []).append((w['offsets'], p))
    return {row['id']: aggregate(row['text'], per_row[r]) if r in per_row else [] for r, row in enumerate(rows)}


def gold_keys(rows):
    return {(r['id'], s['start'], s['end']) for r in rows for s in r['spans']}


def select_cutoff(rows, predictions):
    """Exact-span F2 at every observed span score (and 0); best F2, then precision, then the higher cutoff."""
    gold = gold_keys(rows)
    scored = sorted(((s['score'], (rid, s['start'], s['end'])) for rid, spans in predictions.items() for s in spans),
                    reverse=True)
    best, tp, fp = None, 0, 0
    for i, (score, key) in enumerate(scored):
        tp += key in gold
        fp += key not in gold
        if i + 1 < len(scored) and scored[i + 1][0] == score:
            continue  # a cutoff keeps every span with that score
        m = dict(threshold=score, **measures(gold, {k for _, k in scored[:i + 1]}))
        if best is None or (m['f2'] or 0, m['precision'] or 0, score) > (best['f2'] or 0, best['precision'] or 0, best['threshold']):
            best = m
    raw = dict(threshold=0.0, **measures(gold, {k for _, k in scored}))
    return raw, best or raw, len({s for s, _ in scored})


def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(directory, model, optimizer, scheduler, scaler, state):
    directory = Path(directory)
    tmp = directory.with_name(directory.name + '.tmp')
    tmp.mkdir(parents=True, exist_ok=True)
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
               cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    torch.save(dict(trainable=trainable_state(model), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict(), rng=rng), tmp / 'checkpoint.pt')
    write_json(tmp / 'state.json', state)
    for p in tmp.iterdir():
        if p.stat().st_size == 0:
            raise OSError(f'{p} was written empty (disk full?)')
    if directory.exists():
        for p in directory.iterdir():
            p.unlink()
        directory.rmdir()
    tmp.rename(directory)


def save_weights(model, tokenizer, directory):
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)


def schedule(config, total_steps):
    warmup = int(config.get('warmup_ratio', 0.0) * total_steps)
    if config.get('schedule', 'constant') == 'constant':
        return lambda s: min(1.0, (s + 1) / warmup) if warmup else 1.0
    return lambda s: min((s + 1) / max(1, warmup), 0.5 * (1 + math.cos(math.pi * max(0, s + 1 - warmup) / max(1, total_steps - warmup))))


def train(config, resume=False, log=print):
    out = Path(config['output_dir'])
    ckpt = Path(config.get('checkpoint_dir') or out / 'last')
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f'{out} is not empty; pass resume to continue')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', config)
    seed_everything(config['seed'])
    device = resolve_device(config.get('device', 'auto'))
    fp16 = bool(config.get('fp16')) and device.type == 'cuda'
    data = Path(config['data_dir'])
    train_rows, val_rows = load(data / 'train.jsonl'), load(data / 'validation.jsonl')
    tokenizer = load_tokenizer(config)
    model = load_model(config).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_params, trainable_params = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in trainable)
    began = time.perf_counter()
    ids, labels = encode(train_rows, tokenizer, config['max_length'], config['stride'])
    lengths = [len(x) for x in ids]
    log(json.dumps(dict(event='encoded', windows=len(ids), tokens=int(sum(lengths)), seconds=round(time.perf_counter() - began, 1),
                        total_parameters=total_params, trainable_parameters=trainable_params, device=str(device), fp16=fp16)))
    steps_per_epoch = math.ceil(len(ids) / config['batch_size'])
    optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0.0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule(config, steps_per_epoch * config['epochs']))
    scaler = torch.amp.GradScaler('cuda', enabled=fp16)
    weights = torch.tensor(config['class_weights'], dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    state = dict(epoch=0, batch=0, step=0, history=[], loss_sum=0.0, loss_batches=0)
    if resume and (ckpt / 'state.json').exists():
        saved = torch.load(ckpt / 'checkpoint.pt', map_location='cpu', weights_only=False)
        named = dict(model.named_parameters())
        for name, value in saved['trainable'].items():
            named[name].data.copy_(value.to(device))
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        random.setstate(saved['rng']['python'])
        np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['torch'])
        if saved['rng']['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        state = json.loads((ckpt / 'state.json').read_text())
        log(json.dumps(dict(event='resumed', epoch=state['epoch'], batch=state['batch'], step=state['step'])))
    write_json(out / 'run.json', dict(total_parameters=total_params, trainable_parameters=trainable_params, windows=len(ids),
                                      tokens=int(sum(lengths)), steps_per_epoch=steps_per_epoch, device=str(device), fp16=fp16,
                                      torch=torch.__version__))
    best, stale = -1.0, 0  # early-stopping state, replayed from the history after a resume
    for h in state['history']:
        if (h['selected']['f2'] or 0) > best:
            best, stale = h['selected']['f2'] or 0, 0
        else:
            stale += 1
    progress = open(out / 'progress.jsonl', 'a')
    last_save = time.perf_counter()
    while state['epoch'] < config['epochs']:
        epoch = state['epoch']
        batches = epoch_batches(lengths, config['batch_size'], config['seed'], epoch)
        model.train()
        started = time.perf_counter()
        for b in range(state['batch'], len(batches)):
            input_ids, mask, target = collate(batches[b], ids, labels, tokenizer.pad_token_id, device)
            if not bool((target != -100).any()):
                state['batch'] = b + 1
                continue
            with torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
                logits = model(input_ids=input_ids, attention_mask=mask).logits
            loss = loss_fn(logits.float().reshape(-1, 3), target.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, config.get('max_grad_norm', 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            state['step'] += 1
            state['batch'] = b + 1
            state['loss_sum'] += loss.item()
            state['loss_batches'] += 1
            if state['step'] % config.get('log_every', 200) == 0:
                record = dict(epoch=epoch, batch=b + 1, batches=len(batches), step=state['step'],
                              mean_loss=state['loss_sum'] / state['loss_batches'], lr=scheduler.get_last_lr()[0],
                              seconds=round(time.perf_counter() - started, 1))
                progress.write(json.dumps(record) + '\n')
                progress.flush()
                log(json.dumps(record))
            due = config.get('save_every') and state['step'] % config['save_every'] == 0
            if config.get('save_every_minutes') and time.perf_counter() - last_save >= 60 * config['save_every_minutes']:
                due = True
            if due:
                save_checkpoint(ckpt, model, optimizer, scheduler, scaler, state)
                last_save = time.perf_counter()
        predictions = predict(model, tokenizer, val_rows, config['max_length'], config['stride'], device, fp16,
                              config.get('eval_batch_size', 16))
        raw, selected, cutoffs = select_cutoff(val_rows, predictions)
        entry = dict(epoch=epoch + 1, loss=state['loss_sum'] / max(1, state['loss_batches']), raw=raw, selected=selected,
                     cutoffs_evaluated=cutoffs, seconds=round(time.perf_counter() - started, 1))
        state['history'].append(entry)
        log(json.dumps(entry))
        save_weights(model, tokenizer, out / f'epoch-{epoch + 1}')
        state.update(epoch=epoch + 1, batch=0, loss_sum=0.0, loss_batches=0)
        write_json(out / 'history.json', state['history'])
        if (selected['f2'] or 0) > best:
            best, stale = selected['f2'] or 0, 0
        else:
            stale += 1
        save_checkpoint(ckpt, model, optimizer, scheduler, scaler, state)
        if stale >= config.get('patience', config['epochs']):
            break
    progress.close()
    chosen = max(state['history'], key=lambda h: ((h['selected']['f2'] or 0), (h['selected']['precision'] or 0), -h['epoch']))
    selection = dict(policy='Validation only: exact-span F2 at every observed span score, precision tie-break; the epoch '
                            'with the best such F2 (earlier epoch on ties). Test not read.',
                     selected=dict(epoch=chosen['epoch'], threshold=chosen['selected']['threshold']),
                     history=state['history'], validation=dict(path=str(data / 'validation.jsonl'), rows=len(val_rows)))
    write_json(out / 'selection.json', selection)
    return selection
