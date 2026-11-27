"""One trainer for BIO token-classification span detectors: mmBERT (full fine-tune) or Qwen (LoRA + head).

Both models read the same windows (ml_span.bio: max_length tokens, `stride` overlap), predict O / B-SECRET /
I-SECRET per token, and turn overlapping windows into spans the same way (bio.aggregate: averaged probabilities,
span score = mean secret probability). The ml_span modules are imported, not changed.

Epoch and cutoff are chosen on validation only (exact-span F2, precision tie-break, every observed span score is a
cutoff candidate). The test split is never read here.

Qwen is a causal decoder: with Qwen2ForTokenClassification each token sees only the text before it. That is the
model as Hugging Face defines it. `bidirectional: true` in the config replaces the causal mask with the padding mask
alone, so every token sees the whole window as in an encoder (see `bidirectional`).

On CUDA the forward pass runs in float16 autocast with a gradient scaler (the trainable weights stay float32).
Larger Qwen models (7B, 14B) do not fit in float32: `torch_dtype: bfloat16` loads the frozen base in bfloat16 and
`bf16: true` runs the forward pass in bfloat16 autocast (no scaler); `load_in_4bit: true` loads it in 4-bit NF4
(QLoRA, needs bitsandbytes). Trainable weights (LoRA, head) are float32 in every case.
`checkpoint_dir` keeps a resumable checkpoint (trainable weights, optimizer, scheduler, scaler, position in the
epoch) so a Colab runtime reset loses at most `save_every` steps.

For a full fine-tune that outlives a runtime, `checkpoint_dir` and `best_dir` can sit on Google Drive while
`output_dir` stays on the runtime's disk:
- `checkpoint_staging` (a local folder): the checkpoint is written there first, then swapped into `checkpoint_dir`, so
  Drive never holds two checkpoints at once.
- `best_dir`: the weights of the best epoch so far (the epoch `selection.json` would choose), replaced when a later
  epoch becomes best, so the chosen epoch survives the runtime that trained it.
"""
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}

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


def amp_dtype(config, device):
    """Autocast dtype: bfloat16 with `bf16: true` (CUDA or CPU), float16 with `fp16: true` (CUDA only), else none."""
    if device.type == 'cpu':
        return torch.bfloat16 if config.get('bf16') else None
    if device.type != 'cuda':
        return None
    if config.get('bf16'):
        if torch.cuda.get_device_capability(device)[0] < 8:  # native from Ampere; a T4 only emulates it, slowly
            raise RuntimeError('bf16 requested but this GPU has no native bfloat16 (T4): use an A100/L4/H100')
        return torch.bfloat16
    return torch.float16 if config.get('fp16') else None


def base_loading(config):
    """from_pretrained arguments for the frozen base: dtype, and 4-bit quantization for QLoRA."""
    dtype = DTYPES[config.get('torch_dtype', 'float32')]
    if dtype == torch.bfloat16 and config.get('fp16') and not config.get('bf16'):
        raise ValueError('a bfloat16 base needs bf16: true (float16 autocast would mix the two half types)')
    if not config.get('load_in_4bit'):
        return dict(torch_dtype=dtype)
    if not config.get('lora'):
        raise ValueError('load_in_4bit needs lora: a 4-bit base cannot be fine-tuned whole')
    from transformers import BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.float32 else dtype)
    return dict(torch_dtype=dtype, quantization_config=quant, device_map={'': torch.cuda.current_device()})


def place(model, device):
    """Move to the device, except a 4-bit model, which from_pretrained already put on the GPU."""
    return model if getattr(model, 'is_loaded_in_4bit', False) else model.to(device)


def float32_trainable(model):
    """Keep every trainable weight (LoRA, head) in float32 even when the frozen base is bfloat16 or 4-bit."""
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model


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
        model = AutoModelForTokenClassification.from_pretrained(adapter, **extra)
        return bidirectional(model) if config.get('bidirectional') else model
    model = AutoModelForTokenClassification.from_pretrained(config['model_name'], config=cfg, **base_loading(config),
                                                            **kwargs, **extra)
    if config.get('bidirectional'):
        bidirectional(model)
    if config.get('load_in_4bit'):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(config.get('gradient_checkpointing')),
                                                gradient_checkpointing_kwargs={'use_reentrant': False})
    elif config.get('gradient_checkpointing'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
    if lora:
        from peft import LoraConfig, PeftModel, get_peft_model
        if adapter is not None:
            return float32_trainable(PeftModel.from_pretrained(model, adapter))
        model = get_peft_model(model, LoraConfig(task_type='TOKEN_CLS', r=lora['r'], lora_alpha=lora['alpha'],
                                                 lora_dropout=lora['dropout'], target_modules=lora['target_modules'],
                                                 modules_to_save=lora.get('modules_to_save'), bias='none'))
        if config.get('gradient_checkpointing'):
            model.enable_input_require_grads()
        float32_trainable(model)
    return model


def bidirectional(model):
    """Make a decoder's token classifier attend both ways: each token sees the whole window except padding.

    For Qwen2-style models in transformers 4.57, whose inner model uses a ready-made mask when it is given a dict of
    masks per layer type. A hook on the inner model swaps the (batch, length) padding mask for that dict, so training,
    prediction and the LoRA wrapper all go through it. SDPA only: it reads a boolean mask as "may attend"."""
    inner = getattr(model, model.base_model_prefix)
    if inner.config._attn_implementation != 'sdpa':
        raise ValueError(f'bidirectional attention needs sdpa, not {inner.config._attn_implementation}')
    for layer in inner.layers:
        layer.self_attn.is_causal = False

    def padding_only(module, args, kwargs):
        mask = kwargs.get('attention_mask')
        if isinstance(mask, dict):
            return None
        if mask is None:
            ids = kwargs.get('input_ids')
            ids = args[0] if ids is None and args else ids
            mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
        keys = mask.bool()[:, None, None, :]
        full = keys.expand(-1, 1, mask.shape[1], -1)
        kwargs['attention_mask'] = {kind: full for kind in set(module.config.layer_types)}
        return args, kwargs

    inner.register_forward_pre_hook(padding_only, with_kwargs=True)
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
def predict(model, tokenizer, rows, max_length, stride, device, amp=None, batch_size=16):
    """{row id: spans with scores, cutoff 0}. Windows of all rows are batched by length, then aggregated per row.
    `amp` is the autocast dtype on CUDA (amp_dtype), or None; True means float16, as the old `fp16` flag did."""
    amp = torch.float16 if amp is True else (amp or None)
    if amp == torch.float16 and device.type != 'cuda':
        amp = None  # float16 autocast is CUDA only, as before
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
        with torch.autocast(device.type, dtype=amp or torch.float16, enabled=amp is not None):
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


def source_of(row_id):
    return row_id.split(':', 1)[0]


def by_source(rows, predictions, threshold):
    """Exact-span counts for each data source (the row ID prefix) at one cutoff, or None when validation holds
    one source. Reported beside the combined counts; selection uses the combined counts only."""
    names = sorted({source_of(r['id']) for r in rows})
    if len(names) < 2:
        return None
    gold = gold_keys(rows)
    kept = {(rid, s['start'], s['end']) for rid, spans in predictions.items() for s in spans if s['score'] >= threshold}
    return {n: measures({k for k in gold if source_of(k[0]) == n}, {k for k in kept if source_of(k[0]) == n})
            for n in names}


def resolved_revision(model):
    """The Hub commit the base weights came from, so a config with `revision: null` can be pinned afterwards."""
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    return getattr(base.config, '_commit_hash', None)


def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(directory, model, optimizer, scheduler, scaler, state, staging=None):
    """Write the checkpoint next to `directory` and swap it in. With `staging` (a local folder) it is written there,
    the old checkpoint is removed, and the new one is copied in: `directory` never holds two checkpoints, but a crash
    between the removal and the copy leaves none (training then starts over)."""
    directory = Path(directory)
    tmp = Path(staging) if staging else directory.with_name(directory.name + '.tmp')
    if staging and tmp.exists():
        shutil.rmtree(tmp)
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
    if staging:
        part = directory.with_name(directory.name + '.part')
        if part.exists():
            shutil.rmtree(part)
        shutil.copytree(tmp, part)
        part.rename(directory)
        shutil.rmtree(tmp)
    else:
        tmp.rename(directory)


def save_weights(model, tokenizer, directory):
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)


def chosen_epoch(history):
    """The epoch selection.json chooses: best exact-span F2, then precision, then the earlier epoch."""
    return max(history, key=lambda h: ((h['selected']['f2'] or 0), (h['selected']['precision'] or 0), -h['epoch']))


def save_best(model, tokenizer, directory, epoch):
    """Replace `directory` with this epoch's weights (removed first, so it never holds two copies) and its number."""
    directory = Path(directory)
    part = directory.with_name(directory.name + '.part')
    for d in (directory, part):
        if d.exists():
            shutil.rmtree(d)
    save_weights(model, tokenizer, part)
    write_json(part / 'epoch.json', dict(epoch=epoch))
    part.rename(directory)


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
    amp = amp_dtype(config, device)
    fp16 = amp == torch.float16  # only float16 needs the gradient scaler
    data = Path(config['data_dir'])
    train_rows, val_rows = load(data / 'train.jsonl'), load(data / 'validation.jsonl')
    tokenizer = load_tokenizer(config)
    model = place(load_model(config), device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_params, trainable_params = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in trainable)
    began = time.perf_counter()
    ids, labels = encode(train_rows, tokenizer, config['max_length'], config['stride'])
    lengths = [len(x) for x in ids]
    log(json.dumps(dict(event='encoded', windows=len(ids), tokens=int(sum(lengths)), seconds=round(time.perf_counter() - began, 1),
                        total_parameters=total_params, trainable_parameters=trainable_params, device=str(device),
                        amp=str(amp).replace('torch.', '') if amp else None)))
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
    elif config.get('best_dir') and Path(config['best_dir']).exists():
        shutil.rmtree(config['best_dir'])  # a fresh start: a best epoch from an earlier attempt does not belong to it
    write_json(out / 'run.json', dict(total_parameters=total_params, trainable_parameters=trainable_params, windows=len(ids),
                                      tokens=int(sum(lengths)), steps_per_epoch=steps_per_epoch, device=str(device), fp16=fp16,
                                      amp=str(amp).replace('torch.', '') if amp else None, torch=torch.__version__,
                                      gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                                      revision_resolved=resolved_revision(model)))
    best, stale = -1.0, 0  # early-stopping state, replayed from the history after a resume
    for h in state['history']:
        if (h['selected']['f2'] or 0) > best:
            best, stale = h['selected']['f2'] or 0, 0
        else:
            stale += 1
    progress = open(out / 'progress.jsonl', 'a')
    last_save = time.perf_counter()
    staging = config.get('checkpoint_staging')
    # The patience check also covers a resume from the checkpoint saved just before an early stop.
    while state['epoch'] < config['epochs'] and stale < config.get('patience', config['epochs']):
        epoch = state['epoch']
        batches = epoch_batches(lengths, config['batch_size'], config['seed'], epoch)
        model.train()
        started = time.perf_counter()
        for b in range(state['batch'], len(batches)):
            input_ids, mask, target = collate(batches[b], ids, labels, tokenizer.pad_token_id, device)
            if not bool((target != -100).any()):
                state['batch'] = b + 1
                continue
            with torch.autocast(device.type, dtype=amp or torch.float16, enabled=amp is not None):
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
                save_checkpoint(ckpt, model, optimizer, scheduler, scaler, state, staging)
                last_save = time.perf_counter()
        predictions = predict(model, tokenizer, val_rows, config['max_length'], config['stride'], device, amp,
                              config.get('eval_batch_size', 16))
        raw, selected, cutoffs = select_cutoff(val_rows, predictions)
        entry = dict(epoch=epoch + 1, loss=state['loss_sum'] / max(1, state['loss_batches']), raw=raw, selected=selected,
                     cutoffs_evaluated=cutoffs, seconds=round(time.perf_counter() - started, 1))
        sources = by_source(val_rows, predictions, selected['threshold'])
        if sources:
            entry['by_source'] = sources
        state['history'].append(entry)
        log(json.dumps(entry))
        save_weights(model, tokenizer, out / f'epoch-{epoch + 1}')
        if config.get('best_dir') and chosen_epoch(state['history'])['epoch'] == epoch + 1:
            save_best(model, tokenizer, config['best_dir'], epoch + 1)
        state.update(epoch=epoch + 1, batch=0, loss_sum=0.0, loss_batches=0)
        write_json(out / 'history.json', state['history'])
        if (selected['f2'] or 0) > best:
            best, stale = selected['f2'] or 0, 0
        else:
            stale += 1
        save_checkpoint(ckpt, model, optimizer, scheduler, scaler, state, staging)
        if stale >= config.get('patience', config['epochs']):
            break
    progress.close()
    chosen = chosen_epoch(state['history'])
    selection = dict(policy='Validation only: exact-span F2 at every observed span score, precision tie-break; the epoch '
                            'with the best such F2 (earlier epoch on ties). Test not read.',
                     selected=dict(epoch=chosen['epoch'], threshold=chosen['selected']['threshold']),
                     history=state['history'], validation=dict(path=str(data / 'validation.jsonl'), rows=len(val_rows)))
    write_json(out / 'selection.json', selection)
    return selection
