"""Back-end verifier, 2nd stage (on-premises server): a Qwen decoder as the candidate-v1 classifier, trained with LoRA.

The task, data and selection are the local verifier's (privai_model.training.trainer with mmBERT): the candidate-v1
input with the value between [V] and [/V] (data.input_builder.encode_candidate), P(MASK) per candidate, the epoch
with the lowest validation loss, thresholds at the recall targets on validation, test closed unless evaluate_test.
What differs is how a 1.5B to 14B model is trained, taken from the token-span trainer (token_span.train):

- the frozen base with LoRA on attention and MLP and a fresh 2-label head (Qwen2ForSequenceClassification, which
  reads the state of the last token that is not padding); `bidirectional: true` lets every token see the whole
  input; the base may be bfloat16 or 4-bit, the forward pass runs in float16 or bfloat16 autocast, and the
  trainable weights stay float32;
- the three markers are new tokens, and LoRA leaves the embedding frozen, so only their three rows are trained
  (PEFT trainable tokens, peft 0.15 or later). Each row starts as the mean embedding of the marker's text
  ('[', 'V', ']' ...), so the open and close markers start apart instead of as the same untrained row;
- a resumable checkpoint (trainable weights, optimizer, scheduler, position in the epoch) every
  `save_every_minutes`, because a 7B or 14B epoch takes hours and a Colab runtime can be reset.

`model/` holds the LoRA adapter, the tokenizer, `verifier_lora.json` (the base model and how to load it) and, at
the end, `bundle.json`; inference.predict.load_model reads it, so scripts/final_evaluate.py and
scripts/evaluate_verifier_system.py take this verifier as they take the mmBERT one.
"""
import gc
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from privai_model.data.adapter import load_examples
from privai_model.data.input_builder import MARKERS, encode_candidate, register_markers
from privai_model.data.split import restore_split, split_groups
from privai_model.evaluation.evaluator import evaluate, select_thresholds
from privai_model.inference.predict import ModelBundle, predict_examples
from privai_model.io import write_json, write_jsonl
from privai_model.token_span.train import (amp_dtype, base_loading, bidirectional, epoch_batches, float32_trainable,
                                           place, resolve_device, resolved_revision, save_checkpoint, schedule,
                                           seed_everything)

LORA_FILE = 'verifier_lora.json'
SCHEMA = 'verifier-lora-v1'
# What loading the base needs again at inference; written to LORA_FILE next to the adapter.
LOADING = ('model_name', 'revision', 'local_files_only', 'torch_dtype', 'load_in_4bit', 'bf16', 'fp16',
           'bidirectional', 'attn_implementation')


def validate(config):
    if not config.get('lora'):
        raise ValueError('the LoRA verifier needs a lora section')
    if config['epochs'] < 1 or config['batch_size'] < 1:
        raise ValueError('epochs and batch_size must be positive')
    if config.get('sampling', 'none') != 'none':
        raise ValueError('the LoRA verifier trains on the data as it is: sampling must be none')
    if config.get('character_features'):
        raise ValueError('character_features belong to the mmBERT fusion classifier, not the LoRA verifier')
    weights = config.get('class_weights')
    if weights is not None and (len(weights) != 2 or any(not 0 < w < float('inf') for w in weights)):
        raise ValueError('class_weights must be two positive finite values [KEEP, MASK]')
    if config.get('deployment_recall_target', 0.99) not in config.get('recall_targets', [0.95, 0.99]):
        raise ValueError('deployment recall target must appear in recall_targets')
    from peft import LoraConfig
    if 'trainable_token_indices' not in LoraConfig.__dataclass_fields__:
        raise RuntimeError('the marker rows are trained with PEFT trainable tokens: install peft>=0.15,<1')


def marker_pieces(tokenizer):
    """The ordinary tokens of each marker's text, read before the markers become tokens of their own."""
    return {text: tokenizer.encode(text, add_special_tokens=False) for text in MARKERS.values()}


def load_tokenizer(source, config):
    """The base tokenizer with the three markers added; `source` is the base model or a saved verifier."""
    kwargs = dict(local_files_only=config.get('local_files_only', False))
    if source == config['model_name']:
        kwargs['revision'] = config.get('revision')
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    pieces = None if all(t in tokenizer.get_vocab() for t in MARKERS.values()) else marker_pieces(tokenizer)
    register_markers(tokenizer)
    tokenizer.padding_side = 'right'  # positions count from the first token; training pads on the right too
    if tokenizer.pad_token_id is None:
        raise ValueError('tokenizer has no pad token')
    return tokenizer, pieces


def marker_ids(tokenizer):
    ids = tokenizer.convert_tokens_to_ids(list(MARKERS.values()))
    if len(set(ids)) != len(MARKERS) or tokenizer.unk_token_id in ids:
        raise ValueError('markers are not tokens of their own')
    return ids


def init_marker_rows(model, tokenizer, pieces):
    """Each marker row starts as the mean embedding of the marker's text, so the three markers start apart."""
    weight = model.get_input_embeddings().weight
    with torch.no_grad():
        for text, row in zip(MARKERS.values(), marker_ids(tokenizer)):
            weight[row] = weight[pieces[text]].float().mean(0).to(weight.dtype)


def load_classifier(config, tokenizer, pieces=None, adapter=None):
    """The base with a fresh 2-label head and LoRA, or with the trained adapter when `adapter` is a saved model/."""
    kwargs = dict(revision=config.get('revision'), local_files_only=config.get('local_files_only', False))
    cfg = AutoConfig.from_pretrained(config['model_name'], **kwargs)
    cfg.num_labels = 2
    cfg.id2label = {0: 'KEEP', 1: 'MASK'}
    cfg.label2id = {'KEEP': 0, 'MASK': 1}
    cfg.problem_type = 'single_label_classification'
    cfg.pad_token_id = tokenizer.pad_token_id  # the head reads the last token that is not padding
    extra = {'attn_implementation': config['attn_implementation']} if config.get('attn_implementation') else {}
    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], config=cfg, **base_loading(config),
                                                               **kwargs, **extra)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        # Qwen's embedding already has spare rows for the markers; only a smaller vocabulary is grown.
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    limit = getattr(model.config, 'max_position_embeddings', None)
    if limit and config['window_size'] > limit:
        raise ValueError('window_size exceeds backbone position limit')
    if config.get('bidirectional'):
        bidirectional(model)
    if adapter is None:
        init_marker_rows(model, tokenizer, pieces)
    checkpointing = bool(config.get('gradient_checkpointing')) and adapter is None
    if config.get('load_in_4bit'):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpointing,
                                                gradient_checkpointing_kwargs={'use_reentrant': False})
    elif checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
    from peft import LoraConfig, PeftModel, get_peft_model
    if adapter is not None:
        return PeftModel.from_pretrained(model, adapter).eval()
    lora = config['lora']
    model = get_peft_model(model, LoraConfig(task_type='SEQ_CLS', r=lora['r'], lora_alpha=lora['alpha'],
                                             lora_dropout=lora['dropout'], target_modules=lora['target_modules'],
                                             modules_to_save=lora.get('modules_to_save', ['score']),
                                             trainable_token_indices=marker_ids(tokenizer), bias='none'))
    if checkpointing:
        model.enable_input_require_grads()
    return float32_trainable(model)


def load_bundle(path, device, window_size, model_version, threshold=0.5):
    """A saved LoRA verifier as the ModelBundle predict_examples takes, with the autocast it was trained in."""
    path = Path(path)
    saved = json.loads((path / LORA_FILE).read_text(encoding='utf-8'))
    if saved.get('schema_version') != SCHEMA:
        raise ValueError(f'unsupported {LORA_FILE}')
    config = dict(saved['loading'], window_size=window_size)
    tokenizer, pieces = load_tokenizer(str(path), config)
    if pieces is not None:
        raise ValueError('saved tokenizer has no markers')
    model = place(load_classifier(config, tokenizer, adapter=str(path)), device)
    return ModelBundle(tokenizer, model, device, window_size, model_version, threshold, amp_dtype(config, device))


def save_model(model, tokenizer, directory, config, revision):
    directory = Path(directory)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    loading = {k: config[k] for k in LOADING if k in config}
    loading['revision'] = revision or config.get('revision')
    write_json(directory / LORA_FILE, dict(schema_version=SCHEMA, loading=loading))


def collate(index, rows, pad_id, device):
    width = max(len(rows[i][0]) for i in index)
    input_ids = torch.full((len(index), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(index), width), dtype=torch.long)
    for r, i in enumerate(index):
        n = len(rows[i][0])
        input_ids[r, :n] = torch.tensor(rows[i][0])
        mask[r, :n] = 1
    labels = torch.tensor([rows[i][1] for i in index], dtype=torch.long)
    return input_ids.to(device), mask.to(device), labels.to(device)


@torch.inference_mode()
def validation_loss(model, rows, pad_id, device, amp, batch_size):
    model.eval()
    total = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
    for start in range(0, len(order), batch_size):
        input_ids, mask, labels = collate(order[start:start + batch_size], rows, pad_id, device)
        with torch.autocast(device.type, dtype=amp or torch.float16, enabled=amp is not None):
            logits = model(input_ids=input_ids, attention_mask=mask).logits
        total += loss_fn(logits.float(), labels).item()
    return total / len(rows)


def train(config, resume=False, log=print):
    validate(config)
    examples = load_examples(config['data_path'])
    if config.get('split_manifest'):
        manifest = json.loads(Path(config['split_manifest']).read_text())
        splits = restore_split(examples, manifest)
    else:
        splits, manifest = split_groups(examples, config['seed'], config['split_ratios'])
    for name in ('train', 'validation'):
        if {e.label for e in splits[name]} != {0, 1}:
            raise ValueError(f'{name} must contain both classes; review group/data design')
    out = Path(config['output_dir'])
    ckpt = Path(config.get('checkpoint_dir') or out / 'last')
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f'{out} is not empty; pass resume to continue')
    if (out / 'model' / 'bundle.json').exists():
        raise FileExistsError(f'{out} is finished; choose a new run directory')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', config)
    write_json(out / 'split_manifest.json', manifest)
    log(json.dumps(manifest['statistics'], ensure_ascii=False))
    seed_everything(config['seed'])
    device = resolve_device(config.get('device', 'auto'))
    amp = amp_dtype(config, device)
    fp16 = amp == torch.float16  # only float16 needs the gradient scaler
    tokenizer, pieces = load_tokenizer(config['model_name'], config)
    model = place(load_classifier(config, tokenizer, pieces), device)
    revision = resolved_revision(model)
    # Encode only train/validation; test stays untouched until final evaluation.
    encoded, truncation = {}, {}
    for name in ('train', 'validation'):
        encoded[name], truncation[name] = [], 0
        for ex in splits[name]:
            tokens, info = encode_candidate(ex, tokenizer, config['window_size'])
            encoded[name].append((tokens['input_ids'], ex.label))
            truncation[name] += info['candidate_truncated']
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_params, trainable_params = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in trainable)
    lengths = [len(r[0]) for r in encoded['train']]
    steps_per_epoch = math.ceil(len(lengths) / config['batch_size'])
    optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0.0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule(config, steps_per_epoch * config['epochs']))
    scaler = torch.amp.GradScaler('cuda', enabled=fp16)
    weights = config.get('class_weights')
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device, dtype=torch.float32) if weights else None)
    state = dict(epoch=0, batch=0, step=0, history=[], loss_sum=0.0, count=0, best_loss=None, best_epoch=None)
    if resume and (ckpt / 'state.json').exists():
        saved = torch.load(ckpt / 'checkpoint.pt', map_location='cpu', weights_only=False)
        named = dict(model.named_parameters())
        for name, value in saved['trainable'].items():
            named[name].data.copy_(value.to(named[name].device))
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
    write_json(out / 'run.json', dict(total_parameters=total_params, trainable_parameters=trainable_params,
                                      candidates={k: len(v) for k, v in encoded.items()},
                                      steps_per_epoch=steps_per_epoch, device=str(device),
                                      amp=str(amp).replace('torch.', '') if amp else None, torch=torch.__version__,
                                      gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                                      revision_resolved=revision))
    log(json.dumps(dict(event='encoded', train=len(lengths), validation=len(encoded['validation']),
                        trainable_parameters=trainable_params, device=str(device),
                        amp=str(amp).replace('torch.', '') if amp else None)))
    eval_batch = config.get('eval_batch_size', config['batch_size'])
    last_save = time.perf_counter()
    while state['epoch'] < config['epochs']:
        epoch = state['epoch']
        batches = epoch_batches(lengths, config['batch_size'], config['seed'], epoch)
        model.train()
        started = time.perf_counter()
        for b in range(state['batch'], len(batches)):
            input_ids, mask, labels = collate(batches[b], encoded['train'], tokenizer.pad_token_id, device)
            with torch.autocast(device.type, dtype=amp or torch.float16, enabled=amp is not None):
                logits = model(input_ids=input_ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, config.get('max_grad_norm', 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            state['step'] += 1
            state['batch'] = b + 1
            state['loss_sum'] += loss.item() * len(labels)
            state['count'] += len(labels)
            if state['step'] % config.get('log_every', 200) == 0:
                log(json.dumps(dict(epoch=epoch, batch=b + 1, batches=len(batches), step=state['step'],
                                    mean_loss=state['loss_sum'] / state['count'], lr=scheduler.get_last_lr()[0],
                                    seconds=round(time.perf_counter() - started, 1))))
            if config.get('save_every_minutes') and time.perf_counter() - last_save >= 60 * config['save_every_minutes']:
                save_checkpoint(ckpt, model, optimizer, scheduler, scaler, state)
                last_save = time.perf_counter()
        val_loss = validation_loss(model, encoded['validation'], tokenizer.pad_token_id, device, amp, eval_batch)
        if not math.isfinite(val_loss):
            raise RuntimeError('nonfinite validation loss')
        record = dict(epoch=epoch + 1, train_loss=state['loss_sum'] / max(1, state['count']), validation_loss=val_loss,
                      seconds=round(time.perf_counter() - started, 1))
        state['history'].append(record)
        log(json.dumps(record))
        write_json(out / 'history.json', state['history'])
        if state['best_loss'] is None or val_loss < state['best_loss']:
            state.update(best_loss=val_loss, best_epoch=epoch + 1)
            save_model(model, tokenizer, out / 'model', config, revision)
        state.update(epoch=epoch + 1, batch=0, loss_sum=0.0, count=0)
        save_checkpoint(ckpt, model, optimizer, scheduler, scaler, state)
        last_save = time.perf_counter()
    # Thresholds come from the saved best epoch loaded as inference loads it, so final_evaluate sees the same scores.
    del optimizer, scheduler, scaler, model, trainable
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bundle = load_bundle(out / 'model', device, config['window_size'], config['model_version'])
    val_predictions = predict_examples(bundle, splits['validation'], eval_batch)
    targets = config.get('recall_targets', [0.95, 0.99])
    thresholds = select_thresholds(splits['validation'], val_predictions, targets)
    metadata = {'schema_version': 'model-bundle-v1', 'model_version': config['model_version'],
                'backbone': config['model_name'], 'revision': config.get('revision'), 'resolved_revision': revision,
                'adapter': 'lora', 'lora': config['lora'], 'bidirectional': bool(config.get('bidirectional')),
                'window_size': config['window_size'], 'seed': config['seed'], 'best_epoch': state['best_epoch'],
                'thresholds': thresholds, 'deployment_recall_target': config.get('deployment_recall_target', 0.99),
                'dataset_fingerprint': manifest['dataset_fingerprint'],
                'provisional_records': sum(e.provisional for e in examples),
                'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
                'python': platform.python_version(), 'device': str(device),
                'candidate_truncation_counts': truncation,
                'character_features': False, 'feature_dimension': 0, 'feature_spec': None,
                'threshold_status': 'synthetic_only' if config.get('synthetic_only', False) else 'validation_selected'}
    write_json(out / 'model' / 'bundle.json', metadata)
    bundle.threshold = thresholds[str(metadata['deployment_recall_target'])]
    for row in val_predictions:
        row['threshold'] = bundle.threshold
        row['action'] = 'MASK' if row['p_mask'] >= bundle.threshold else 'KEEP'
    test_predictions = (predict_examples(bundle, splits['test'], eval_batch)
                        if config.get('evaluate_test', True) else None)
    write_jsonl(out / 'validation_predictions.jsonl', val_predictions)
    if test_predictions is not None:
        write_jsonl(out / 'test_predictions.jsonl', test_predictions)
    report = {'notice': 'Provisional dataset pipeline validation; not final PrivAI performance.',
              'best_epoch': state['best_epoch'], 'threshold_source': 'validation', 'thresholds': thresholds,
              'validation': {t: evaluate(splits['validation'], val_predictions, v) for t, v in thresholds.items()},
              'test': ({t: evaluate(splits['test'], test_predictions, v) for t, v in thresholds.items()}
                       if test_predictions is not None else None)}
    write_json(out / 'evaluation.json', report)
    return report
