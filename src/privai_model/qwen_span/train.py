"""LoRA SFT with completion-only loss, batch-1 accumulation and resumable checkpoints.

Nothing is truncated: rows longer than max_length stop the run before training, so a
secret is never learned from a clipped answer. Logs carry counts, never text.
"""
import json
import math
import random
import time
from pathlib import Path
import torch
from privai_model.io import write_json
from . import format as fmt
from .modeling import load_model, load_tokenizer, resolve_device, encode_example, completion_loss


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def read_sft(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    for row in rows:
        roles = [m['role'] for m in row['messages']]
        if roles != ['system', 'user', 'assistant'] or row['messages'][0]['content'] != fmt.SYSTEM_PROMPT:
            raise ValueError('unexpected SFT record layout; rerun prepare')
    return rows


def text_of(record):
    user = record['messages'][1]['content']
    prefix, suffix = '<text>\n', '\n</text>'
    if not (user.startswith(prefix) and user.endswith(suffix)):
        raise ValueError('unexpected user prompt')
    return user[len(prefix):-len(suffix)]


def encode_all(tokenizer, records):
    encoded = []
    for record in records:
        example = encode_example(tokenizer, text_of(record), record['messages'][2]['content'])
        example['id'] = record['id']
        encoded.append(example)
    return encoded


def length_report(encoded, max_length):
    lengths = sorted(len(e['input_ids']) for e in encoded)
    over = [e['id'] for e in encoded if len(e['input_ids']) > max_length]
    return dict(rows=len(lengths), max_tokens=lengths[-1], p95_tokens=lengths[int(.95 * (len(lengths) - 1))],
                mean_tokens=sum(lengths) / len(lengths), max_length=max_length,
                over_limit=len(over), over_limit_ids=over,
                max_completion_tokens=max(len(e['input_ids']) - e['prompt_length'] for e in encoded))


def epoch_order(n, seed, epoch):
    order = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(order)
    return order


def lora_model(model, config):
    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(r=config['lora_r'], lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
                      target_modules=config['lora_target_modules'], bias='none', task_type='CAUSAL_LM')
    return get_peft_model(model, lora)


def save_checkpoint(model, optimizer, scheduler, state, directory):
    directory = Path(directory)
    tmp = directory.with_name(directory.name + '.tmp')
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp)
    torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), tmp / 'trainer_state.pt')
    write_json(tmp / 'state.json', state)
    if directory.exists():
        old = directory.with_name(directory.name + '.old')
        directory.rename(old)
        tmp.rename(directory)
        for p in sorted(old.rglob('*'), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        old.rmdir()
    else:
        tmp.rename(directory)


def train(config, resume=False):
    seed_everything(config['seed'])
    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.get('device', 'auto'))
    tokenizer = load_tokenizer(config['model_name'], config.get('revision'), config.get('local_files_only', False))
    records = read_sft(Path(config['sft_dir']) / 'train.jsonl')
    if config.get('limit'):
        records = records[:config['limit']]
    encoded = encode_all(tokenizer, records)
    lengths = length_report(encoded, config['max_length'])
    write_json(out / 'lengths.json', lengths)
    if lengths['over_limit']:
        raise ValueError(f"{lengths['over_limit']} rows exceed max_length={config['max_length']}; "
                         'raise max_length instead of truncating (see lengths.json)')

    model = load_model(config['model_name'], config.get('dtype', 'float32'), config.get('revision'),
                       config.get('local_files_only', False))
    model.config.use_cache = False
    if config.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = lora_model(model, config)
    # Trainable LoRA weights stay float32 even when the frozen base is half precision.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    accumulation = config['gradient_accumulation']
    steps_per_epoch = math.ceil(len(encoded) / accumulation)
    total_steps = steps_per_epoch * config['epochs']
    warmup = max(1, int(config.get('warmup_ratio', 0.05) * total_steps))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0.0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup,
        max(0.0, 0.5 * (1 + math.cos(math.pi * max(0, s + 1 - warmup) / max(1, total_steps - warmup))))))

    state = dict(epoch=0, example=0, step=0, history=[])
    last = out / 'last'
    if resume and (last / 'state.json').exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(last / 'adapter_model.safetensors'), device=str(device)))
        saved = torch.load(last / 'trainer_state.pt', map_location=device)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        state = json.loads((last / 'state.json').read_text())

    write_json(out / 'run.json', dict(config=config, device=str(device), trainable_parameters=trainable,
        total_parameters=total, steps_per_epoch=steps_per_epoch, total_steps=total_steps, warmup_steps=warmup,
        lengths={k: v for k, v in lengths.items() if k != 'over_limit_ids'}))
    progress = out / 'progress.jsonl'
    model.train()
    began = time.time()
    for epoch in range(state['epoch'], config['epochs']):
        order = epoch_order(len(encoded), config['seed'], epoch)
        window_loss, window_n, tokens = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        start = state['example'] if epoch == state['epoch'] else 0
        for position in range(start, len(order)):
            example = encoded[order[position]]
            ids = torch.tensor([example['input_ids']], device=device)
            loss = completion_loss(model, ids, example['prompt_length'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at epoch {epoch} example {position}; try dtype float32')
            group = min(accumulation, len(order) - (position // accumulation) * accumulation)
            (loss / group).backward()
            window_loss += loss.item()
            window_n += 1
            tokens += len(example['input_ids'])
            if (position + 1) % accumulation == 0 or position + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(params, config.get('max_grad_norm', 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state['step'] += 1
                state['example'] = position + 1
                if state['step'] % config.get('log_every', 5) == 0 or position + 1 == len(order):
                    entry = dict(epoch=epoch, step=state['step'], total_steps=total_steps,
                                 examples=position + 1, loss=window_loss / window_n,
                                 lr=scheduler.get_last_lr()[0], tokens=tokens,
                                 elapsed_s=round(time.time() - began, 1))
                    if device.type == 'mps':
                        entry['mps_allocated_gb'] = round(torch.mps.current_allocated_memory() / 2**30, 2)
                    with progress.open('a', encoding='utf-8') as f:
                        f.write(json.dumps(entry) + '\n')
                    print(json.dumps(entry), flush=True)
                    window_loss, window_n = 0.0, 0
                if state['step'] % config.get('save_every', 25) == 0:
                    save_checkpoint(model, optimizer, scheduler, state, last)
        state['history'].append(dict(epoch=epoch, steps=state['step'], elapsed_s=round(time.time() - began, 1)))
        model.save_pretrained(out / f'epoch-{epoch + 1}')
        state['epoch'] = epoch + 1
        state['example'] = 0
        save_checkpoint(model, optimizer, scheduler, state, last)
    write_json(out / 'training.json', dict(state=state, adapters=[f'epoch-{e + 1}' for e in range(config['epochs'])]))
    return out
