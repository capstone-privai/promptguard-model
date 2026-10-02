"""Simple end-to-end torch loop; validation selects epoch, then thresholds; test once."""
import json
import platform
import math
from pathlib import Path
import torch
import transformers
from torch.utils.data import DataLoader, WeightedRandomSampler
from privai_model.data.adapter import load_examples
from privai_model.data.split import split_groups, restore_split
from privai_model.data.input_builder import encode_candidate
from privai_model.modeling.features import validate_feature_options, extract_features
from privai_model.modeling.model import create_model, seed_everything, resolve_device, load_checkpoint
from privai_model.inference.predict import ModelBundle, predict_examples
from privai_model.evaluation.evaluator import select_thresholds, evaluate
from privai_model.io import write_json, write_jsonl


def train(config):
    validate_feature_options(config)
    if config['epochs'] < 1 or config['batch_size'] < 1:
        raise ValueError('epochs and batch_size must be positive')
    if config.get('sampling', 'none') not in ('none', 'balanced'):
        raise ValueError('sampling must be none or balanced')
    weights = config.get('class_weights')
    if weights is not None and (len(weights) != 2 or any(not 0 < w < float('inf') for w in weights)):
        raise ValueError('class_weights must be two positive finite values [KEEP, MASK]')
    targets = config.get('recall_targets', [0.95, 0.99])
    if config.get('deployment_recall_target', 0.99) not in targets:
        raise ValueError('deployment recall target must appear in recall_targets')
    examples = load_examples(config['data_path'])
    if config.get('split_manifest'):
        manifest = json.loads(Path(config['split_manifest']).read_text())
        splits = restore_split(examples, manifest)
    else:
        splits, manifest = split_groups(examples, config['seed'], config['split_ratios'])
    # Never retry seeds based on test metrics or test class balance.
    for name in ('train', 'validation'):
        if {e.label for e in splits[name]} != {0, 1}:
            raise ValueError(f'{name} must contain both classes; review group/data design')
    output = Path(config['output_dir'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('output_dir is not empty; choose a new run directory')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'config.json', config)
    write_json(output/'split_manifest.json', manifest)
    print(json.dumps(manifest['statistics'], ensure_ascii=False))
    seed_everything(config['seed'], config.get('deterministic', True))
    device = resolve_device(config.get('device', 'auto'))
    tokenizer, model = create_model(config)
    resolved_revision = getattr(model.config, '_commit_hash', None)
    model.to(device)
    spec = getattr(model, 'feature_spec', None)
    # Encode only train/validation before fitting; test remains untouched until final evaluation.
    encoded = {}
    truncation_stats = {}
    for name in ('train', 'validation'):
        encoded[name] = []
        truncated = 0
        for ex in splits[name]:
            tokens, info = encode_candidate(ex, tokenizer, config['window_size'])
            encoded[name].append((tokens, ex.label, extract_features(ex.value, spec) if spec else None))
            truncated += info['candidate_truncated']
        truncation_stats[name] = truncated

    def collate(rows):
        batch = tokenizer.pad([row[0] for row in rows], padding=True, return_tensors='pt')
        batch['labels'] = torch.tensor([row[1] for row in rows], dtype=torch.long)
        if spec is not None:
            batch['character_features'] = torch.tensor([row[2] for row in rows], dtype=torch.float32)
        return batch

    generator = torch.Generator().manual_seed(config['seed'])
    sampler = None
    if config.get('sampling', 'none') == 'balanced':
        counts = [sum(e.label == y for e in splits['train']) for y in (0, 1)]
        sampler = WeightedRandomSampler([1/counts[e.label] for e in splits['train']], len(splits['train']), generator=generator)
    loaders = {name: DataLoader(rows, batch_size=config['batch_size'], collate_fn=collate,
                               shuffle=(name == 'train' and sampler is None),
                               sampler=sampler if name == 'train' else None, generator=generator)
               for name, rows in encoded.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device, dtype=torch.float32) if weights else None)
    val_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    best_loss, best_epoch, history = float('inf'), None, []
    checkpoint = output/'model'
    for epoch in range(1, config['epochs']+1):
        model.train()
        loss_sum = count = 0
        for batch in loaders['train']:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(**batch).logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()*len(labels)
            count += len(labels)
        model.eval()
        val_loss = 0
        with torch.inference_mode():
            for batch in loaders['validation']:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop('labels')
                val_loss += val_loss_fn(model(**batch).logits, labels).item()
        val_loss /= len(splits['validation'])
        if not math.isfinite(val_loss):
            raise ValueError('nonfinite validation loss')
        record = {'epoch': epoch, 'train_loss': loss_sum/count, 'validation_loss': val_loss}
        history.append(record)
        write_json(output/'history.json', history)
        print(json.dumps(record))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
    # Load saved best validation-loss checkpoint, never choose using test scores.
    del optimizer, model
    model = load_checkpoint(checkpoint).to(device)
    bundle = ModelBundle(tokenizer, model, device, config['window_size'], config['model_version'], 0.5)
    val_predictions = predict_examples(bundle, splits['validation'], config['batch_size'])
    thresholds = select_thresholds(splits['validation'], val_predictions, targets)
    metadata = {'schema_version': 'model-bundle-v1', 'model_version': config['model_version'],
                'backbone': config['model_name'], 'revision': config.get('revision'),
                'resolved_revision': resolved_revision,
                'window_size': config['window_size'], 'seed': config['seed'], 'best_epoch': best_epoch,
                'thresholds': thresholds, 'deployment_recall_target': config.get('deployment_recall_target', 0.99),
                'dataset_fingerprint': manifest['dataset_fingerprint'],
                'provisional_records': sum(e.provisional for e in examples),
                'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
                'python': platform.python_version(), 'device': str(device),
                'candidate_truncation_counts': truncation_stats,
                'character_features': spec is not None, 'feature_dimension': spec.dimension if spec else 0,
                'feature_spec': spec.to_dict() if spec else None,
                'threshold_status': 'synthetic_only' if config.get('synthetic_only', False) else 'validation_selected'}
    write_json(checkpoint/'bundle.json', metadata)
    bundle.threshold = thresholds[str(metadata['deployment_recall_target'])]
    # Serialize validation actions with the selected deployment threshold.
    for row in val_predictions:
        row['threshold'] = bundle.threshold
        row['action'] = 'MASK' if row['p_mask'] >= bundle.threshold else 'KEEP'
    test_predictions = (predict_examples(bundle, splits['test'], config['batch_size'])
                        if config.get('evaluate_test', True) else None)
    write_jsonl(output/'validation_predictions.jsonl', val_predictions)
    if test_predictions is not None:
        write_jsonl(output/'test_predictions.jsonl', test_predictions)
    report = {'notice': 'Provisional dataset pipeline validation; not final PrivAI performance.',
              'best_epoch': best_epoch, 'threshold_source': 'validation', 'thresholds': thresholds,
              'validation': {target: evaluate(splits['validation'], val_predictions, t) for target, t in thresholds.items()},
              'test': ({target: evaluate(splits['test'], test_predictions, t) for target, t in thresholds.items()}
                       if test_predictions is not None else None)}
    write_json(output/'evaluation.json', report)
    return report
