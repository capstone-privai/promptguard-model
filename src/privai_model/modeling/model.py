import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel
from privai_model.data.input_builder import register_markers
from privai_model.modeling.features import feature_spec, validate_feature_options
from privai_model.modeling.fusion import FeatureFusionClassifier
from pathlib import Path


def seed_everything(seed, deterministic=True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False


def resolve_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable on this machine')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable on this machine')
    return torch.device(name)


def create_model(config):
    validate_feature_options(config)
    name = config['model_name']
    kwargs = {'local_files_only': config.get('local_files_only', False)}
    # Revision pins belong to one backbone; avoid accidentally applying the Ettin SHA
    # when the user changes only model_name in its config.
    if config.get('revision') and config.get('revision_model_name', name) == name:
        kwargs['revision'] = config['revision']
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    register_markers(tokenizer)
    model_config = AutoConfig.from_pretrained(name, **kwargs)
    model_config.num_labels = 2
    model_config.id2label = {0: 'KEEP', 1: 'MASK'}
    model_config.label2id = {'KEEP': 0, 'MASK': 1}
    model_config.problem_type = 'single_label_classification'
    # ModernBERT compilation is optional; disable it for portable CPU/MPS operation.
    if hasattr(model_config, 'reference_compile'):
        model_config.reference_compile = False
    model_class = AutoModel if config.get('character_features', False) else AutoModelForSequenceClassification
    model = model_class.from_pretrained(
        name, config=model_config, attn_implementation=config.get('attention_implementation', 'eager'), **kwargs)
    if config.get('character_features', False):
        model = FeatureFusionClassifier(model, feature_spec(config))
    model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token_id is None:
        raise ValueError('backbone tokenizer must define a pad token')
    limit = getattr(model.config, 'max_position_embeddings', None)
    if limit and config['window_size'] > limit:
        raise ValueError('window_size exceeds backbone position limit')
    return tokenizer, model


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def load_checkpoint(path, attention_implementation='eager'):
    if (Path(path)/'fusion_config.json').exists():
        return FeatureFusionClassifier.from_pretrained(path, attention_implementation)
    return AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True, attn_implementation=attention_implementation)
