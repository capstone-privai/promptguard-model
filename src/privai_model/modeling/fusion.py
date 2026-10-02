"""Opt-in encoder + character features. OFF remains the original HF classifier."""
import json
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import save_file, load_file
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from privai_model.modeling.features import FeatureSpec


class FeatureFusionClassifier(nn.Module):
    def __init__(self, encoder, spec, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        self.feature_spec = spec
        self.dropout_rate = dropout
        self.text_norm = nn.LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size+spec.dimension, 2)

    def forward(self, input_ids, attention_mask=None, character_features=None, labels=None, **kwargs):
        if character_features is None or character_features.shape != (input_ids.shape[0], self.feature_spec.dimension):
            raise ValueError('character feature batch/dimension mismatch')
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs)
        hidden = outputs.last_hidden_state
        mask = (attention_mask if attention_mask is not None else torch.ones_like(input_ids)).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
        combined = torch.cat((self.text_norm(pooled), character_features.to(hidden.dtype)), dim=-1)
        logits = self.classifier(self.dropout(combined))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(logits=logits, loss=loss)

    def resize_token_embeddings(self, size):
        return self.encoder.resize_token_embeddings(size)

    def save_pretrained(self, path, safe_serialization=True):
        if not safe_serialization:
            raise ValueError('fusion checkpoint requires safetensors')
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path/'encoder', safe_serialization=True)
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items() if not k.startswith('encoder.')}
        save_file(state, path/'fusion.safetensors')
        metadata = {'schema_version':'fusion-v1', 'feature_spec':self.feature_spec.to_dict(),
                    'pooling':'attention_mask_mean', 'dropout':self.dropout_rate}
        (path/'fusion_config.json').write_text(json.dumps(metadata, indent=2)+'\n')

    @classmethod
    def from_pretrained(cls, path, attention_implementation='eager'):
        path = Path(path)
        meta = json.loads((path/'fusion_config.json').read_text())
        if meta['schema_version'] != 'fusion-v1' or meta['pooling'] != 'attention_mask_mean':
            raise ValueError('unsupported fusion checkpoint')
        encoder = AutoModel.from_pretrained(path/'encoder', local_files_only=True,
                                           attn_implementation=attention_implementation)
        model = cls(encoder, FeatureSpec.from_dict(meta['feature_spec']), meta['dropout'])
        state = load_file(path/'fusion.safetensors')
        expected = {k for k in model.state_dict() if not k.startswith('encoder.')}
        if set(state) != expected:
            raise ValueError('fusion head checkpoint incomplete')
        model.load_state_dict(state, strict=False)  # Encoder already loaded strictly by HF.
        return model
