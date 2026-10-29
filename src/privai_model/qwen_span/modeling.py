"""Qwen loading and tokenization shared by training and inference (torch required)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import format as fmt

DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}


def resolve_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable in this process (sandboxed shells often hide it)')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    return torch.device(name)


def load_tokenizer(name, revision=None, local_files_only=False):
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(name, dtype='float32', revision=None, local_files_only=False):
    if dtype not in DTYPES:
        raise ValueError('dtype must be one of ' + ', '.join(DTYPES))
    return AutoModelForCausalLM.from_pretrained(name, revision=revision, local_files_only=local_files_only,
                                                torch_dtype=DTYPES[dtype], attn_implementation='sdpa')


def end_token(tokenizer):
    """Qwen chat turns end with <|im_end|>; fall back to eos for other tokenizers."""
    token = '<|im_end|>'
    ids = tokenizer.convert_tokens_to_ids(token)
    if isinstance(ids, int) and ids != tokenizer.unk_token_id and ids is not None:
        return token, ids
    return tokenizer.eos_token, tokenizer.eos_token_id


def stop_ids(tokenizer):
    ids = {end_token(tokenizer)[1]}
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)


def prompt_ids(tokenizer, text):
    prompt = tokenizer.apply_chat_template(fmt.messages(text), tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, add_special_tokens=False)['input_ids']


def encode_example(tokenizer, text, answer):
    """Prompt and completion are tokenized separately so the loss boundary is exact
    and the prompt tokens equal those seen at inference time."""
    prompt = prompt_ids(tokenizer, text)
    completion = tokenizer(answer + end_token(tokenizer)[0], add_special_tokens=False)['input_ids']
    return dict(input_ids=prompt + completion, prompt_length=len(prompt))


def completion_loss(model, input_ids, prompt_length):
    """Mean next-token CE over completion tokens only.

    The vocabulary projection is applied only at the positions that predict completion
    tokens, so a 3k-token log input does not materialize a [T, 151k] logit matrix.
    This is numerically the same objective as full logits with prompt labels = -100.
    """
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hidden = base.model(input_ids=input_ids, use_cache=False).last_hidden_state
    predictors = hidden[:, prompt_length - 1:-1, :]
    targets = input_ids[:, prompt_length:]
    logits = base.lm_head(predictors).float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def reference_loss(model, input_ids, prompt_length):
    """Full-logit masked loss; used only by tests to check completion_loss."""
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    return model(input_ids=input_ids, labels=labels, use_cache=False).loss
