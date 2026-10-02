import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
from privai_model.data.input_builder import register_markers


def candidate(i=0, action='MASK', group=None, value='synthetic-token'):
    return {'schema_version': 'candidate-v1', 'candidate_id': f'synthetic:{i}',
            'group_id': group or f'g{i}', 'granularity': 'value', 'value': value,
            'line': {'text': 'KEY='+value, 'value_start': 4, 'value_end': 4+len(value)},
            'context': {'before': ['environment synthetic'], 'after': ['end synthetic']},
            'type': 'TOKEN', 'source': {'dataset': 'synthetic-test-only', 'channel': 'tool_output', 'file_path':'demo.env'},
            'detector': {'name':'synthetic', 'rule_id':'token'},
            'label': {'action':action, 'provisional':True, 'reason':'synthetic fixture'}}


@pytest.fixture
def record():
    return candidate()


@pytest.fixture
def tokenizer():
    vocab = {x:i for i,x in enumerate(['[PAD]','[UNK]','[CLS]','[SEP]', 'synthetic','token', 'KEY', '=', '-', 'x', 'tail'])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]',
                                                           special_tokens=[('[CLS]',2),('[SEP]',3)])
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]',
                                 cls_token='[CLS]', sep_token='[SEP]', model_max_length=2048)
    register_markers(tok)
    return tok
