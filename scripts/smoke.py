"""Synthetic pipeline verification only. Never interpreted as credential performance."""
import argparse
import gzip
import json
from pathlib import Path
import torch
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, ModernBertConfig, ModernBertForSequenceClassification
from privai_model.io import write_json
from privai_model.training.trainer import train


def prepare(root, real_ettin=False):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('smoke directory must be empty')
    root.mkdir(parents=True, exist_ok=True)
    data = root/'synthetic_candidates_v1.jsonl.gz'
    # Both classes within every group make the synthetic smoke robust to group split.
    with gzip.open(data, 'wt', encoding='utf-8') as handle:
        for group in range(20):
            for label in (0,1):
                value = 'synthetic-fixture-credential' if label else 'YOUR_TEST_KEY'
                line = 'TOKEN='+value
                record = {'schema_version':'candidate-v1','candidate_id':f'synthetic:{group}:{label}',
                    'group_id':f'synthetic-group:{group}','granularity':'value','value':value,
                    'line':{'text':line,'value_start':6,'value_end':len(line)},
                    'context':{'before':['synthetic only'], 'after':['never a real credential']},
                    'source':{'dataset':'synthetic-smoke-only','channel':'tool_output','file_path':'fixture.env'},
                    'type':'TOKEN','detector':{'name':'synthetic-fixture'},
                    'label':{'action':'MASK' if label else 'KEEP','reason':'synthetic plumbing test', 'provisional':True}}
                handle.write(json.dumps(record)+'\n')
        handle.write(json.dumps({'granularity':'line'})+'\n')
    model_path = 'jhu-clsp/ettin-encoder-68m'
    if not real_ettin:
        model_path = str(root/'tiny-modernbert')
        vocab = {x:i for i,x in enumerate(['[PAD]','[UNK]','[CLS]','[SEP]','TOKEN','=','synthetic','fixture','credential','YOUR_TEST_KEY','-'])}
        backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]',special_tokens=[('[CLS]',2),('[SEP]',3)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',unk_token='[UNK]',cls_token='[CLS]',sep_token='[SEP]')
        tokenizer.save_pretrained(model_path)
        torch.manual_seed(42)
        tiny = ModernBertConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=1024,
            local_attention=32, global_attn_every_n_layers=2, pad_token_id=0, cls_token_id=2, sep_token_id=3,
            reference_compile=False, num_labels=2, id2label={0:'KEEP',1:'MASK'},label2id={'KEEP':0,'MASK':1})
        tiny._attn_implementation = 'eager'
        ModernBertForSequenceClassification(tiny).save_pretrained(model_path)
    config = {'model_name':model_path,'model_version':'SYNTHETIC-ettin68m-smoke' if real_ettin else 'SYNTHETIC-tiny-modernbert-smoke',
        'data_path':str(data),'output_dir':str(root/'run'),'seed':42,'split_ratios':[.8,.1,.1],
        'window_size':64,'batch_size':4,'epochs':1,'learning_rate':2e-5,'weight_decay':.01,
        'device':'cpu','attention_implementation':'eager','local_files_only':not real_ettin,
        'synthetic_only':True,'deterministic':True,'sampling':'none','class_weights':None,'character_features':False,
        'recall_targets':[.95,.99],'deployment_recall_target':.99}
    (root/'smoke.yaml').write_text(yaml.safe_dump(config))
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/smoke')
    parser.add_argument('--real-ettin', action='store_true', help='Download and fine-tune actual Ettin on synthetic examples')
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = prepare(args.output,args.real_ettin)
    report = train(config)
    print('Synthetic pipeline run completed; NOT real credential performance.')
