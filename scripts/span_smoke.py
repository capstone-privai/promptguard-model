"""Offline tiny ModernBERT forward/backward/save/reload through the real span pipeline."""
import argparse
from pathlib import Path
from tokenizers import Tokenizer,models,pre_tokenizers,processors
from transformers import PreTrainedTokenizerFast,ModernBertConfig,ModernBertForTokenClassification
from privai_model.ml_span.bio import LABELS
from privai_model.ml_span.training import train
from privai_model.modeling.model import seed_everything
from span_pipeline import prepare


def tiny_model(path):
    seed_everything(42)
    alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789=!:;/_.-(){}[]"\n 가나접속암호번호인증비밀입니다사용공개추적값'
    vocab={v:i for i,v in enumerate(['[PAD]','[UNK]','[CLS]','[SEP]']+sorted(set(alphabet)))}
    backend=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'))
    backend.pre_tokenizer=pre_tokenizers.Split('',behavior='isolated')
    backend.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',special_tokens=[('[CLS]',2),('[SEP]',3)])
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',unk_token='[UNK]',cls_token='[CLS]',sep_token='[SEP]')
    tokenizer.save_pretrained(path)
    cfg=ModernBertConfig(vocab_size=len(tokenizer),hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,max_position_embeddings=512,local_attention=32,global_attn_every_n_layers=2,
        pad_token_id=0,cls_token_id=2,sep_token_id=3,reference_compile=False,num_labels=3,
        id2label=LABELS,label2id={v:k for k,v in LABELS.items()})
    ModernBertForTokenClassification(cfg).save_pretrained(path)
    return tokenizer


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    tiny_model(root/'backbone');prepare(root/'data',per_family=1)
    return train(dict(model_name=str(root/'backbone'),data_dir=str(root/'data'),output_dir=str(root/'run'),
        seed=42,epochs=1,batch_size=4,learning_rate=3e-5,max_length=128,stride=32,device='cpu',threads=2,
        thresholds=[.5],local_files_only=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='runs/span-smoke-v1');args=p.parse_args();run(args.output)
