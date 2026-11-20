import argparse
import json
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description='Train credential candidate classifier')
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[], help='override a config key: key=value (YAML value)')
    parser.add_argument('--resume', action='store_true',
                        help='LoRA configs only: continue from the checkpoint in output_dir after an interruption')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    for item in args.set:
        key, value = item.split('=', 1)
        config[key] = yaml.safe_load(value)
    # Relative paths are project-root-relative, independent of shell CWD.
    root = config_path.parent.parent
    for field in ('data_path', 'output_dir', 'split_manifest', 'checkpoint_dir'):
        if config.get(field) and not Path(config[field]).is_absolute():
            config[field] = str(root/config[field])
    if config.get('lora'):  # a decoder with LoRA (Qwen): the back-end verifier's server stage
        from privai_model.verifier.lora import train as train_lora
        report = train_lora(config, resume=args.resume)
    elif args.resume:
        parser.error('--resume is for LoRA configs; a full fine-tune starts again in a new output_dir')
    else:
        from privai_model.training.trainer import train
        report = train(config)
    print(json.dumps({'best_epoch': report['best_epoch'], 'thresholds': report['thresholds']}))


if __name__ == '__main__':
    main()
