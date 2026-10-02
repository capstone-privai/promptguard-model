import argparse
import json
from pathlib import Path
import yaml
from privai_model.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description='Train credential candidate classifier')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    # Relative paths are project-root-relative, independent of shell CWD.
    root = config_path.parent.parent
    for field in ('data_path', 'output_dir', 'split_manifest'):
        if config.get(field) and not Path(config[field]).is_absolute():
            config[field] = str(root/config[field])
    report = train(config)
    print(json.dumps({'best_epoch': report['best_epoch'], 'thresholds': report['thresholds']}))


if __name__ == '__main__':
    main()
