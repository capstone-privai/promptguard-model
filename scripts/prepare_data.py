import argparse
import json
from privai_model.data.adapter import iter_jsonl, adapt_records
from privai_model.data.split import split_groups
from privai_model.io import write_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    records=list(iter_jsonl(args.data))
    examples=adapt_records(records)
    _, manifest = split_groups(examples, args.seed)
    manifest['load_statistics']={'total_records':len(records),'value_records':len(examples),
                                 'excluded_non_value_records':len(records)-len(examples)}
    write_json(args.output, manifest)
    print(json.dumps({'load':manifest['load_statistics'],**manifest['statistics']}, indent=2, ensure_ascii=False))
