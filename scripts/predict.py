import argparse
from privai_model.data.adapter import iter_jsonl
from privai_model.inference.predict import load_model, predict_candidates
from privai_model.io import write_jsonl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--recall-target', type=float, default=None)
    args = parser.parse_args()
    bundle = load_model(args.model, args.device, args.recall_target)
    rows = predict_candidates(bundle, iter_jsonl(args.data), args.batch_size)
    write_jsonl(args.output, rows)
    print(f'Wrote {len(rows)} prediction-v1 records')
