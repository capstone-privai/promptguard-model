"""Build a concise, source-free report from the frozen comparison artifact."""
import json
from pathlib import Path


def main():
    report=json.loads(Path('reports/ml_span/context-v5-comparison.json').read_text())
    chosen=report['validation_decision']['selected']
    fresh=report['datasets']['fresh_test'];old=fresh['previous'];new=fresh[chosen]
    lines=['# Context refinement results','',
        'All results below are independent synthetic families or explicitly labelled development checks. This is not evaluation of the deployed PromptGuard ensemble.',
        '',f"Validation-selected checkpoint: `{report['validation_decision']['checkpoint']}`.",
        '','## Fresh-family test','',f"{new['records']} records; {new['gold_spans']} gold spans; {len(new['by_family_id'])} families.",'',
        '| Method | TP | FP | FN | Exact precision | Exact recall |', '|---|---:|---:|---:|---:|---:|']
    def pct(value):return 'undefined' if value is None else f'{value*100:.2f}%'
    def row(name,m):return f"| {name} | {m['tp']} | {m['fp']} | {m['fn']} | {pct(m['precision'])} | {pct(m['recall'])} |"
    lines.append(row('Rule',new['rule_only']))
    for name,result in fresh.items():lines.append(row(name,result['ml_only']))
    lines.append(row('Rule + selected ML candidate union',new['candidate_union']))
    lines += ['',f"Selected ML adds {new['ml_added_true_positives']} exact true positives to Rule and induces {new['ml_induced_false_positives']} false positives.",
        f"Exact Rule-miss recall: {pct(new['rule_miss_recall'])} ({new['rule_miss_count']} misses). No-overlap Rule-miss recall: {pct(new['rule_no_overlap_miss_recall'])} ({new['rule_no_overlap_miss_count']} misses). Coverage Rule-miss recall: {pct(new['rule_coverage_miss_recall'])} ({new['rule_coverage_miss_count']} misses).",
        f"Compared with previous ML: TP delta {new['ml_only']['tp']-old['ml_only']['tp']:+d}; FP delta {new['ml_only']['fp']-old['ml_only']['fp']:+d}.",
        '', '## Family-level results','', '| Family | Previous TP / FP | Selected TP / FP | Gold |', '|---|---:|---:|---:|']
    for family,r in new['by_family_id'].items():
        before=old['by_family_id'][family]['ml_only'];after=r['ml_only']
        lines.append(f"| {family} | {before['tp']} / {before['fp']} | {after['tp']} / {after['fp']} | {r['gold_spans']} |")
    lines += ['', '## Previously examined development sets','', '| Set | Previous TP / FP / FN | Selected TP / FP / FN |', '|---|---:|---:|']
    for name in ['v3_development','v1_development','original_development']:
        r=report['datasets'][name];a=r['previous']['ml_only'];b=r[chosen]['ml_only']
        lines.append(f"| {name} | {a['tp']} / {a['fp']} / {a['fn']} | {b['tp']} / {b['fp']} / {b['fn']} |")
    negatives=report['datasets']['negative_challenge'];a=negatives['previous'];b=negatives[chosen]
    lines += ['', '## Unseen negative-only stress challenge','',
        f"{b['records']} rows, zero gold secrets. Previous ML: {a['ml_only']['fp']} false-positive spans across {a['error_summary']['negative_rows_with_findings']} rows. Selected ML: {b['ml_only']['fp']} false-positive spans across {b['error_summary']['negative_rows_with_findings']} rows.",
        'This set was not used for training or selection. Recall and precision with no positive findings are undefined, not 100%.',
        '', '## Runtime and limitations','',
        f"Sequential fresh-test mean ML latency (warm model, first scan included): previous {old['ml_latency']['mean']:.2f} ms/input, selected {new['ml_latency']['mean']:.2f} ms/input. Source/length mix matters; these means are not comparable to older short-input tests.",
        'Small family counts and synthetic labels limit generalization. The boundary model only contracts existing candidates; it cannot recover absent candidates, expand truncated spans or merge fragmented multiline predictions. The local Rule proxy uses CredSweeper only and category-based file suffixes; it does not reproduce deployed per-detector ballots or exact original source-file configuration.',
        'Validation was reused for iterative development. Test results must not be used for further tuning while retaining a hold-out claim.',
        '', 'Detailed machine-readable results: `reports/ml_span/context-v5-comparison.json`. Training/design: [Context v4/v5](ML_C_CONTEXT_V4.md).']
    Path('docs/ml_span/ML_C_CONTEXT_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(selected=chosen,checkpoint=report['validation_decision']['checkpoint'],fresh_previous=old['ml_only'],fresh_selected=new['ml_only'],rule=new['rule_only'],added_tp=new['ml_added_true_positives'],added_fp=new['ml_induced_false_positives'],negative_fp=b['ml_only']['fp'])))
if __name__=='__main__':main()
