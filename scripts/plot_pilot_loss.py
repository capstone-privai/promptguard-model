"""Loss curves for the step-0 pilot: training loss from the run log, validation F2 from history.json.

Writes one self-contained SVG (two stacked panels, no second y-axis) and a CSV of the plotted points.
Usage: plot_pilot_loss.py OUT_STEM [--note=TEXT] LABEL=RUN_DIR:LOG ... (RUN_DIR holds history.json and model/span_bundle.json)
"""
import json
import math
import sys
from pathlib import Path

COLORS = [('#2a78d6', '#3987e5'), ('#eb6834', '#d95926'), ('#1baf7a', '#199e70'), ('#eda100', '#c98500')]
W, PAD_L, PAD_R, PANEL_H, GAP, TOP = 820, 64, 190, 220, 70, 48


def parse(spec):
    label, rest = spec.split('=', 1)
    run, log = rest.split(':', 1)
    steps = []
    for line in Path(log).read_text(errors='replace').splitlines():
        if line.startswith('{"event": "training_progress"'):
            e = json.loads(line)
            steps.append((e['epoch'] - 1 + e['batch'] / e['batches'], e['mean_loss']))
    history = json.loads((Path(run) / 'history.json').read_text())
    best = json.loads((Path(run) / 'model/span_bundle.json').read_text())['best_epoch']
    return dict(label=label, steps=steps, history=history, best=best)


def main():
    out, specs = Path(sys.argv[1]), sys.argv[2:]
    note = next((a.split('=', 1)[1] for a in specs if a.startswith('--note=')), None)
    specs = [a for a in specs if not a.startswith('--note=')]
    runs = [parse(s) for s in specs]
    epochs = max(max(h['epoch'] for h in r['history']) for r in runs)
    losses = [l for r in runs for _, l in r['steps']] + [h['loss'] for r in runs for h in r['history']]
    lo, hi = math.floor(math.log10(min(losses))), math.ceil(math.log10(max(losses)))
    f2s = [h['validation']['f2'] or 0 for r in runs for h in r['history']]
    f_lo, f_hi = math.floor(min(f2s) * 50) / 50, math.ceil(max(f2s) * 50) / 50
    if f_hi - f_lo < .04:
        f_hi = f_lo + .04
    plot_w = W - PAD_L - PAD_R
    x = lambda e: PAD_L + plot_w * e / epochs
    y1 = lambda v: TOP + PANEL_H * (hi - math.log10(v)) / (hi - lo)
    top2 = TOP + PANEL_H + GAP
    y2 = lambda v: top2 + PANEL_H * (f_hi - v) / (f_hi - f_lo)
    h_total = top2 + PANEL_H + 56
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h_total}" width="{W}" font-family="system-ui, sans-serif" font-size="12">',
         '<style>svg{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e0;'
         + ''.join(f'--s{i}:{c[0]};' for i, c in enumerate(COLORS)) + '}'
         '@media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#33332f;'
         + ''.join(f'--s{i}:{c[1]};' for i, c in enumerate(COLORS)) + '}}'
         'text{fill:var(--ink2)} .t{fill:var(--ink);font-size:14px;font-weight:600} .g{stroke:var(--grid)}</style>',
         f'<rect width="{W}" height="{h_total}" fill="var(--surface)"/>',
         f'<text class="t" x="{PAD_L}" y="22">Training loss (mean within each epoch, log scale)</text>',
         f'<text class="t" x="{PAD_L}" y="{top2 - 26}">Validation exact-span F2 by epoch (ring = selected epoch)</text>']
    if note:
        s.append(f'<text x="{PAD_L}" y="{top2 - 10}">{note}</text>')
    for d in range(lo, hi + 1):
        yy = y1(10 ** d)
        s.append(f'<line class="g" x1="{PAD_L}" x2="{W - PAD_R}" y1="{yy:.1f}" y2="{yy:.1f}"/><text x="{PAD_L - 8}" y="{yy + 4:.1f}" text-anchor="end">1e{d}</text>')
    steps_f = round((f_hi - f_lo) / .02)
    for k in range(steps_f + 1):
        v = f_lo + k * .02
        s.append(f'<line class="g" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y2(v):.1f}" y2="{y2(v):.1f}"/><text x="{PAD_L - 8}" y="{y2(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    for e in range(epochs + 1):
        for base, bottom in ((TOP, TOP + PANEL_H), (top2, top2 + PANEL_H)):
            s.append(f'<line class="g" x1="{x(e):.1f}" x2="{x(e):.1f}" y1="{base}" y2="{bottom}" stroke-dasharray="2 4"/>')
        s.append(f'<text x="{x(e):.1f}" y="{top2 + PANEL_H + 18}" text-anchor="middle">{e}</text>')
    s.append(f'<text x="{PAD_L + plot_w / 2}" y="{top2 + PANEL_H + 38}" text-anchor="middle">epoch</text>')
    rows = ['model,kind,epoch,value']
    for i, r in enumerate(runs):
        c = f'var(--s{i})'
        pts = ' '.join(f'{x(e):.1f},{y1(l):.1f}' for e, l in r['steps'])
        s.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{pts}"/>')
        if r['steps']:
            e, l = r['steps'][-1]
            s.append(f'<text x="{x(e) + 6:.1f}" y="{y1(l) + 4 + 14 * i:.1f}" style="fill:var(--ink)">{r["label"]}</text>')
        rows += [f'{r["label"]},train_loss,{e:.4f},{l:.6g}' for e, l in r['steps']]
        pts = ' '.join(f'{x(h["epoch"]):.1f},{y2(h["validation"]["f2"] or 0):.1f}' for h in r['history'])
        s.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{pts}"/>')
        for h in r['history']:
            v = h['validation']
            cx, cy = x(h['epoch']), y2(v['f2'] or 0)
            tip = f'{r["label"]} epoch {h["epoch"]}: F2 {v["f2"]:.4f}, precision {v["precision"]:.3f}, TP/FP/FN {v["tp"]}/{v["fp"]}/{v["fn"]}, cutoff {h["threshold"]}'
            s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{c}" stroke="var(--surface)" stroke-width="2"><title>{tip}</title></circle>')
            if h['epoch'] == r['best']:
                s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="none" stroke="{c}" stroke-width="2"/>')
            rows.append(f'{r["label"]},validation_f2,{h["epoch"]},{v["f2"]:.6f}')
        last = r['history'][-1]
        s.append(f'<text x="{x(last["epoch"]) + 10:.1f}" y="{y2(last["validation"]["f2"] or 0) + 4 + 14 * i:.1f}" style="fill:var(--ink)">{r["label"]}</text>')
    lx = W - PAD_R + 16
    for i, r in enumerate(runs if len(runs) > 1 else []):  # one series: the title and direct label name it
        s.append(f'<rect x="{lx}" y="{TOP - 30 + 18 * i}" width="12" height="3" rx="1.5" fill="var(--s{i})"/><text x="{lx + 18}" y="{TOP - 30 + 18 * i + 5}" style="fill:var(--ink)">{r["label"]}</text>')
    s.append('</svg>')
    out.with_suffix('.svg').write_text('\n'.join(s) + '\n')
    out.with_suffix('.csv').write_text('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
