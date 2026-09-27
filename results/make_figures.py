"""Make the report figures and the README summary table from results/ (run from the repository root).

Plotting only: needs matplotlib (`python -m pip install matplotlib`), which is NOT an evaluation
dependency. Every number is read from the JSON files in results/; nothing is typed in by hand.
Outputs: results/figures/*.png and results/summary_table.md.
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('results')
FIG = R / 'figures'
FIG.mkdir(exist_ok=True)


def load(run, name='metrics.json'):
    return json.loads((R / run / name).read_text(encoding='utf-8'))


def val_bpb(run):
    return load(run)['validation']['bpb']


def ratio(run, name):
    return load(run, name)['summary']['time_ratio_median']


def grid_best(run, name='cache_grid_validation_fine.json'):
    return load(run, name)['best']['validation_bpb']


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print('wrote', FIG / name)


# (a) validation curves
fig, ax = plt.subplots(figsize=(7, 4.2))
curves = [('e7-long3x-drop10', 'E7 d6 drop0.1 (14400)', '-'), ('e8-long3x-d8-drop10', 'E8 d8 drop0.1 (14400)', '-'),
          ('e9-long3x-drop05', 'E9 d6 drop0.05 (14400)', '-'), ('baseline-long3x', 'baseline, 14400 steps', '-'),
          ('e4-long-drop10', 'E4 d6 drop0.1 (4800)', '--'), ('e5-long-drop20', 'E5 d6 drop0.2 (4800)', '--')]
for run, label, style in curves:
    h = load(run)['validation_history']
    ax.plot([p['step'] for p in h], [p['bpb'] for p in h], style, marker='o', ms=3, label=label)
ax.set_xlabel('training step (32 x 256 targets per step)')
ax.set_ylabel('validation BPB (raw weights)')
ax.set_title('Validation curves')
ax.grid(alpha=.3)
ax.legend(fontsize=8)
save(fig, 'a_validation_curves.png')

# (b) equal training budget
runs = [('baseline', 'baseline\n(CPU)'), ('e1-scale', 'E1 scale\nw256 d6'), ('e2-modern', 'E2 modern\nlr 1e-3'),
        ('e3-modern-lr2e-3', 'E3 modern\nlr 2e-3')]
vals = [val_bpb(r) for r, _ in runs]
fig, ax = plt.subplots(figsize=(6, 3.8))
bars = ax.bar([l for _, l in runs], vals, color=['#888', '#6a8fc7', '#3b6fb6', '#1f4e8c'])
ax.bar_label(bars, fmt='%.4f', fontsize=8)
ax.set_ylim(min(vals) - .1, max(vals) + .05)
ax.set_ylabel('validation BPB')
ax.set_title('Same training targets (1200 x 32 x 256)')
save(fig, 'b_equal_budget.png')

# (c) ablations relative to E2
e2 = val_bpb('e2-modern')
abl = [('abl-no_rope', 'learned pos\n(no RoPE)'), ('abl-no_swiglu', 'GELU MLP\n(no SwiGLU)'),
       ('abl-no_scaledinit', 'no scaled\ninit'), ('abl-with_bias', 'with bias'), ('abl-no_rms', 'LayerNorm\n(no RMSNorm)')]
deltas = [val_bpb(r) - e2 for r, _ in abl]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
bars = ax.bar([l for _, l in abl], deltas, color=['#c0392b' if d > 0 else '#27ae60' for d in deltas])
ax.bar_label(bars, fmt='%+.4f', fontsize=8)
ax.axhline(0, color='k', lw=.8)
ax.set_ylabel('validation BPB minus E2 (%.4f)' % e2)
ax.set_title('Removing one component from E2 (single seed)')
save(fig, 'c_ablations.png')

# (d) cache grid heatmap for E7 (coarse grid)
g = load('e7-long3x-drop10', 'cache_grid_validation_coarse.json')
lams = sorted({p['cache_lambda'] for p in g['grid'] if p['cache_lambda'] > 0})
ths = sorted({p['cache_theta'] for p in g['grid'] if p['cache_lambda'] > 0})
cell = {(p['cache_lambda'], p['cache_theta']): p['validation_bpb'] for p in g['grid']}
data = [[cell[(l, t)] for t in ths] for l in lams]
fig, ax = plt.subplots(figsize=(6, 4.2))
im = ax.imshow(data, cmap='viridis_r', aspect='auto')
ax.set_xticks(range(len(ths)), [f'{t:g}' for t in ths])
ax.set_yticks(range(len(lams)), [f'{l:g}' for l in lams])
for i, row in enumerate(data):
    for j, v in enumerate(row):
        ax.text(j, i, f'{v:.4f}', ha='center', va='center', fontsize=7,
                color='white' if v > statistics.median(sum(data, [])) else 'black')
ax.set_xlabel('theta (cosine-kernel sharpness)')
ax.set_ylabel('lambda (cache weight)')
ax.set_title(f"E7 cache grid, validation BPB (cache off {g['cache_off_bpb']:.4f})")
fig.colorbar(im, ax=ax)
save(fig, 'd_cache_grid_e7.png')

# (e) quality vs evaluation cost
e7_ratio = ratio('e7-long3x-drop10', 'measure_validation_r9.json')
e7c_ratio = ratio('e7-cache', 'measure_validation_r9.json')
d8_ratio = ratio('timing-modern_w256d8', 'measure_validation_r9.json')
points = [
    ('baseline', 1.0, val_bpb('baseline'), False),
    ('E1 scale', ratio('timing-scale_w256d6', 'measure_validation_r3.json'), val_bpb('e1-scale'), False),
    ('E2 modern', ratio('timing-modern_w256d6', 'measure_validation_r3.json'), val_bpb('e2-modern'), False),
    ('E7', e7_ratio, val_bpb('e7-long3x-drop10'), False),
    ('E7 + cache (final)', e7c_ratio, grid_best('e7-long3x-drop10'), False),
    ('E8 d8', d8_ratio, val_bpb('e8-long3x-d8-drop10'), False),
    # e8-cache was not timed: estimate = d8 ratio + measured cache overhead (e7-cache - e7).
    ('E8 + cache (est.)', d8_ratio + e7c_ratio - e7_ratio, grid_best('e8-long3x-d8-drop10'), True),
]
offsets = {'E7 + cache (final)': (-8, -3), 'E8 + cache (est.)': (6, -4)}
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for name, x, y, est in points:
    ax.scatter(x, y, s=40, facecolors='none' if est else 'C0', edgecolors='C0')
    dx, dy = offsets.get(name, (5, 4))
    ax.annotate(name, (x, y), textcoords='offset points', xytext=(dx, dy), fontsize=8,
                ha='right' if dx < 0 else 'left')
ax.margins(y=.12)
ax.axvline(5, color='r', ls='--', lw=.8)
ax.text(4.95, ax.get_ylim()[1], 'limit 5x', color='r', ha='right', va='top', fontsize=8)
ax.set_xlabel('CPU evaluation time / baseline (validation, median of rounds)')
ax.set_ylabel('validation BPB')
ax.set_title('Quality vs evaluation cost')
ax.grid(alpha=.3)
save(fig, 'e_quality_vs_cost.png')

# README summary table
TABLE = [
    ('baseline', 'baseline', 'baseline (official, CPU)', ('abs', 1.0)),
    ('baseline-gpu-fp32', 'baseline', 'baseline on GPU fp32', ('abs', 1.0)),
    ('e1-scale', 'scale_w256d6', 'E1 scale only', ('timing-scale_w256d6', 'measure_validation_r3.json')),
    ('e2-modern', 'modern_w256d6', 'E2 modern arch', ('timing-modern_w256d6', 'measure_validation_r3.json')),
    ('e3-modern-lr2e-3', 'modern_w256d6', 'E3 + lr 2e-3', ('timing-modern_w256d6', 'measure_validation_r3.json')),
    ('e4-long-drop10', 'modern_w256d6_drop10', 'E4 4x steps', ('timing-modern_w256d6', 'measure_validation_r3.json')),
    ('e5-long-drop20', 'modern_w256d6_drop20', 'E5 4x steps', ('timing-modern_w256d6', 'measure_validation_r3.json')),
    ('e7-long3x-drop10', 'modern_w256d6_drop10', 'E7 12x steps', ('e7-long3x-drop10', 'measure_validation_r9.json')),
    ('e8-long3x-d8-drop10', 'modern_w256d8_drop10', 'E8 12x steps, depth 8', ('timing-modern_w256d8', 'measure_validation_r9.json')),
    ('e9-long3x-drop05', 'modern_w256d6_drop05', 'E9 12x steps', ('e7-long3x-drop10', 'measure_validation_r9.json')),
    ('baseline-long3x', 'baseline', 'baseline 12x steps', ('abs', 1.0)),
]
lines = ['| run | config | params | steps | training targets | lr | dropout | EMA | train s | val BPB | val BPB raw | eval time ratio |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for run, cfg, _, tr in TABLE:
    m = load(run)
    rec = m.get('recipe', {})
    raw = m.get('validation_raw_weights')
    t = tr[1] if tr[0] == 'abs' else ratio(*tr)
    lines.append(f"| {run} | {cfg} | {m['parameters']:,} | {m['history'][-1]['step']} | {m['train_tokens']:,} | "
                 f"{rec.get('lr', 0.001):g} | {m['config'].get('dropout', 0):g} | {rec.get('ema', 0) or '-'} | "
                 f"{m['train_seconds']:.0f} | {m['validation']['bpb']:.4f} | {raw['bpb']:.4f} | {t:.2f} |"
                 if raw else
                 f"| {run} | {cfg} | {m['parameters']:,} | {m['history'][-1]['step']} | {m['train_tokens']:,} | "
                 f"{rec.get('lr', 0.001):g} | {m['config'].get('dropout', 0):g} | {rec.get('ema', 0) or '-'} | "
                 f"{m['train_seconds']:.0f} | {m['validation']['bpb']:.4f} | - | {t:.2f} |")
for run, parent, label in [('e7-cache-fine', 'e7-long3x-drop10', 'E7 + cache (final)'),
                           ('e8-cache', 'e8-long3x-d8-drop10', 'E8 + cache')]:
    g = load(parent, 'cache_grid_validation_fine.json')
    t = e7c_ratio if parent.startswith('e7') else d8_ratio + e7c_ratio - e7_ratio
    lines.append(f"| {run} | {parent} + cache λ={g['best']['cache_lambda']:g} θ={g['best']['cache_theta']:g} | "
                 f"{load(parent)['parameters']:,} | – | – | – | – | – | 0 (no retraining) | "
                 f"{g['best']['validation_bpb']:.4f} | – | {t:.2f}{' (est.)' if parent.startswith('e8') else ''} |")
lines.append('')
lines.append('Ablations (1200 steps, lr 1e-3, seed 17; difference vs E2 = %.4f):' % e2)
lines.append('')
lines.append('| run | change | params | val BPB | Δ vs E2 |')
lines.append('|---|---|---:|---:|---:|')
for run, label in abl:
    m = load(run)
    lines.append(f"| {run} | {label.replace(chr(10), ' ')} | {m['parameters']:,} | {m['validation']['bpb']:.4f} | "
                 f"{m['validation']['bpb'] - e2:+.4f} |")
(R / 'summary_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('wrote', R / 'summary_table.md')
