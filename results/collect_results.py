"""Copy small experiment evidence from code/runs/ into results/ (run from the repository root).

Copies metrics.json of every reported run, all cache grids, the final test JSON and selection
note, and converts the saved measure.py console output (*.txt) into JSON. Checkpoints are never
copied; they are published separately.
"""
import json
import shutil
from pathlib import Path

RUNS = Path('code/runs')
OUT = Path('results')

METRIC_RUNS = ['baseline', 'baseline-gpu-fp32', 'e1-scale', 'e2-modern', 'e3-modern-lr2e-3',
               'e4-long-drop10', 'e5-long-drop20', 'e7-long3x-drop10', 'e8-long3x-d8-drop10',
               'e9-long3x-drop05', 'baseline-long3x', 'abl-no_rope', 'abl-no_swiglu', 'abl-no_rms',
               'abl-with_bias', 'abl-no_scaledinit',
               'timing-scale_w256d6', 'timing-modern_w256d6', 'timing-modern_w256d5',
               'timing-modern_w192d6', 'timing-modern_w256d8']

# (run, saved console output, output name, note)
MEASUREMENTS = [
    ('timing-scale_w256d6', 'measure.txt', 'measure_validation_r3.json', ''),
    ('timing-modern_w256d6', 'measure.txt', 'measure_validation_r3.json', ''),
    ('timing-modern_w256d5', 'measure.txt', 'measure_validation_r3.json', ''),
    ('timing-modern_w192d6', 'measure.txt', 'measure_validation_r3.json', ''),
    ('timing-modern_w256d8', 'measure.txt', 'measure_validation_r3.json', ''),
    ('timing-modern_w256d8', 'measure_compare_r9_validation.txt', 'measure_validation_r9.json', 'stable environment'),
    ('e4-long-drop10', 'measure_compare_validation.txt', 'measure_validation_r5_unstable.json',
     'unstable environment (Balanced power plan, background apps); superseded'),
    ('e6-cache', 'measure_compare_validation.txt', 'measure_validation_r5_dense_unstable.json',
     'dense cache implementation, unstable environment; superseded'),
    ('e7-cache', 'measure_compare_validation.txt', 'measure_validation_r5_dense_unstable.json',
     'dense cache implementation, unstable environment; superseded'),
    ('e7-cache', 'measure_compare_r9_validation.txt', 'measure_validation_r9.json', 'sparse cache, stable environment'),
    ('e7-long3x-drop10', 'measure_compare_r9_validation.txt', 'measure_validation_r9.json', 'stable environment'),
    ('final', 'measure_validation_r9.txt', 'measure_validation_r9.json', 'final model, before test'),
    ('final', 'measure_test_r5.txt', 'measure_test_r5.json', 'final model, frozen, official test measurement'),
]

EXTRA = [('final', 'test.json'), ('final', 'SELECTION.md'), ('e7-cache', 'recheck_sparse.json'),
         ('e3-modern-lr2e-3', 'recheck_validation.json')]


def parse_measure(path):
    text = path.read_text(encoding='utf-8-sig', errors='replace')
    rounds = [json.loads(line[line.index('{'):]) for line in text.splitlines() if '"role"' in line]
    summary = json.loads(text[text.index('\n{\n'):])
    return rounds, summary


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


for run in METRIC_RUNS:
    copy(RUNS / run / 'metrics.json', OUT / run / 'metrics.json')
for grid in sorted(RUNS.glob('*/cache_grid_validation*.json')):
    copy(grid, OUT / grid.parent.name / grid.name)
for run, name in EXTRA:
    copy(RUNS / run / name, OUT / run / name)
for run, src, name, note in MEASUREMENTS:
    rounds, summary = parse_measure(RUNS / run / src)
    record = {'source': f'code/runs/{run}/{src}', 'note': note, 'rounds': rounds, 'summary': summary}
    (OUT / run).mkdir(parents=True, exist_ok=True)
    (OUT / run / name).write_text(json.dumps(record, indent=2) + '\n')
copy(Path('code/run_log.csv'), OUT / 'run_log.csv')
print('files:', sum(1 for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json', '.md', '.csv')))
