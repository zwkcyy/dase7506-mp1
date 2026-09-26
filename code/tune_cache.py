"""Tune the within-window cache of a trained checkpoint on the VALIDATION split (no retraining).

Usage:
  python tune_cache.py --checkpoint runs/e4-long-drop10/checkpoint.pt
  python tune_cache.py --checkpoint runs/e4-long-drop10/checkpoint.pt --save-best runs/e6-cache/checkpoint.pt

Writes <checkpoint dir>/cache_grid_validation.json. With --save-best, writes a new checkpoint
with identical weights whose config carries the best (cache_lambda, cache_theta); evaluate.py
then builds the cached model from that config. Never tune on the test split.
"""
import argparse
import itertools
import json
from pathlib import Path
import torch
from common import load_data, make_model, setup
from evaluate import score

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True, type=Path)
p.add_argument('--device', default='cuda')
p.add_argument('--lambdas', default='0,0.02,0.05,0.1,0.15,0.2,0.3')
p.add_argument('--thetas', default='0,5,10,20,40,80')
p.add_argument('--save-best', type=Path)
args = p.parse_args()
device, _ = setup(args.device, 'fp32', 4)
checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
model.load_state_dict(checkpoint['model'])
data = load_data()
lambdas = [float(v) for v in args.lambdas.split(',')]
thetas = [float(v) for v in args.thetas.split(',')]
rows = []
for lam, theta in itertools.product(lambdas, thetas):
    if lam == 0 and theta != thetas[0]:
        continue  # theta is irrelevant when the cache is off
    model.cache_lambda, model.cache_theta = lam, theta
    result = score(model, *data['validation'], device, 'fp32')
    rows.append({'cache_lambda': lam, 'cache_theta': theta, 'validation_bpb': result['bpb']})
    print(json.dumps(rows[-1]), flush=True)
best = min(rows, key=lambda r: r['validation_bpb'])
off = next(r for r in rows if r['cache_lambda'] == 0)
report = {'checkpoint': str(args.checkpoint), 'best': best, 'cache_off_bpb': off['validation_bpb'],
          'gain_bpb': off['validation_bpb'] - best['validation_bpb'], 'grid': rows}
(args.checkpoint.parent / 'cache_grid_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'grid'}, indent=2))
if args.save_best:
    if args.save_best.exists():
        raise SystemExit(f'{args.save_best} already exists; choose a new path.')
    args.save_best.parent.mkdir(parents=True, exist_ok=True)
    checkpoint['config'] = {**checkpoint['config'], 'cache_lambda': best['cache_lambda'],
                            'cache_theta': best['cache_theta']}
    checkpoint['cache_tuned_from'] = str(args.checkpoint)
    torch.save(checkpoint, args.save_best)
    print('saved', args.save_best)
