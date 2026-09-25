"""Measure the three MP1 evaluation limits for one checkpoint (run from code/).

Runs the unmodified evaluate.py on CPU/FP32 several times and reports:
  - scoring seconds (the 'seconds' field written by evaluate.py, excludes loading)
  - peak resident memory of the whole evaluate.py process tree (polled every 20 ms;
    on Windows the venv python.exe is a launcher that spawns the real interpreter)
  - checkpoint size in MiB (uncompressed inference asset)

With --compare, the --checkpoint (baseline) and --compare (candidate) are run
alternately (baseline, candidate, baseline, candidate, ...) so both see the same
machine conditions, and each round's candidate/baseline time ratio is reported.

During development, measure on --split validation. Use --split test only for
the baseline and for your frozen final predictor.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True, type=Path, help='Baseline checkpoint (or the only one)')
p.add_argument('--compare', type=Path, help='Candidate checkpoint, run alternately with --checkpoint')
p.add_argument('--split', choices=['validation', 'test'], default='validation')
p.add_argument('--repeats', type=int, default=5)
p.add_argument('--threads', type=int, default=4)
p.add_argument('--baseline-seconds', type=float, help='Median baseline scoring time on this machine')
args = p.parse_args()


def tree_rss(proc):
    total = 0
    for p in [proc] + proc.children(recursive=True):
        try:
            total += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def run_once(checkpoint, out):
    cmd = [sys.executable, 'evaluate.py', '--checkpoint', str(checkpoint), '--device', 'cpu',
           '--precision', 'fp32', '--threads', str(args.threads), '--split', args.split, '--output', str(out)]
    proc = psutil.Popen(cmd, stdout=subprocess.DEVNULL)
    peak = 0
    while proc.poll() is None:
        try:
            peak = max(peak, tree_rss(proc))
        except psutil.NoSuchProcess:
            break
        time.sleep(0.02)
    if proc.returncode:
        sys.exit(f'evaluate.py failed with exit code {proc.returncode}')
    result = json.loads(out.read_text())
    return {'scoring_seconds': result['seconds'], 'peak_ram_gib': peak / 2**30, 'bpb': result['bpb']}


def summarize(checkpoint, runs):
    times = [r['scoring_seconds'] for r in runs]
    return {
        'median_scoring_seconds': statistics.median(times),
        'min_scoring_seconds': min(times),
        'max_peak_ram_gib': max(r['peak_ram_gib'] for r in runs),
        'checkpoint_mib': checkpoint.stat().st_size / 2**20,
        'bpb': runs[-1]['bpb'],
    }


if args.compare is None:
    runs = []
    for i in range(args.repeats):
        runs.append(run_once(args.checkpoint, args.checkpoint.parent / f'measure_{args.split}_{i}.json'))
        print(json.dumps(runs[-1]), flush=True)

    summary = {'split': args.split, **summarize(args.checkpoint, runs)}
    del summary['min_scoring_seconds']
    if args.baseline_seconds:
        summary['time_ratio_vs_baseline'] = summary['median_scoring_seconds'] / args.baseline_seconds
        summary['within_limits'] = (summary['time_ratio_vs_baseline'] <= 5 and summary['max_peak_ram_gib'] <= 4
                                    and summary['checkpoint_mib'] <= 64)
    print(json.dumps(summary, indent=2))
    sys.exit()

baseline_runs, candidate_runs, ratios = [], [], []
for i in range(args.repeats):
    b = run_once(args.checkpoint, args.checkpoint.parent / f'measure_{args.split}_baseline_{i}.json')
    print(json.dumps({'round': i, 'role': 'baseline', **b}), flush=True)
    c = run_once(args.compare, args.compare.parent / f'measure_{args.split}_candidate_{i}.json')
    print(json.dumps({'round': i, 'role': 'candidate', **c}), flush=True)
    baseline_runs.append(b)
    candidate_runs.append(c)
    ratios.append(c['scoring_seconds'] / b['scoring_seconds'])

baseline = summarize(args.checkpoint, baseline_runs)
candidate = summarize(args.compare, candidate_runs)
time_ratio_median = statistics.median(ratios)
summary = {
    'split': args.split,
    'baseline': baseline,
    'candidate': candidate,
    'time_ratios': ratios,
    'time_ratio_median': time_ratio_median,
    'within_limits': (time_ratio_median <= 5 and candidate['max_peak_ram_gib'] <= 4
                      and candidate['checkpoint_mib'] <= 64),
}
print(json.dumps(summary, indent=2))
