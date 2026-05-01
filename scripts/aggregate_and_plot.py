"""Aggregate experiment results under `results/` and plot comparisons.

Collects `summary.json` (preferred) or `metrics.csv` from result folders, writes
a combined CSV and two PNG plots: `val_success_rate` and `val_mse` per experiment.
"""
import argparse
import json
import os
import csv
from glob import glob
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_summary(path):
    j = os.path.join(path, 'summary.json')
    if os.path.exists(j):
        try:
            with open(j, 'r') as f:
                s = json.load(f)
            fm = s.get('final_metrics', {})
            return fm
        except Exception:
            pass
    # fallback to metrics.csv last line
    csvp = os.path.join(path, 'metrics.csv')
    if os.path.exists(csvp):
        with open(csvp, 'r') as f:
            rows = list(csv.reader(f))
            if len(rows) >= 2:
                header = rows[0]
                last = rows[-1]
                d = dict(zip(header, last))
                return {
                    'val_mse': float(d.get('val_mse', math.nan)),
                    'val_success_rate': float(d.get('val_success_rate', math.nan)),
                    'val_cumulative_reward': float(d.get('val_cumulative_reward', math.nan)),
                }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', default='results')
    parser.add_argument('--out-csv', default='results/exp_summary/summary_table.csv')
    parser.add_argument('--out-dir', default='results/exp_summary')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for d in sorted(glob(os.path.join(args.results_root, '*'))):
        if not os.path.isdir(d):
            continue
        fm = load_summary(d)
        if not fm:
            continue
        # try to infer experiment and seed from folder name
        name = os.path.basename(d)
        seed = None
        if '_seed' in name:
            parts = name.rsplit('_seed', 1)
            exp_name = parts[0]
            seed = parts[1]
        else:
            exp_name = name
        rows.append({
            'experiment': exp_name,
            'folder': d,
            'seed': seed,
            'val_mse': float(fm.get('val_mse', float('nan'))),
            'val_success_rate': float(fm.get('val_success_rate', fm.get('val_success', float('nan')))),
            'val_cumulative_reward': float(fm.get('val_cumulative_reward', fm.get('val_cumulative_reward', float('nan')))),
        })

    if not rows:
        print('No result summaries found under', args.results_root)
        return

    # write CSV
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment', 'seed', 'val_mse', 'val_success_rate', 'val_cumulative_reward', 'folder'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print('Wrote', args.out_csv)

    # aggregate by experiment
    groups = {}
    for r in rows:
        groups.setdefault(r['experiment'], []).append(r)

    # plot success_rate
    labels = []
    means = []
    errs = []
    for name, items in groups.items():
        vals = [i['val_success_rate'] for i in items if not math.isnan(i['val_success_rate'])]
        if vals:
            labels.append(name)
            means.append(sum(vals)/len(vals))
            errs.append((max(vals)-min(vals))/2 if len(vals)>1 else 0.0)

    if labels:
        x = range(len(labels))
        plt.figure(figsize=(8,4))
        plt.bar(x, means, yerr=errs, capsize=5)
        plt.xticks(x, labels, rotation=30)
        plt.ylabel('val_success_rate')
        plt.tight_layout()
        outp = os.path.join(args.out_dir, 'val_success_rate.png')
        plt.savefig(outp)
        print('Saved', outp)

    # plot val_mse as line plot
    labels = []
    means = []
    for name, items in groups.items():
        vals = [i['val_mse'] for i in items if not math.isnan(i['val_mse'])]
        if vals:
            labels.append(name)
            means.append(sum(vals)/len(vals))
    if labels:
        x = range(len(labels))
        plt.figure(figsize=(8,4))
        plt.plot(x, means, marker='o')
        plt.xticks(x, labels, rotation=30)
        plt.ylabel('val_mse')
        plt.tight_layout()
        outp = os.path.join(args.out_dir, 'val_mse.png')
        plt.savefig(outp)
        print('Saved', outp)


if __name__ == '__main__':
    main()
