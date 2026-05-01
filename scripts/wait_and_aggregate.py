"""Wait for all experiment runs to finish then run the aggregator.

Usage: python scripts/wait_and_aggregate.py
It expects experiments: pickcube_baseline_run, pickcube_attnres_run, pickcube_uniform_run
with seeds 42,123,777 (matching how exp_runner named output dirs).
"""
import time
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = os.path.join(ROOT, 'results')
EXPS = ['pickcube_baseline_run', 'pickcube_attnres_run', 'pickcube_uniform_run']
SEEDS = [42, 123, 777]

def all_done():
    missing = []
    for e in EXPS:
        for s in SEEDS:
            d = os.path.join(RESULTS, f"{e}_seed{s}")
            summary = os.path.join(d, 'summary.json')
            if not os.path.exists(summary):
                missing.append(summary)
    return missing

def main():
    print('Waiting for runs to finish...')
    while True:
        missing = all_done()
        if not missing:
            print('All runs finished.')
            break
        print(f'{len(missing)} summaries missing; next check in 30s')
        time.sleep(30)

    # run aggregator
    agg = [os.path.join(ROOT, 'scripts', 'aggregate_and_plot.py')]
    print('Running aggregator:', ' '.join([os.sys.executable] + agg))
    subprocess.run([os.sys.executable] + agg)
    print('Aggregator finished. See results/exp_summary')

if __name__ == '__main__':
    main()
