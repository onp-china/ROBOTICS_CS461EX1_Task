"""Run multiple experiment configs with multiple seeds (parallel-capable).

Creates temporary config copies with updated seed and output_dir per seed,
launches `python scripts/train.py --config <tmp>` processes and stores logs.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_single(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "wb") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True, help="config yaml paths")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tmp_dir = os.path.join("tmp_configs")
    os.makedirs(tmp_dir, exist_ok=True)

    tasks = []
    for cfg_path in args.configs:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        base_out = cfg["experiment"].get("output_dir", "results/exp")
        for seed in args.seeds:
            tmp_cfg = dict(cfg)
            tmp_cfg["experiment"]["seed"] = int(seed)
            out_dir = f"{base_out}_seed{seed}"
            tmp_cfg["experiment"]["output_dir"] = out_dir
            name = os.path.splitext(os.path.basename(cfg_path))[0]
            tmp_path = os.path.join(tmp_dir, f"{name}_seed{seed}.yaml")
            with open(tmp_path, "w") as wf:
                yaml.safe_dump(tmp_cfg, wf)
            log_path = os.path.join(out_dir, "run.log")
            cmd = [sys.executable, "scripts/train.py", "--config", tmp_path]
            tasks.append((cmd, log_path, tmp_path, out_dir))

    if args.dry_run:
        print("Dry run tasks:")
        for cmd, log, cfg, out in tasks:
            print("CMD:", " ".join(cmd))
            print("LOG:", log)
        return

    procs = []
    running = []
    idx = 0
    total = len(tasks)
    while idx < total or running:
        while idx < total and len(running) < args.max_parallel:
            cmd, log, cfg, out = tasks[idx]
            os.makedirs(out, exist_ok=True)
            print(f"Starting: {' '.join(cmd)} -> {log}")
            p = run_single(cmd, log)
            running.append((p, cmd, log))
            idx += 1
        # poll
        time.sleep(5)
        new_running = []
        for p, cmd, log in running:
            ret = p.poll()
            if ret is None:
                new_running.append((p, cmd, log))
            else:
                print(f"Finished: {' '.join(cmd)} (ret={ret})")
        running = new_running


if __name__ == "__main__":
    main()
