#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def iter_json_objects_from_line(line: str):
    decoder = json.JSONDecoder()
    idx = 0
    n = len(line)
    while idx < n:
        brace_idx = line.find("{", idx)
        if brace_idx == -1:
            break
        try:
            obj, end = decoder.raw_decode(line, brace_idx)
        except json.JSONDecodeError:
            idx = brace_idx + 1
            continue
        yield obj
        idx = end


def parse_train_loss(log_path: Path):
    xs = []
    ys = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for obj in iter_json_objects_from_line(line):
                if "train_loss" not in obj:
                    continue
                x = obj.get("global_step")
                if x is None:
                    x = obj.get("epoch")
                if x is None:
                    continue
                xs.append(x)
                ys.append(obj["train_loss"])
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train loss curve from logs.json.txt")
    parser.add_argument("--log", required=True, help="Path to logs.json.txt")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--title", default="Train Loss Curve")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xs, ys = parse_train_loss(log_path)
    if not xs:
        raise RuntimeError(f"No train_loss records found in {log_path}")

    plt.figure(figsize=(10, 4.5))
    plt.plot(xs, ys, linewidth=1.0, alpha=0.9, label="train_loss")
    plt.xlabel("global_step")
    plt.ylabel("loss")
    plt.title(args.title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"saved: {out_path}")
    print(f"points: {len(xs)}")


if __name__ == "__main__":
    main()
