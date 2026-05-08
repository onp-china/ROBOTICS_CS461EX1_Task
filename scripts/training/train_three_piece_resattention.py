#!/usr/bin/env python3
"""
兼容入口：三块装配 + ResAttention。等价于:

  python scripts/train_resattention_mimicgen.py --task three_piece_assembly_d0_lowdim_abs [选项...]

默认输出目录仍为 outputs_resattention/run2（历史习惯）；请改用 train_resattention_mimicgen 指定其它任务。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    repo_root = Path(__file__).resolve().parents[1]
    runner = repo_root / "scripts" / "train_resattention_mimicgen.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)

    extra = sys.argv[1:]
    has_out = any(a == "--out-dir" or a.startswith("--out-dir=") for a in extra)
    cmd = [sys.executable, str(runner), "--task", "three_piece_assembly_d0_lowdim_abs"]
    if not has_out:
        cmd += ["--out-dir", str(repo_root / "outputs_resattention" / "run2")]
    cmd += extra
    print("Forwarding to train_resattention_mimicgen.py:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo_root))


if __name__ == "__main__":
    main()
