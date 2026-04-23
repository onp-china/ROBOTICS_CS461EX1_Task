import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trainers.baseline_trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 full baseline 训练。")
    parser.add_argument("--config", required=True, type=str, help="配置文件路径。")
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
