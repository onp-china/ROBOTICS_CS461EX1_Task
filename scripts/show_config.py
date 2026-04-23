import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="展示配置。")
    parser.add_argument("--config", required=True, type=str, help="配置文件路径。")
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
