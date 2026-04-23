import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.demo_npz import inspect_demo_directory
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 ManiSkill demonstration npz 数据。")
    parser.add_argument("--config", required=True, type=str, help="配置文件路径。")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        summary = inspect_demo_directory(config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except FileNotFoundError as exc:
        print(f"[ERROR] 未找到 demonstration 数据: {exc}")
        print("[HINT] 请检查 data.demo_root 是否是存在的目录，并确认目录下有 .npz 文件。")
        raise
    except KeyError as exc:
        print(f"[ERROR] demonstration 字段缺失: {exc}")
        print("[HINT] 请检查配置中的 obs_key/action_key 是否与 .npz 中真实键名一致。")
        raise
    except ValueError as exc:
        print(f"[ERROR] demonstration 形状或切窗参数不合法: {exc}")
        print("[HINT] 请检查 observations/actions 形状、sequence_length 和 stride 是否合理。")
        raise


if __name__ == "__main__":
    main()
