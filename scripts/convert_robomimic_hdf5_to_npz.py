import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.robomimic_hdf5 import convert_robomimic_hdf5_to_npz


def parse_obs_keys(raw_obs_keys: str) -> list[str]:
    obs_keys = [key.strip() for key in raw_obs_keys.split(",") if key.strip()]
    if not obs_keys:
        raise ValueError("至少需要通过 --obs-keys 提供一个 observation 键。")
    return obs_keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把 robomimic 的 low-dim demonstration HDF5 转成 baseline 使用的 .npz。"
    )
    parser.add_argument("--input-hdf5", required=True, type=str, help="输入 robomimic HDF5 路径。")
    parser.add_argument(
        "--obs-keys",
        required=True,
        type=str,
        help="按固定顺序拼接的 low-dim obs 键，使用逗号分隔，例如 robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,object",
    )
    parser.add_argument("--output-npz", required=True, type=str, help="输出 npz 路径。")
    parser.add_argument(
        "--action-key",
        default="actions",
        type=str,
        help="动作数据集键名，默认 actions。",
    )
    args = parser.parse_args()

    summary = convert_robomimic_hdf5_to_npz(
        h5_path=Path(args.input_hdf5),
        output_path=Path(args.output_npz),
        obs_keys=parse_obs_keys(args.obs_keys),
        action_key=args.action_key,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
