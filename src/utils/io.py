import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def write_history_csv(history: List[Dict], save_path: str) -> str:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)
    return str(path)


def write_summary_json(summary: Dict, save_path: str) -> str:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return str(path)
