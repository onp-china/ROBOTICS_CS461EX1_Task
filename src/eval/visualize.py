import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def save_training_curves(history: List[Dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(history)

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["train_mse"], label="train_mse")
    plt.plot(df["epoch"], df["val_mse"], label="val_mse")
    plt.plot(df["epoch"], df["val_success_rate"], label="val_success_rate")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Full Baseline Training Curve")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "training_curve.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path
