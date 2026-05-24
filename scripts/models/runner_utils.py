import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def save_model(model, output_dir: Path, model_name: str, seed: int) -> Path:
    """保存模型权重。PyTorch → .pt, sklearn/boosting → .pkl"""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            path = output_dir / f"{model_name}_seed{seed}.pt"
            torch.save(model.state_dict(), path)
            return path
    except ImportError:
        pass

    path = output_dir / f"{model_name}_seed{seed}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def write_model_outputs(output_dir: Path, model_name: str, seed: int,
                        metrics: dict, y_true, y_pred):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 指标 JSON
    metrics_path = output_dir / f"{model_name}_seed{seed}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "seed": seed, **metrics},
                  f, ensure_ascii=False, indent=2)

    # 预测 CSV
    preds_path = output_dir / f"{model_name}_seed{seed}_preds.csv"
    pred_df = pd.DataFrame({
        "sample_id": np.arange(len(y_true)),
        "y_true": y_true,
        "y_pred": y_pred,
    })
    pred_df.to_csv(preds_path, index=False)
    return metrics_path, preds_path
