import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.data_utils import load_prepared_data, metrics_report, set_seed
from scripts.common.torch_utils import predict_single_input_torch, train_single_input_torch
from scripts.models.nn_defs import DCNV2Regressor
from scripts.models.runner_utils import save_model, write_model_outputs

MODEL_NAME = "DCN-V2"


def run(
    seed: int,
    split_seed: int,
    output_dir: Path,
    data_path: Path | None = None,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    warmup_epochs: int = 5,
):
    set_seed(seed)
    data = load_prepared_data(data_path=data_path, split_seed=split_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = DCNV2Regressor(
        input_dim=data.fused_train.shape[1],
        deep_dim=128,
        n_cross_layers=3,
        n_deep_layers=3,
        dropout=0.15,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    model = train_single_input_torch(
        model,
        data.fused_train, data.y_train_scaled,
        data.fused_val, data.y_val_scaled,
        epochs=epochs, batch_size=batch_size,
        lr=lr, wd=weight_decay, patience=patience,
        device=device, warmup_epochs=warmup_epochs,
    )

    y_pred_scaled = predict_single_input_torch(model, data.fused_test, device)
    y_pred = data.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)

    metrics = metrics_report(data.y_test, y_pred)
    write_model_outputs(output_dir, MODEL_NAME, seed, metrics, data.y_test, y_pred)
    save_model(model, output_dir, MODEL_NAME, seed)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train DCN-V2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    args = parser.parse_args()

    dp = Path(args.data_path) if args.data_path else None
    metrics = run(args.seed, args.split_seed, args.output_dir, data_path=dp,
                  epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, weight_decay=args.weight_decay,
                  patience=args.patience, warmup_epochs=args.warmup_epochs)
    print(f"{MODEL_NAME}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
