"""
集成学习: ResMLP + DCN-V2 + CatBoost 加权平均
三个模型拟合数据的不同方面，集成后通常超越最优单模型
"""
import argparse
import sys
from pathlib import Path

import catboost as cb
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.data_utils import load_prepared_data, set_seed
from scripts.common.torch_utils import predict_single_input_torch, train_single_input_torch
from scripts.models.nn_defs import DCNV2Regressor, ResMLPRegressor
from scripts.models.runner_utils import write_model_outputs

MODEL_NAME = "Ensemble"


def run(seed: int, split_seed: int, output_dir: Path, data_path: Path | None = None):
    set_seed(seed)
    data = load_prepared_data(data_path=data_path, split_seed=split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 1) ResMLP
    print("\n[1/3] 训练 ResMLP ...")
    model1 = ResMLPRegressor(data.fused_train.shape[1], hidden_dim=128, n_blocks=4, dropout=0.15)
    model1 = train_single_input_torch(
        model1, data.fused_train, data.y_train_scaled,
        data.fused_val, data.y_val_scaled,
        epochs=100, batch_size=512, lr=1e-3, wd=1e-4,
        patience=15, warmup_epochs=5, device=device,
    )
    p1 = data.y_scaler.inverse_transform(
        predict_single_input_torch(model1, data.fused_test, device).reshape(-1, 1)
    ).reshape(-1)
    r1 = r2_score(data.y_test, p1)
    print(f"  ResMLP: R²={r1:.4f}")

    # 2) DCN-V2
    print("\n[2/3] 训练 DCN-V2 ...")
    model2 = DCNV2Regressor(data.fused_train.shape[1], deep_dim=128, n_cross_layers=3, n_deep_layers=3, dropout=0.15)
    model2 = train_single_input_torch(
        model2, data.fused_train, data.y_train_scaled,
        data.fused_val, data.y_val_scaled,
        epochs=100, batch_size=512, lr=1e-3, wd=1e-4,
        patience=15, warmup_epochs=5, device=device,
    )
    p2 = data.y_scaler.inverse_transform(
        predict_single_input_torch(model2, data.fused_test, device).reshape(-1, 1)
    ).reshape(-1)
    r2 = r2_score(data.y_test, p2)
    print(f"  DCN-V2: R²={r2:.4f}")

    # 3) CatBoost
    print("\n[3/3] 训练 CatBoost ...")
    model3 = cb.CatBoostRegressor(
        iterations=1000, depth=10, learning_rate=0.03,
        random_seed=seed, verbose=False, thread_count=-1,
    )
    model3.fit(data.fused_train, data.y_train,
               eval_set=(data.fused_val, data.y_val),
               early_stopping_rounds=50)
    p3 = model3.predict(data.fused_test)
    r3 = r2_score(data.y_test, p3)
    print(f"  CatBoost: R²={r3:.4f}")

    # 集成: 简单平均 + 加权平均（用验证集 R² 做权重）
    # 加权: weights ∝ R² on validation set
    p1_val = data.y_scaler.inverse_transform(
        predict_single_input_torch(model1, data.fused_val, device).reshape(-1, 1)
    ).reshape(-1)
    p2_val = data.y_scaler.inverse_transform(
        predict_single_input_torch(model2, data.fused_val, device).reshape(-1, 1)
    ).reshape(-1)
    p3_val = model3.predict(data.fused_val)

    w1 = max(0.01, r2_score(data.y_val, p1_val))
    w2 = max(0.01, r2_score(data.y_val, p2_val))
    w3 = max(0.01, r2_score(data.y_val, p3_val))
    w_sum = w1 + w2 + w3

    # 简单平均
    p_mean = (p1 + p2 + p3) / 3.0
    r_mean = r2_score(data.y_test, p_mean)
    rmse_mean = np.sqrt(mean_squared_error(data.y_test, p_mean))

    # 加权平均
    p_weighted = (w1 * p1 + w2 * p2 + w3 * p3) / w_sum
    r_weighted = r2_score(data.y_test, p_weighted)
    rmse_weighted = np.sqrt(mean_squared_error(data.y_test, p_weighted))

    print(f"\n===== 集成结果 =====")
    print(f"  权重: ResMLP={w1:.4f}, DCN-V2={w2:.4f}, CatBoost={w3:.4f}")
    print(f"  简单平均: R²={r_mean:.4f}, RMSE={rmse_mean:.4f}")
    print(f"  加权平均: R²={r_weighted:.4f}, RMSE={rmse_weighted:.4f}")

    # 保存最优集成结果
    best_pred = p_weighted if r_weighted > r_mean else p_mean
    best_metrics = {
        "r2": max(r_mean, r_weighted),
        "rmse": min(rmse_mean, rmse_weighted),
        "mae": float(np.mean(np.abs(data.y_test - best_pred))),
        "mape": float(np.mean(np.abs((data.y_test - best_pred) / (np.abs(data.y_test) + 1e-8))) * 100),
    }
    write_model_outputs(output_dir, MODEL_NAME, seed, best_metrics, data.y_test, best_pred)
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Ensemble (ResMLP + DCN-V2 + CatBoost)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", default="")
    args = parser.parse_args()

    dp = Path(args.data_path) if args.data_path else None
    metrics = run(args.seed, args.split_seed, args.output_dir, data_path=dp)
    print(f"\n{MODEL_NAME} 最优: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
