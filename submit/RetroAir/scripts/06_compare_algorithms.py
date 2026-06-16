"""
06_compare_algorithms.py —— 五种算法统一对比 + 可视化

用法:
  python scripts/06_compare_algorithms.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# 中文字体
_CJK_FONT = None
for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
    if Path(fp).exists():
        _CJK_FONT = fm.FontProperties(fname=fp)
        break

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common.data_utils import load_prepared_data, metrics_report, set_seed
from scripts.common.torch_utils import predict_single_input_torch, train_single_input_torch
from scripts.models.runner_utils import save_model

DEFAULT_SEEDS = list(range(42, 52))
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "models" / "compare"
DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "final" / "merged_8_cities_2018_2025.csv"


def _torch_predict(model, fused_test, y_scaler, device):
    import torch
    scaled = predict_single_input_torch(model, fused_test, device)
    return y_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(-1)


# ---- 模型训练函数 ----

def run_resmlp(data, seed: int, save_artifacts: bool = False) -> tuple[dict, np.ndarray]:
    from scripts.models.nn_defs import ResMLPRegressor
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResMLPRegressor(data.fused_train.shape[1], hidden_dim=128, n_blocks=4, dropout=0.15)
    model = train_single_input_torch(
        model, data.fused_train, data.y_train_scaled,
        data.fused_val, data.y_val_scaled,
        epochs=120, batch_size=256, lr=1e-3, wd=1e-4,
        patience=20, warmup_epochs=5, device=device,
    )
    yp = _torch_predict(model, data.fused_test, data.y_scaler, device)
    if save_artifacts:
        save_model(model, OUTPUT_DIR, "ResMLP", seed)
    return metrics_report(data.y_test, yp), yp


def run_dcnv2(data, seed: int, save_artifacts: bool = False) -> tuple[dict, np.ndarray]:
    from scripts.models.nn_defs import DCNV2Regressor
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNV2Regressor(data.fused_train.shape[1], deep_dim=128, n_cross_layers=3, n_deep_layers=3, dropout=0.15)
    model = train_single_input_torch(
        model, data.fused_train, data.y_train_scaled,
        data.fused_val, data.y_val_scaled,
        epochs=100, batch_size=512, lr=1e-3, wd=1e-4,
        patience=15, warmup_epochs=5, device=device,
    )
    yp = _torch_predict(model, data.fused_test, data.y_scaler, device)
    if save_artifacts:
        save_model(model, OUTPUT_DIR, "DCN-V2", seed)
    return metrics_report(data.y_test, yp), yp


def run_catboost(data, seed: int, save_artifacts: bool = False) -> tuple[dict, np.ndarray]:
    import catboost as cb
    model = cb.CatBoostRegressor(
        iterations=1000, depth=10, learning_rate=0.03,
        random_seed=seed, verbose=False, thread_count=-1,
    )
    model.fit(data.fused_train, data.y_train,
              eval_set=(data.fused_val, data.y_val), early_stopping_rounds=50)
    yp = model.predict(data.fused_test)
    if save_artifacts:
        save_model(model, OUTPUT_DIR, "CatBoost", seed)
    return metrics_report(data.y_test, yp), yp


def run_lightgbm(data, seed: int, save_artifacts: bool = False) -> tuple[dict, np.ndarray]:
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, num_leaves=63,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(data.fused_train, data.y_train,
              eval_set=[(data.fused_val, data.y_val)], eval_metric="rmse")
    yp = model.predict(data.fused_test)
    if save_artifacts:
        save_model(model, OUTPUT_DIR, "LightGBM", seed)
    return metrics_report(data.y_test, yp), yp


def run_xgboost(data, seed: int, save_artifacts: bool = False) -> tuple[dict, np.ndarray]:
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        random_state=seed, n_jobs=-1,
    )
    model.fit(data.fused_train, data.y_train,
              eval_set=[(data.fused_val, data.y_val)], verbose=False)
    yp = model.predict(data.fused_test)
    if save_artifacts:
        save_model(model, OUTPUT_DIR, "XGBoost", seed)
    return metrics_report(data.y_test, yp), yp


MODELS = [
    ("ResMLP",    run_resmlp),
    ("DCN-V2",    run_dcnv2),
    ("CatBoost",  run_catboost),
    ("LightGBM",  run_lightgbm),
    ("XGBoost",   run_xgboost),
]

DISPLAY_METRICS = ["r2", "rmse", "mae", "smape", "mape", "pearson_r", "explained_variance"]


def _save_result(name: str, seed: int, metrics: dict, y_true, y_pred, save_preds: bool):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m_path = OUTPUT_DIR / f"{name}_seed{seed}_metrics.json"
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump({"model": name, "seed": seed, **metrics}, f, ensure_ascii=False, indent=2)
    p_path = None
    if save_preds:
        p_path = OUTPUT_DIR / f"{name}_seed{seed}_preds.csv"
        pd.DataFrame({"sample_id": np.arange(len(y_true)), "y_true": y_true, "y_pred": y_pred}).to_csv(p_path, index=False)
    return m_path, p_path


def _plot(summary_df: pd.DataFrame, output_dir: Path):
    """按 10 个随机种子的均值生成模型对比柱状图。"""
    names = summary_df["model"].tolist()
    r2_vals = summary_df["r2_mean"].tolist()
    rmse_vals = summary_df["rmse_mean"].tolist()
    mae_vals = summary_df["mae_mean"].tolist()
    smape_vals = summary_df["smape_mean"].tolist()
    pearson_vals = summary_df["pearson_r_mean"].tolist()
    r2_err = summary_df["r2_std"].tolist()
    rmse_err = summary_df["rmse_std"].tolist()
    mae_err = summary_df["mae_std"].tolist()
    smape_err = summary_df["smape_std"].tolist()
    pearson_err = summary_df["pearson_r_std"].tolist()
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("RetroAir — 五算法对比 (10 seeds, mean ± std)", fontsize=16, fontweight="bold", y=0.98,
                 fontproperties=_CJK_FONT)

    bar_configs = [
        (axes[0, 0], r2_vals, r2_err, "R²", "higher is better", colors, (0.65, 1.0)),
        (axes[0, 1], rmse_vals, rmse_err, "RMSE (μg/m³)", "lower is better", colors, None),
        (axes[0, 2], mae_vals, mae_err, "MAE (μg/m³)", "lower is better", colors, None),
        (axes[1, 0], smape_vals, smape_err, "SMAPE (%)", "lower is better", colors, None),
        (axes[1, 1], pearson_vals, pearson_err, "Pearson r", "higher is better", colors, (0.7, 1.0)),
    ]

    for ax, vals, errs, title, subtitle, clrs, ylim in bar_configs:
        bars = ax.bar(names, vals, yerr=errs, capsize=4, color=clrs, edgecolor="white", linewidth=0.8)
        ax.set_title(f"{title}\n({subtitle})", fontsize=11)
        ax.set_ylabel(title)
        if ylim:
            ax.set_ylim(*ylim)
        for bar, val, err in zip(bars, vals, errs):
            offset = 0.01 if "higher" in subtitle else 0.02
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    (f"{val:.4f}\n±{err:.4f}" if abs(val) < 100 else f"{val:.1f}\n±{err:.1f}"),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)

    # 隐藏空子图
    axes[1, 2].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "comparison_chart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n对比图: {path}")
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="五种算法统一对比")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="逗号分隔的随机种子列表，默认 42-51")
    parser.add_argument("--save-preds", action="store_true", help="保存每个模型每个 seed 的预测明细")
    parser.add_argument("--save-models", action="store_true", help="保存每个模型每个 seed 的模型文件")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.data_path:
        dp = Path(args.data_path)
    elif DEFAULT_DATA_PATH.exists():
        dp = DEFAULT_DATA_PATH
    else:
        dp = None

    all_rows = []
    for seed in seeds:
        set_seed(seed)
        data = load_prepared_data(data_path=dp, split_seed=seed)

        print(f"\n{'='*75}")
        print(f"  五算法对比 (seed={seed})")
        print(f"  数据: {len(data.y_train)+len(data.y_val)+len(data.y_test)} 行, "
              f"{data.fused_train.shape[1]} 特征")
        print(f"{'='*75}\n")

        for name, fn in MODELS:
            metric_path = OUTPUT_DIR / f"{name}_seed{seed}_metrics.json"
            if metric_path.exists():
                with open(metric_path, encoding="utf-8") as f:
                    saved = json.load(f)
                metrics = {k: saved[k] for k in DISPLAY_METRICS if k in saved}
                all_rows.append({"model": name, "seed": seed, "time_s": saved.get("time_s", np.nan), **metrics})
                print(f"[{name}] skip existing metrics (seed={seed})")
                continue

            print(f"[{name}]", end=" ", flush=True)
            t0 = time.time()
            try:
                save_artifacts = args.save_models or seed == 42
                metrics, y_pred = fn(data, seed=seed, save_artifacts=save_artifacts)
                elapsed = time.time() - t0
                row = {"model": name, "seed": seed, "time_s": elapsed, **metrics}
                all_rows.append(row)
                print(f"R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.2f}  "
                      f"MAE={metrics['mae']:.2f}  SMAPE={metrics['smape']:.1f}%  "
                      f"Pearson r={metrics['pearson_r']:.4f}  ({elapsed:.0f}s)")
                _save_result(name, seed, metrics, data.y_test, y_pred,
                             save_preds=args.save_preds or seed == 42)
            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAIL ({elapsed:.0f}s): {e}")

    if not all_rows:
        print("\n无有效结果。")
        return

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(OUTPUT_DIR / "comparison_all_seeds.csv", index=False)

    agg_spec = {}
    for metric in DISPLAY_METRICS:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")
    agg_spec["time_s_mean"] = ("time_s", "mean")
    summary_df = (
        all_df.groupby("model")
        .agg(**agg_spec)
        .reset_index()
        .sort_values("r2_mean", ascending=False)
    )

    # 对比表
    print(f"\n{'='*95}")
    header = f"{'模型':<14s}  {'R²均值':>8s}  {'R² std':>8s}  {'RMSE均值':>9s}  {'MAE均值':>8s}  {'SMAPE均值':>9s}  {'Pearson均值':>10s}  {'平均用时':>8s}"
    print(header)
    print(f"{'-'*95}")
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<14s}  {row['r2_mean']:8.4f}  {row['r2_std']:8.4f}  "
              f"{row['rmse_mean']:9.2f}  {row['mae_mean']:8.2f}  "
              f"{row['smape_mean']:9.1f}  {row['pearson_r_mean']:10.4f}  "
              f"{row['time_s_mean']:7.0f}s")

    # 保存均值汇总
    rounded = summary_df.copy()
    for col in rounded.columns:
        if col != "model":
            rounded[col] = rounded[col].round(4)
    rounded.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False)

    # 可视化
    _plot(summary_df, OUTPUT_DIR)

    print(f"\n全部结果: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
