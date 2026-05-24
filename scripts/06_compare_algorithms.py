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

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "models" / "compare"


def _torch_predict(model, fused_test, y_scaler, device):
    import torch
    scaled = predict_single_input_torch(model, fused_test, device)
    return y_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(-1)


# ---- 模型训练函数 ----

def run_resmlp(data) -> tuple[dict, np.ndarray]:
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
    save_model(model, OUTPUT_DIR, "ResMLP", SEED)
    return metrics_report(data.y_test, yp), yp


def run_dcnv2(data) -> tuple[dict, np.ndarray]:
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
    save_model(model, OUTPUT_DIR, "DCN-V2", SEED)
    return metrics_report(data.y_test, yp), yp


def run_catboost(data) -> tuple[dict, np.ndarray]:
    import catboost as cb
    model = cb.CatBoostRegressor(
        iterations=1000, depth=10, learning_rate=0.03,
        random_seed=SEED, verbose=False, thread_count=-1,
    )
    model.fit(data.fused_train, data.y_train,
              eval_set=(data.fused_val, data.y_val), early_stopping_rounds=50)
    yp = model.predict(data.fused_test)
    save_model(model, OUTPUT_DIR, "CatBoost", SEED)
    return metrics_report(data.y_test, yp), yp


def run_lightgbm(data) -> tuple[dict, np.ndarray]:
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, num_leaves=63,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(data.fused_train, data.y_train,
              eval_set=[(data.fused_val, data.y_val)], eval_metric="rmse")
    yp = model.predict(data.fused_test)
    save_model(model, OUTPUT_DIR, "LightGBM", SEED)
    return metrics_report(data.y_test, yp), yp


def run_xgboost(data) -> tuple[dict, np.ndarray]:
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        random_state=SEED, n_jobs=-1,
    )
    model.fit(data.fused_train, data.y_train,
              eval_set=[(data.fused_val, data.y_val)], verbose=False)
    yp = model.predict(data.fused_test)
    save_model(model, OUTPUT_DIR, "XGBoost", SEED)
    return metrics_report(data.y_test, yp), yp


MODELS = [
    ("ResMLP",    run_resmlp),
    ("DCN-V2",    run_dcnv2),
    ("CatBoost",  run_catboost),
    ("LightGBM",  run_lightgbm),
    ("XGBoost",   run_xgboost),
]

DISPLAY_METRICS = ["r2", "rmse", "mae", "smape", "mape", "pearson_r", "explained_variance"]


def _save_result(name: str, metrics: dict, y_true, y_pred):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m_path = OUTPUT_DIR / f"{name}_seed{SEED}_metrics.json"
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump({"model": name, "seed": SEED, **metrics}, f, ensure_ascii=False, indent=2)
    p_path = OUTPUT_DIR / f"{name}_seed{SEED}_preds.csv"
    pd.DataFrame({"sample_id": np.arange(len(y_true)), "y_true": y_true, "y_pred": y_pred}).to_csv(p_path, index=False)
    return m_path, p_path


def _plot(results: list[tuple[str, dict]], output_dir: Path):
    """生成模型对比柱状图。"""
    names = [r[0] for r in results]
    r2_vals = [r[1]["r2"] for r in results]
    rmse_vals = [r[1]["rmse"] for r in results]
    mae_vals = [r[1]["mae"] for r in results]
    smape_vals = [r[1]["smape"] for r in results]
    pearson_vals = [r[1]["pearson_r"] for r in results]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("UrbanAir — 五算法对比 (seed=42)", fontsize=16, fontweight="bold", y=0.98,
                 fontproperties=_CJK_FONT)

    bar_configs = [
        (axes[0, 0], r2_vals, "R²", "higher is better", colors, (0.65, 1.0)),
        (axes[0, 1], rmse_vals, "RMSE (μg/m³)", "lower is better", colors, None),
        (axes[0, 2], mae_vals, "MAE (μg/m³)", "lower is better", colors, None),
        (axes[1, 0], smape_vals, "SMAPE (%)", "lower is better", colors, None),
        (axes[1, 1], pearson_vals, "Pearson r", "higher is better", colors, (0.7, 1.0)),
    ]

    for ax, vals, title, subtitle, clrs, ylim in bar_configs:
        bars = ax.bar(names, vals, color=clrs, edgecolor="white", linewidth=0.8)
        ax.set_title(f"{title}\n({subtitle})", fontsize=11)
        ax.set_ylabel(title)
        if ylim:
            ax.set_ylim(*ylim)
        for bar, val in zip(bars, vals):
            offset = 0.01 if "higher" in subtitle else 0.02
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f"{val:.4f}" if abs(val) < 100 else f"{val:.1f}",
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
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    seed = args.seed
    set_seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dp = Path(args.data_path) if args.data_path else None
    data = load_prepared_data(data_path=dp, split_seed=seed)

    print(f"\n{'='*75}")
    print(f"  五算法对比 (seed={seed})")
    print(f"  数据: {len(data.y_train)+len(data.y_val)+len(data.y_test)} 行, "
          f"{data.fused_train.shape[1]} 特征")
    print(f"{'='*75}\n")

    results = []
    for name, fn in MODELS:
        print(f"[{name}]", end=" ", flush=True)
        t0 = time.time()
        try:
            metrics, y_pred = fn(data)
            elapsed = time.time() - t0
            results.append((name, metrics, elapsed))
            print(f"R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.2f}  "
                  f"MAE={metrics['mae']:.2f}  SMAPE={metrics['smape']:.1f}%  "
                  f"Pearson r={metrics['pearson_r']:.4f}  ({elapsed:.0f}s)")
            _save_result(name, metrics, data.y_test, y_pred)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAIL ({elapsed:.0f}s): {e}")

    if not results:
        print("\n无有效结果。")
        return

    # 按 R² 排序
    results.sort(key=lambda x: x[1]["r2"], reverse=True)

    # 对比表
    print(f"\n{'='*95}")
    header = f"{'模型':<14s}  {'R²':>8s}  {'RMSE':>8s}  {'MAE':>7s}  {'SMAPE%':>7s}  {'MAPE%':>7s}  {'Pearson r':>9s}  {'ExpVar':>7s}  {'用时':>6s}"
    print(header)
    print(f"{'-'*95}")
    for name, m, elapsed in results:
        mape_str = f"{m['mape']:7.1f}" if not np.isnan(m['mape']) else "      —"
        print(f"{name:<14s}  {m['r2']:8.4f}  {m['rmse']:8.2f}  {m['mae']:7.2f}  "
              f"{m['smape']:7.1f}  {mape_str}  {m['pearson_r']:9.4f}  "
              f"{m['explained_variance']:7.4f}  {elapsed:5.0f}s")

    # 保存汇总
    summary = []
    for name, m, elapsed in results:
        row = {"model": name, "time_s": int(elapsed)}
        for k in DISPLAY_METRICS:
            v = m.get(k, float("nan"))
            row[k] = round(v, 4) if isinstance(v, float) and not np.isnan(v) else v
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False)

    # 可视化
    plot_data = [(name, m) for name, m, _ in results]
    _plot(plot_data, OUTPUT_DIR)

    print(f"\n全部结果: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
