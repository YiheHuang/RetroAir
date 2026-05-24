import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_FINAL, PM25_COL, SATELLITE_FEATURES, SPLIT_SEED,
    TEST_SIZE, TIME_FEATURES, WEATHER_VARIABLES,
)


@dataclass
class PreparedData:
    sat_cols: list[str]
    met_cols: list[str]
    sat_scaler: StandardScaler
    met_scaler: StandardScaler
    fused_scaler: StandardScaler
    y_scaler: StandardScaler
    sat_train: np.ndarray
    sat_val: np.ndarray
    sat_test: np.ndarray
    met_train: np.ndarray
    met_val: np.ndarray
    met_test: np.ndarray
    fused_train: np.ndarray
    fused_val: np.ndarray
    fused_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_train_scaled: np.ndarray
    y_val_scaled: np.ndarray
    test_meta: pd.DataFrame


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算回归评估指标。SMAPE 替代 MAPE 作为主百分比指标（对零值鲁棒）。"""
    residual = y_true - y_pred
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))

    # SMAPE: 对称 MAPE，值域 [0, 200]，对 y≈0 鲁棒
    smape = float(np.mean(2.0 * np.abs(residual) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)

    # MAPE: 仅对 y_true ≥ 1 的样本计算（过滤零值噪声）
    mask = np.abs(y_true) >= 1.0
    if mask.sum() > 10:
        mape = float(np.mean(np.abs(residual[mask] / y_true[mask])) * 100)
    else:
        mape = float("nan")

    # 解释方差
    var_y = np.var(y_true)
    explained_variance = float(1.0 - np.var(residual) / var_y) if var_y > 1e-8 else float("nan")

    # Pearson 相关系数
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "smape": smape,
        "mape": mape,
        "explained_variance": explained_variance,
        "pearson_r": pearson,
    }


def _pick_existing_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def load_prepared_data(
    data_path: Path | None = None,
    split_seed: int = SPLIT_SEED,
    test_size: float = TEST_SIZE,
) -> PreparedData:
    """加载数据并按随机切分准备训练/验证/测试集（空间插值任务）。"""
    if data_path is None:
        data_path = DATA_FINAL / "merged_dataset.csv"
    if not Path(data_path).exists():
        raise FileNotFoundError(f"{data_path} 不存在")

    df = pd.read_csv(data_path)
    if PM25_COL not in df.columns:
        raise ValueError(f"未找到标签列: {PM25_COL}")

    sat_cols = _pick_existing_columns(df, SATELLITE_FEATURES)
    met_cols = _pick_existing_columns(df, WEATHER_VARIABLES)
    time_cols = _pick_existing_columns(df, TIME_FEATURES)
    feature_cols = sat_cols + met_cols + time_cols

    if len(sat_cols) == 0 or len(met_cols) == 0:
        raise ValueError(f"特征缺失 sat={sat_cols}, met={met_cols}")

    keep_cols = ["city", "station_name", "date", PM25_COL] + feature_cols
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    df = df.dropna(subset=[PM25_COL]).reset_index(drop=True)

    # 随机切分（空间插值任务）
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=split_seed,
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=split_seed,
    )

    sat_scaler = StandardScaler()
    met_scaler = StandardScaler()
    fused_scaler = StandardScaler()
    y_scaler = StandardScaler()

    sat_train = sat_scaler.fit_transform(train_df[sat_cols].values)
    sat_val = sat_scaler.transform(val_df[sat_cols].values)
    sat_test = sat_scaler.transform(test_df[sat_cols].values)

    met_train = met_scaler.fit_transform(train_df[met_cols].values)
    met_val = met_scaler.transform(val_df[met_cols].values)
    met_test = met_scaler.transform(test_df[met_cols].values)

    fused_cols = feature_cols
    fused_train = fused_scaler.fit_transform(train_df[fused_cols].values)
    fused_val = fused_scaler.transform(val_df[fused_cols].values)
    fused_test = fused_scaler.transform(test_df[fused_cols].values)

    y_train = train_df[PM25_COL].values.astype(np.float32)
    y_val = val_df[PM25_COL].values.astype(np.float32)
    y_test = test_df[PM25_COL].values.astype(np.float32)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)

    test_meta = test_df[["city", "station_name", "date"]].reset_index(drop=True).copy()
    test_meta.insert(0, "sample_id", np.arange(len(test_meta)))

    print(f"数据: {len(df)} 行, 特征={len(fused_cols)} ({len(sat_cols)} AOD + {len(met_cols)} 气象 + {len(time_cols)} 时间)")
    print(f"切分: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

    return PreparedData(
        sat_cols=sat_cols,
        met_cols=met_cols,
        sat_scaler=sat_scaler,
        met_scaler=met_scaler,
        fused_scaler=fused_scaler,
        y_scaler=y_scaler,
        sat_train=sat_train,
        sat_val=sat_val,
        sat_test=sat_test,
        met_train=met_train,
        met_val=met_val,
        met_test=met_test,
        fused_train=fused_train,
        fused_val=fused_val,
        fused_test=fused_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_train_scaled=y_train_scaled,
        y_val_scaled=y_val_scaled,
        test_meta=test_meta,
    )


def save_groundtruth(prepared: PreparedData, groundtruth_path: Path | None = None):
    if groundtruth_path is None:
        groundtruth_path = DATA_MODELS / "groundtruth_test.csv"
    gt = prepared.test_meta.copy()
    gt["y_true"] = prepared.y_test
    gt.to_csv(groundtruth_path, index=False)
    return groundtruth_path
