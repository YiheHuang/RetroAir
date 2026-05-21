"""
utils/metrics.py —— 评估指标工具函数
"""
import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, cohen_kappa_score,
    confusion_matrix,
)


def regression_report(y_true, y_pred, prefix=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        f"{prefix}R2": r2_score(y_true, y_pred),
        f"{prefix}RMSE": rmse,
        f"{prefix}MAE": mean_absolute_error(y_true, y_pred),
        f"{prefix}MAPE": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
    }


def classification_report(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro"),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "ConfusionMatrix": confusion_matrix(y_true, y_pred),
    }


def pm25_to_aqi_level(pm25):
    """PM2.5 (μg/m³) → AQI 等级 (0-5)"""
    bp = [0, 35, 75, 115, 150, 250, float("inf")]
    for i, (lo, hi) in enumerate(zip(bp[:-1], bp[1:])):
        if lo <= pm25 < hi:
            return i
    return 5
