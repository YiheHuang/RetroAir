"""
evaluate.py —— 评估 + 可视化 + 统计检验
用法: python evaluate.py --results results/ --output figures/
"""
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, cohen_kappa_score,
)


def compute_regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
    }


def compute_classification_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro"),
        "Kappa": cohen_kappa_score(y_true, y_pred),
    }


def bootstrap_test(model_a_preds, model_b_preds, y_true, n_bootstrap=1000):
    """配对Bootstrap显著性检验"""
    errors_a = np.abs(y_true - model_a_preds)
    errors_b = np.abs(y_true - model_b_preds)
    diff = errors_a - errors_b

    n = len(diff)
    bootstrapped_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        bootstrapped_diffs.append(np.mean(diff[idx]))

    p_value = np.mean(np.array(bootstrapped_diffs) <= 0)
    return {"p_value": p_value, "significant_at_005": p_value < 0.05}


# TODO: SHAP 分析
# TODO: 残差诊断图
# TODO: 混淆矩阵热力图
# TODO: 雷达图多模型对比
# TODO: 消融实验汇总图


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/")
    parser.add_argument("--output", default="figures/")
    args = parser.parse_args()

    print("评估框架就绪 (待模型输出对接)")
    # TODO: 加载各模型预测结果
    # TODO: 计算全量指标
    # TODO: 统计检验
    # TODO: 生成所有图表


if __name__ == "__main__":
    main()
