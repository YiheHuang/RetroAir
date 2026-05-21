"""
05_train_model.py —— 训练 XGBoost 预测 PM2.5 + 评估 + 保存模型
输出: data/models/xgb_model.pkl
用法: python scripts/05_train_model.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from config import DATA_PROCESSED, DATA_MODELS, EXCLUDE_COLS, TRAIN_CUTOFF, PM25_COL


def main():
    # 加载
    data_path = DATA_PROCESSED / "merged_dataset.csv"
    if not data_path.exists():
        print(f"错误: {data_path} 不存在! 请先运行 04_merge_data.py")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"加载数据: {len(df)} 行 × {len(df.columns)} 列")

    # 确保 PM2.5 列存在
    if PM25_COL not in df.columns:
        print(f"错误: 未找到列 '{PM25_COL}'! 可用: {list(df.columns)}")
        sys.exit(1)

    # 确定特征列 (数值型, 且不在排除列表中, 不是PM2.5本身)
    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and c != PM25_COL
        and df[c].dtype in ["float64", "int64"]
    ]
    print(f"特征数: {len(feature_cols)}")

    # 数据划分 —— 按时间切分 (杜绝时间泄露)
    df["_date"] = pd.to_datetime(df["date"])
    train_mask = df["_date"] < TRAIN_CUTOFF
    test_mask = df["_date"] >= TRAIN_CUTOFF

    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, PM25_COL].copy()
    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, PM25_COL].copy()

    # 如果时间切分导致测试集为空或训练集过小, 回退到比例切分
    if len(X_test) == 0:
        print("时间切分测试集为空, 改用 80/20 随机切分")
        from sklearn.model_selection import train_test_split
        X_all = pd.concat([X_train, X_test])
        y_all = pd.concat([y_train, y_test])
        if len(X_all) < 10:
            print(f"警告: 总样本仅 {len(X_all)} 条, 模型评估不可靠(仅供测试管线)")
            X_train, y_train = X_all, y_all
            X_test, y_test = X_all, y_all  # 使用相同数据
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42
            )
    print(f"\n训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")

    # 归一化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 训练 XGBoost
    print("\n训练 XGBoost...")
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    # 评估
    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"\n{'='*55}")
    print(f"  XGBoost PM2.5 预测 —— 评估结果")
    print(f"{'='*55}")
    print(f"  训练集 R²:  {train_r2:.4f}")
    print(f"  测试集 R²:  {test_r2:.4f}")
    print(f"  测试集 RMSE: {test_rmse:.2f} μg/m³")
    print(f"  测试集 MAE:  {test_mae:.2f} μg/m³")
    print(f"  PM2.5 均值:  {y_test.mean():.1f} μg/m³")
    print(f"  RMSE/Mean:   {test_rmse/y_test.mean():.2%}")

    # MVP 成功判定
    if test_r2 >= 0.55:
        print(f"\n  ✅ MVP 达标! 测试集 R²={test_r2:.3f} ≥ 0.55")
    else:
        print(f"\n  ⚠️ MVP 未达标: 测试集 R²={test_r2:.3f} < 0.55")
        print(f"     建议: 检查数据对齐质量 / 增加OSM特征 / 考虑加入卫星图模态")

    # 特征重要性 Top-10
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n特征重要性 Top-10:")
    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:<35s} {bar} {row['importance']:.4f}")

    # 保存模型
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "pm25_col": PM25_COL,
        "metrics": {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        },
        "importance": importance.to_dict("records"),
    }

    model_path = DATA_MODELS / "xgb_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\n模型已保存: {model_path}")

    # 同时保存特征列表供 app 参考
    feat_path = DATA_MODELS / "feature_cols.txt"
    with open(feat_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"特征列表: {feat_path}")


if __name__ == "__main__":
    main()
