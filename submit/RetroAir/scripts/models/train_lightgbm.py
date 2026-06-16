import argparse
import sys
from pathlib import Path

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.data_utils import load_prepared_data, metrics_report, set_seed
from scripts.models.runner_utils import save_model, write_model_outputs

MODEL_NAME = "LightGBM"


def run(seed: int, split_seed: int, output_dir: Path, data_path: Path | None = None):
    set_seed(seed)
    data = load_prepared_data(data_path=data_path, split_seed=split_seed)

    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        num_leaves=63,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        data.fused_train, data.y_train,
        eval_set=[(data.fused_val, data.y_val)],
        eval_metric="rmse",
    )
    y_pred = model.predict(data.fused_test)
    metrics = metrics_report(data.y_test, y_pred)
    write_model_outputs(output_dir, MODEL_NAME, seed, metrics, data.y_test, y_pred)
    save_model(model, output_dir, MODEL_NAME, seed)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", default="", help="合并数据集路径")
    args = parser.parse_args()

    dp = Path(args.data_path) if args.data_path else None
    metrics = run(args.seed, args.split_seed, args.output_dir, data_path=dp)
    print(f"{MODEL_NAME}: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
