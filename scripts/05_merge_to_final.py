"""
merge_to_final.py —— 合并各城市 MERGED 数据集为 FINAL 多城数据集

用法:
  # 合并所有城市
  python scripts/merge_to_final.py --input \"data/processed/merged/merged_*.csv\" \\
      --out data/processed/final/merged_all_cities.csv

  # 合并指定城市
  python scripts/merge_to_final.py --input \"data/processed/merged/merged_2018_2025_北京.csv\" \\
      --input \"data/processed/merged/merged_2018_2025_上海.csv\" \\
      --out data/processed/final/merged_京沪.csv
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_FINAL, PM25_COL


def main():
    parser = argparse.ArgumentParser(description="合并各城市 MERGED 数据集 → FINAL")
    parser.add_argument("--input", nargs="+", required=True,
                        help="输入文件路径（支持 glob 通配符）")
    parser.add_argument("--out", default="", help="输出文件路径")
    parser.add_argument("--dedup-subset", nargs="+",
                        default=["city", "station_name", "date"],
                        help="去重键")
    args = parser.parse_args()

    # 展开 glob
    files = []
    for p in args.input:
        matched = sorted(glob.glob(p, recursive=True))
        if matched:
            files.extend(matched)
        elif Path(p).exists():
            files.append(p)

    if not files:
        print("错误: 未找到任何输入文件")
        sys.exit(1)

    print(f"输入文件 ({len(files)} 个):")
    for f in files:
        print(f"  {f}")

    # 加载合并
    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
        print(f"  → {Path(f).name}: {len(df)} 行, {df['city'].nunique()} 城, "
              f"{df['station_name'].nunique()} 站, "
              f"{df['date'].min()} ~ {df['date'].max()}")

    merged = pd.concat(frames, ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=args.dedup_subset, keep="last")
    if n_before > len(merged):
        print(f"去重: {n_before} → {len(merged)} (丢弃 {n_before - len(merged)} 行)")

    merged = merged.sort_values(["city", "station_name", "date"]).reset_index(drop=True)

    # 输出
    out = Path(args.out) if args.out else DATA_FINAL / "merged_dataset.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    print(f"\n===== FINAL 数据集: {out} =====")
    print(f"总样本: {len(merged)}")
    print(f"城市: {merged['city'].nunique()} / 站点: {merged['station_name'].nunique()}")
    print(f"日期范围: {merged['date'].min()} → {merged['date'].max()}")
    print(f"总列数: {len(merged.columns)}")

    if PM25_COL in merged.columns:
        pm = merged[PM25_COL]
        print(f"PM2.5: mean={pm.mean():.1f}, std={pm.std():.1f}, "
              f"min={pm.min():.1f}, max={pm.max():.1f}")

    print(f"\n各城市数据量:")
    for city, grp in merged.groupby("city"):
        n_stations = grp["station_name"].nunique()
        print(f"  {city}: {len(grp)} 行, {n_stations} 站")


if __name__ == "__main__":
    main()
