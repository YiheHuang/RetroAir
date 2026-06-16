"""
04_merge_data.py —— AQI + 气象 + AOD 三源数据合并
合并键: (city, station_name, date)，inner join

用法:
  python scripts/04_merge_data.py --aqi-list data/raw/aqi/aqi_*.csv \\
      --weather-list data/raw/weather/weather_*.csv \\
      --aod-list data/processed/aod/satellite_features_*.csv \\
      --out data/processed/merged/merged_2018_2025_北京.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import glob

import numpy as np
import pandas as pd

from config import (
    DATA_AQI,
    DATA_WEATHER,
    DATA_AOD,
    DATA_MERGED,
    PM25_COL,
    SATELLITE_FEATURES,
    WEATHER_VARIABLES,
)


def _load_multi(paths: list[str]) -> pd.DataFrame:
    """加载并合并多个文件（支持 glob 通配符）。"""
    frames = []
    for p in paths:
        pp = Path(p)
        if pp.exists():
            frames.append(pd.read_csv(pp))
        else:
            for matched in sorted(glob.glob(p)):
                frames.append(pd.read_csv(matched))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["city", "station_name", "date"], keep="last")
    return df


def main():
    parser = argparse.ArgumentParser(description="合并 AQI/气象/AOD 三源数据")
    parser.add_argument("--aqi", default="", help="AQI 文件路径或 glob")
    parser.add_argument("--aqi-list", nargs="+", default=[], help="多个 AQI 文件")
    parser.add_argument("--weather", default="", help="气象文件路径或 glob")
    parser.add_argument("--weather-list", nargs="+", default=[], help="多个气象文件")
    parser.add_argument("--aod", nargs="+", default=[], help="AOD 文件路径（可多个，支持通配符）")
    parser.add_argument("--aod-list", nargs="+", default=[], help="多个 AOD 文件")
    parser.add_argument("--out", default="", help="输出文件路径")
    args = parser.parse_args()

    # 收集路径
    aqi_paths = list(args.aqi_list)
    if args.aqi:
        aqi_paths.append(args.aqi)
    if not aqi_paths:
        aqi_paths = [str(DATA_AQI / "aqi_2024_北京.csv")]

    weather_paths = list(args.weather_list)
    if args.weather:
        weather_paths.append(args.weather)
    if not weather_paths:
        weather_paths = [str(DATA_WEATHER / "weather_2024_北京.csv")]

    aod_paths = list(args.aod_list)
    if args.aod:
        aod_paths.extend(args.aod)
    if not aod_paths:
        aod_paths = [str(DATA_AOD / "satellite_features_2024_北京.csv")]

    # 加载
    aqi = _load_multi(aqi_paths)
    weather = _load_multi(weather_paths)
    aod = _load_multi(aod_paths)

    if aqi.empty or weather.empty or aod.empty:
        print("错误: 至少一个数据源为空!")
        sys.exit(1)

    print(f"AQI:      {aqi.shape[0]:>6} 行, {aqi['city'].nunique()} 城, {aqi['station_name'].nunique()} 站")
    print(f"Weather:  {weather.shape[0]:>6} 行, {weather['city'].nunique()} 城, {weather['station_name'].nunique()} 站")
    print(f"AOD:      {aod.shape[0]:>6} 行, {aod['city'].nunique()} 城, {aod['station_name'].nunique()} 站")

    # 验证必要字段
    required = ["city", "station_name", "lat", "lon"]
    for name, df in [("AQI", aqi), ("Weather", weather), ("AOD", aod)]:
        missing = [c for c in required + ["date"] if c not in df.columns]
        if missing:
            print(f"错误: {name} 缺少字段: {missing}")
            sys.exit(1)

    # 统一日期格式
    aqi["date"] = pd.to_datetime(aqi["date"]).dt.strftime("%Y-%m-%d")
    weather["date"] = pd.to_datetime(weather["date"]).dt.strftime("%Y-%m-%d")
    aod["date"] = pd.to_datetime(aod["date"]).dt.strftime("%Y-%m-%d")

    n_before = len(aqi)
    merged = aqi.merge(
        weather,
        on=["city", "station_name", "date"],
        how="inner",
        suffixes=("", "_weather"),
    )
    print(f"AQI ∩ Weather: {len(merged)} 行 (丢弃 {n_before - len(merged)} 行)")

    # 处理坐标冲突
    for c in ["lat", "lon"]:
        wc = f"{c}_weather"
        if wc in merged.columns:
            merged[c] = merged[c].fillna(merged[wc])
            merged = merged.drop(columns=[wc])

    n_before_aod = len(merged)
    merged = merged.merge(
        aod,
        on=["city", "station_name", "date"],
        how="inner",
        suffixes=("", "_aod"),
    )
    print(f"+ AOD (inner): {len(merged)} 行 (丢弃 {n_before_aod - len(merged)} 行)")

    # 处理 AOD 列名冲突
    for dc in [c for c in merged.columns if c.endswith("_aod")]:
        base = dc.replace("_aod", "")
        if base in merged.columns:
            merged[base] = merged[base].fillna(merged[dc])
            merged = merged.drop(columns=[dc])

    # 时间特征
    merged["day_of_year"] = pd.to_datetime(merged["date"]).dt.dayofyear

    # 丢弃 PM2.5 缺失的行
    merged = merged.dropna(subset=[PM25_COL])

    # 丢弃 AOD 或气象特征缺失的行
    feature_cols = [c for c in SATELLITE_FEATURES + WEATHER_VARIABLES if c in merged.columns]
    n_before_drop = len(merged)
    merged = merged.dropna(subset=feature_cols)
    if n_before_drop > len(merged):
        print(f"丢弃特征缺失行: {n_before_drop - len(merged)} 行")

    merged = merged.sort_values(["city", "station_name", "date"]).reset_index(drop=True)

    out = Path(args.out) if args.out else DATA_MERGED / "merged_dataset.csv"
    merged.to_csv(out, index=False)

    # 输出概况
    print(f"\n===== 合并数据集: {out} =====")
    print(f"总样本: {len(merged)}")
    print(f"城市: {merged['city'].nunique()}, 站点: {merged['station_name'].nunique()}")
    print(f"日期范围: {merged['date'].min()} → {merged['date'].max()}")
    print(f"特征: {len(feature_cols)} 列 (AOD {len(SATELLITE_FEATURES)} + 气象 {len(WEATHER_VARIABLES)} + 时间)")

    print(f"\nPM2.5 标签分布:")
    pm = merged[PM25_COL]
    print(f"  min={pm.min():.1f}, Q1={pm.quantile(0.25):.1f}, median={pm.median():.1f}, "
          f"Q3={pm.quantile(0.75):.1f}, max={pm.max():.1f}, mean={pm.mean():.1f}")

    print(f"\nAOD 特征摘要:")
    for c in SATELLITE_FEATURES:
        if c in merged.columns:
            s = merged[c]
            print(f"  {c:<14s}: mean={s.mean():.4f}, std={s.std():.4f}, "
                  f"min={s.min():.4f}, max={s.max():.4f}")

    print(f"\n气象特征摘要:")
    for c in WEATHER_VARIABLES:
        if c in merged.columns:
            s = merged[c]
            print(f"  {c:<35s}: mean={s.mean():.2f}, std={s.std():.2f}")

    print(f"\n各站数据量:")
    station_counts = merged.groupby("station_name").size().sort_values(ascending=False)
    for name, cnt in station_counts.items():
        print(f"  {name:<12s}: {cnt} 天")


if __name__ == "__main__":
    main()
