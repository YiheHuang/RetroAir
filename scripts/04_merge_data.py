"""
04_merge_data.py —— AQI + 气象 + OSM 三源数据合并
合并键: (city, date) + city
输出: data/processed/merged_dataset.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config import DATA_RAW, DATA_PROCESSED, PM25_COL


def main():
    aqi_path = DATA_RAW / "aqi_2024.csv"
    weather_path = DATA_RAW / "weather_2024.csv"
    osm_path = DATA_PROCESSED / "osm_features.csv"

    for p in [aqi_path, weather_path, osm_path]:
        if not p.exists():
            print(f"错误: {p} 不存在!")
            sys.exit(1)

    aqi = pd.read_csv(aqi_path)
    weather = pd.read_csv(weather_path)
    osm = pd.read_csv(osm_path)

    print(f"AQI:      {aqi.shape[0]:>6} 行, {aqi['city'].nunique()} 城")
    print(f"Weather:  {weather.shape[0]:>6} 行, {weather['city'].nunique()} 城")
    print(f"OSM:      {osm.shape[0]:>6} 行, {osm['city'].nunique()} 城")

    aqi["date"] = pd.to_datetime(aqi["date"]).dt.strftime("%Y-%m-%d")
    weather["date"] = pd.to_datetime(weather["date"]).dt.strftime("%Y-%m-%d")

    merged = aqi.merge(weather, on=["city", "date"], how="inner")
    print(f"AQI+Weather: {len(merged)} 行")

    merged = merged.merge(osm, on="city", how="left", suffixes=("", "_osm"))
    for dc in [c for c in merged.columns if c.endswith("_osm")]:
        base = dc.replace("_osm", "")
        if base in merged.columns:
            merged[base] = merged[base].fillna(merged[dc])
            merged = merged.drop(columns=[dc])
    print(f"+ OSM:       {len(merged)} 行")

    before = len(merged)
    merged = merged.dropna(subset=[PM25_COL])
    merged = merged.sort_values(["city", "date"]).reset_index(drop=True)

    out = DATA_PROCESSED / "merged_dataset.csv"
    merged.to_csv(out, index=False)

    print(f"\n合并数据集: {out}")
    print(f"总样本: {len(merged)}, 城市: {merged['city'].nunique()}, "
          f"日期: {merged['date'].min()} → {merged['date'].max()}")
    print(f"特征列: {len(merged.columns)}")
    print(f"PM2.5: {merged[PM25_COL].min():.0f} - {merged[PM25_COL].max():.0f}, "
          f"均值={merged[PM25_COL].mean():.1f}")


if __name__ == "__main__":
    main()
