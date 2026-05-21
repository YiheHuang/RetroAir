"""
01_fetch_aqi.py —— 从 Open-Meteo Air Quality API 获取 PM2.5 日均数据
输出: data/raw/aqi_2024.csv
用法: python scripts/01_fetch_aqi.py [--start 2024-01-01] [--end 2024-12-31]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from config import DATA_RAW, CITY_COORDS, AQI_VARIABLES


def fetch_city_aqi(city: str, lat: float, lon: float,
                   start: str, end: str, client) -> pd.DataFrame:
    """获取单个城市的历史空气质量数据 (日均聚合)"""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": AQI_VARIABLES,
    }
    responses = client.weather_api(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params=params,
    )
    hourly = responses[0].Hourly()
    date_range = pd.date_range(start, end, freq="D")
    data = {"date": date_range.strftime("%Y-%m-%d")}

    for i, var_name in enumerate(AQI_VARIABLES):
        vals = hourly.Variables(i).ValuesAsNumpy()
        n_days = len(vals) // 24
        vals = vals[:n_days * 24]
        daily_mean = vals.reshape(n_days, 24).mean(axis=1)
        if len(daily_mean) < len(date_range):
            daily_mean = np.pad(daily_mean, (0, len(date_range) - len(daily_mean)),
                                constant_values=np.nan)
        elif len(daily_mean) > len(date_range):
            daily_mean = daily_mean[:len(date_range)]
        data[var_name] = daily_mean

    df = pd.DataFrame(data)
    df["city"] = city
    return df


def main():
    parser = argparse.ArgumentParser(description="下载AQI历史数据 (Open-Meteo)")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--cities", default=None)
    args = parser.parse_args()

    cities = args.cities.split(",") if args.cities else list(CITY_COORDS.keys())
    print(f"下载 AQI: {args.start} → {args.end}, {len(cities)} 城")

    cache = requests_cache.CachedSession(
        str(DATA_RAW.parent.parent / ".cache" / "aqi"), expire_after=3600
    )
    client = openmeteo_requests.Client(session=retry(cache, retries=3, backoff_factor=0.5))

    all_data = []
    for i, city in enumerate(cities):
        lat, lon = CITY_COORDS[city]
        print(f"[{i+1}/{len(cities)}] {city}...")
        try:
            df = fetch_city_aqi(city, lat, lon, args.start, args.end, client)
            all_data.append(df)
            print(f"  OK {len(df)}d, PM2.5={df['pm2_5'].mean():.1f}")
        except Exception as e:
            print(f"  ERR: {str(e)[:120]}")

    if not all_data:
        print("错误: 未获取到任何数据!")
        sys.exit(1)

    full = pd.concat(all_data, ignore_index=True)
    out = DATA_RAW / "aqi_2024.csv"
    full.to_csv(out, index=False)
    print(f"已保存: {out} ({len(full)} 行)")


if __name__ == "__main__":
    main()
