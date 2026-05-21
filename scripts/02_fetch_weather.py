"""
02_fetch_weather.py —— 从 Open-Meteo Archive API 获取气象日数据
输出: data/raw/weather_2024.csv
用法: python scripts/02_fetch_weather.py [--start 2024-01-01] [--end 2024-12-31]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from config import DATA_RAW, CITY_COORDS, WEATHER_VARIABLES


def fetch_city_weather(city: str, lat: float, lon: float,
                       start: str, end: str, client) -> pd.DataFrame:
    """获取单个城市的气象日数据"""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": WEATHER_VARIABLES,
    }
    responses = client.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )
    daily = responses[0].Daily()

    data = {"date": pd.date_range(start, end, freq="D")}
    for i, var_name in enumerate(WEATHER_VARIABLES):
        vals = daily.Variables(i).ValuesAsNumpy()
        # 处理长度不匹配 (闰年等情况)
        if len(vals) < len(data["date"]):
            vals = list(vals) + [None] * (len(data["date"]) - len(vals))
        data[var_name] = vals

    df = pd.DataFrame(data)
    df["city"] = city
    return df


def main():
    parser = argparse.ArgumentParser(description="下载气象历史数据")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    # 请求缓存 (避免重复请求浪费API配额)
    cache_path = str(DATA_RAW.parent.parent / ".cache" / "weather")
    cache = requests_cache.CachedSession(cache_path, expire_after=3600)
    retry_session = retry(cache, retries=3, backoff_factor=0.5)
    client = openmeteo_requests.Client(session=retry_session)

    all_data = []
    cities = list(CITY_COORDS.keys())

    for i, city in enumerate(cities):
        lat, lon = CITY_COORDS[city]
        print(f"[{i+1}/{len(cities)}] 下载 {city} 气象数据 ({lat}, {lon})...")
        try:
            df = fetch_city_weather(city, lat, lon, args.start, args.end, client)
            all_data.append(df)
            print(f"  ✓ {len(df)} 天")
        except Exception as e:
            print(f"  ✗ 失败: {str(e)[:100]}")

    if not all_data:
        print("错误: 未获取到任何气象数据!")
        sys.exit(1)

    full = pd.concat(all_data, ignore_index=True)
    out = DATA_RAW / "weather_2024.csv"
    full.to_csv(out, index=False)
    print(f"已保存: {out} ({len(full)} 行 × {len(full.columns)} 列)")

    # 缺失统计
    for col in WEATHER_VARIABLES:
        missing = full[col].isna().sum()
        if missing > 0:
            print(f"  {col} 缺失: {missing}/{len(full)}")


if __name__ == "__main__":
    main()
