"""
02_fetch_weather.py —— 获取气象日数据（NASA POWER 主源，Open-Meteo 兜底）
输出: data/raw/weather_2024.csv
用法: python scripts/02_fetch_weather.py [--start 2024-01-01] [--end 2024-12-31]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import openmeteo_requests
import requests_cache
import requests
from retry_requests import retry
from config import (
    DATA_RAW,
    CITY_COORDS,
    WEATHER_VARIABLES,
    WEATHER_CANONICAL_COLS,
    WEATHER_NASA_POWER_MAP,
)


def load_station_points(aqi_path: Path | None = None) -> list[dict]:
    """从 AQI 数据读取站点经纬度，缺失时回退到城市中心代理点"""
    if aqi_path is None:
        aqi_path = DATA_RAW / "aqi_2024.csv"
    if aqi_path.exists():
        aqi = pd.read_csv(aqi_path)
        required_cols = {"city", "station_name", "lat", "lon"}
        if required_cols.issubset(set(aqi.columns)):
            stations = (
                aqi[["city", "station_name", "lat", "lon"]]
                .dropna()
                .drop_duplicates()
                .to_dict("records")
            )
            if stations:
                return stations

    stations = []
    for city, (lat, lon) in CITY_COORDS.items():
        stations.append({
            "city": city,
            "station_name": f"{city}_proxy_station",
            "lat": lat,
            "lon": lon,
        })
    return stations


def fetch_city_weather(city: str, lat: float, lon: float,
                       start: str, end: str, client) -> pd.DataFrame:
    """获取单个城市气象日数据（NASA POWER 主源，失败时 Open-Meteo 兜底）"""
    try:
        return _fetch_city_weather_nasa_power(city, lat, lon, start, end)
    except Exception as e:
        print(f"    NASA POWER 失败，切换 Open-Meteo: {str(e)[:100]}")
        return _fetch_city_weather_open_meteo(city, lat, lon, start, end, client)


def _fetch_city_weather_open_meteo(city: str, lat: float, lon: float,
                                   start: str, end: str, client) -> pd.DataFrame:
    """Open-Meteo Archive API"""
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
    df["weather_source"] = "open-meteo"
    return df


def _fetch_city_weather_nasa_power(city: str, lat: float, lon: float,
                                   start: str, end: str) -> pd.DataFrame:
    """NASA POWER Daily API (无 Key)"""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    start_fmt = pd.to_datetime(start).strftime("%Y%m%d")
    end_fmt = pd.to_datetime(end).strftime("%Y%m%d")
    params = {
        "parameters": ",".join(WEATHER_NASA_POWER_MAP.values()),
        "community": "RE",
        "latitude": lat,
        "longitude": lon,
        "start": start_fmt,
        "end": end_fmt,
        "format": "JSON",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    values = payload.get("properties", {}).get("parameter", {})

    date_idx = pd.date_range(start, end, freq="D")
    out = {"date": date_idx}

    for col in WEATHER_CANONICAL_COLS:
        nasa_key = WEATHER_NASA_POWER_MAP[col]
        series = values.get(nasa_key, {})
        mapped = []
        for d in date_idx:
            k = d.strftime("%Y%m%d")
            val = series.get(k, None)
            if val in (-999, -999.0):
                val = None
            mapped.append(val)
        out[col] = mapped

    df = pd.DataFrame(out)
    df["city"] = city
    df["weather_source"] = "nasa-power"
    return df


def _fetch_one_station(city, station_name, lat, lon, start, end) -> pd.DataFrame | None:
    """单个站点气象数据获取（供线程池调用），不依赖 openmeteo session 缓存。"""
    try:
        df = _fetch_city_weather_nasa_power(city, lat, lon, start, end)
    except Exception:
        try:
            cache_path = str(Path(__file__).resolve().parent.parent / ".cache" / "weather")
            cache = requests_cache.CachedSession(cache_path, expire_after=3600)
            retry_session = retry(cache, retries=3, backoff_factor=0.5)
            client = openmeteo_requests.Client(session=retry_session)
            df = _fetch_city_weather_open_meteo(city, lat, lon, start, end, client)
        except Exception:
            return None
    df["station_name"] = station_name
    df["lat"] = lat
    df["lon"] = lon
    return df


def main():
    parser = argparse.ArgumentParser(description="下载气象历史数据")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--max-stations", type=int, default=0, help="仅用于测试，0表示全部")
    parser.add_argument("--aqi", default="", help="输入的 AQI 文件路径（默认 data/raw/aqi_2024.csv）")
    parser.add_argument("--out", default="", help="输出文件路径（默认 data/raw/weather_2024.csv）")
    parser.add_argument("--workers", type=int, default=8, help="并行线程数")
    args = parser.parse_args()

    aqi_path = Path(args.aqi) if args.aqi else DATA_RAW / "aqi_2024.csv"
    stations = load_station_points(aqi_path)
    if args.max_stations > 0:
        stations = stations[:args.max_stations]
    print(f"按 AQI 站点对齐气象: {len(stations)} 个站点")
    print(f"输入: {aqi_path}")

    all_data = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for s in stations:
            city = s["city"]
            station_name = s["station_name"]
            lat = float(s["lat"])
            lon = float(s["lon"])
            fut = pool.submit(_fetch_one_station, city, station_name, lat, lon, args.start, args.end)
            futures[fut] = (city, station_name)

        done = 0
        for fut in as_completed(futures):
            city, station_name = futures[fut]
            done += 1
            try:
                df = fut.result()
                if df is not None:
                    all_data.append(df)
                    print(f"  [{done}/{len(stations)}] ✓ {city}/{station_name} ({len(df)} 天)")
                else:
                    print(f"  [{done}/{len(stations)}] ✗ {city}/{station_name} 失败")
            except Exception as e:
                print(f"  [{done}/{len(stations)}] ✗ {city}/{station_name} ERR: {str(e)[:80]}")

    if not all_data:
        print("错误: 未获取到任何气象数据!")
        sys.exit(1)

    full = pd.concat(all_data, ignore_index=True)
    out = Path(args.out) if args.out else DATA_RAW / "weather_2024.csv"
    full.to_csv(out, index=False)
    print(f"已保存: {out} ({len(full)} 行 × {len(full.columns)} 列)")
    if "weather_source" in full.columns:
        source_stat = full["weather_source"].value_counts(dropna=False).to_dict()
        print(f"数据源分布: {source_stat}")

    # 缺失统计
    for col in WEATHER_CANONICAL_COLS:
        missing = full[col].isna().sum()
        if missing > 0:
            print(f"  {col} 缺失: {missing}/{len(full)}")


if __name__ == "__main__":
    main()
