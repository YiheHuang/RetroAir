"""
01_fetch_aqi.py —— 下载真实监测站 PM2.5 标签数据
数据源:
1) AKShare 真气网观测点接口 (air_quality_watch_point)
2) 公开监测站坐标清单 (GitHub: qwd/LocationList)
输出: data/raw/aqi_2024.csv
用法: python scripts/01_fetch_aqi.py [--start 2024-01-01] [--end 2024-12-31]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd
from config import DATA_RAW, CITY_COORDS

STATION_CATALOG_URL = (
    "https://raw.githubusercontent.com/qwd/LocationList/master/"
    "POI-Air-Monitoring-Station-List-latest.csv"
)
STATION_CATALOG_LOCAL = DATA_RAW / "station_catalog.csv"


def normalize_station_name(x: str) -> str:
    if x is None:
        return ""
    text = str(x).strip()
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace(" ", "")
    return text


def _load_station_catalog_raw(refresh: bool = False) -> pd.DataFrame:
    """优先加载本地站点清单，缺失或刷新时再从 GitHub 下载"""
    if STATION_CATALOG_LOCAL.exists() and (not refresh):
        print(f"站点清单: 使用本地缓存 {STATION_CATALOG_LOCAL}")
        return pd.read_csv(STATION_CATALOG_LOCAL)

    print("站点清单: 从 GitHub 下载最新版本 ...")
    raw_remote = pd.read_csv(STATION_CATALOG_URL, skiprows=1)
    raw_remote.to_csv(STATION_CATALOG_LOCAL, index=False)
    print(f"站点清单: 已缓存到本地 {STATION_CATALOG_LOCAL}")
    return raw_remote


def load_station_catalog(cities: list[str], refresh: bool = False) -> pd.DataFrame:
    """加载真实监测站经纬度清单（公开POI）"""
    raw = _load_station_catalog_raw(refresh=refresh)
    catalog = raw.rename(
        columns={
            "POI_Name": "station_name",
            "Location_Name_ZH": "city",
            "POI_Latitude": "lat",
            "POI_Longitude": "lon",
            "POI_ID": "station_poi_id",
        }
    )
    catalog = catalog[["city", "station_name", "lat", "lon", "station_poi_id"]].copy()
    catalog["city"] = catalog["city"].astype(str).str.replace("市", "", regex=False)
    catalog["station_name_norm"] = catalog["station_name"].map(normalize_station_name)
    catalog = catalog[catalog["city"].isin(cities)].drop_duplicates(
        subset=["city", "station_name_norm"], keep="first"
    )
    return catalog


def fetch_city_day_watch_point(city: str, date_yyyymmdd: str) -> pd.DataFrame:
    """抓取某城市某天的真实监测站观测数据"""
    df = ak.air_quality_watch_point(
        city=city,
        start_date=date_yyyymmdd,
        end_date=date_yyyymmdd,
    )
    if df.empty:
        return df
    out = df.copy()
    out["city"] = city
    out["date"] = pd.to_datetime(date_yyyymmdd, format="%Y%m%d").strftime("%Y-%m-%d")
    out["station_name"] = out["pointname"].astype(str)
    out["station_name_norm"] = out["station_name"].map(normalize_station_name)
    return out


def _fetch_one_day_safe(city: str, day_str: str, city_catalog: pd.DataFrame) -> pd.DataFrame | None:
    """抓取单日 AQI 数据（线程安全，由 ThreadPoolExecutor 调用）。"""
    try:
        day_df = fetch_city_day_watch_point(city, day_str)
        if day_df.empty:
            return None
        merged = day_df.merge(
            city_catalog[["city", "station_name_norm", "lat", "lon", "station_poi_id"]],
            on=["city", "station_name_norm"],
            how="left",
        )
        merged["pm2_5"] = pd.to_numeric(merged.get("pm2_5"), errors="coerce")
        merged["pm10"] = pd.to_numeric(merged.get("pm10"), errors="coerce")
        merged["nitrogen_dioxide"] = pd.to_numeric(merged.get("no2"), errors="coerce")
        merged["ozone"] = pd.to_numeric(merged.get("o3"), errors="coerce")
        merged["european_aqi"] = pd.to_numeric(merged.get("aqi"), errors="coerce")
        keep_cols = [
            "date", "city", "station_name", "lat", "lon",
            "station_poi_id", "pm2_5", "pm10", "nitrogen_dioxide", "ozone", "european_aqi",
        ]
        return merged[keep_cols]
    except Exception:
        return None


def build_real_station_aqi(
    cities: list[str],
    start: str,
    end: str,
    sleep_sec: float = 0.0,
    refresh_station_catalog: bool = False,
    workers: int = 12,
) -> pd.DataFrame:
    catalog = load_station_catalog(cities, refresh=refresh_station_catalog)
    if catalog.empty:
        raise RuntimeError("站点坐标清单为空，无法继续")

    dates = pd.date_range(start, end, freq="D")
    all_rows = []

    for ci, city in enumerate(cities):
        print(f"\n[{ci+1}/{len(cities)}] 城市: {city}")
        city_catalog = catalog[catalog["city"] == city].copy()
        print(f"  站点坐标清单: {len(city_catalog)} 个")

        day_strs = [d.strftime("%Y%m%d") for d in dates]
        total_days = len(day_strs)

        # 主线程先做一次"预热"调用，让 V8 引擎完成初始化
        print("  预热 V8 引擎 ...")
        _ = _fetch_one_day_safe(city, day_strs[0], city_catalog)
        print(f"  启动 {workers} 个线程并行抓取 {total_days} 天 ...")

        done, err = 0, 0
        collected_prev = 0

        # 剩余天数从第 2 天开始（第 1 天已预热）
        remaining = day_strs[1:] if len(day_strs) > 1 else []
        tasks = [(city, ds, city_catalog) for ds in remaining]

        if total_days == 1:
            # 如果只有一天，预热调用的结果就是最终结果
            warmup_result = _fetch_one_day_safe(city, day_strs[0], city_catalog)
            if warmup_result is not None:
                all_rows.append(warmup_result)
            done, err = (1, 0) if warmup_result is not None else (0, 1)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_fetch_one_day_safe, c, ds, cat): ds
                for c, ds, cat in tasks
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    all_rows.append(result)
                    done += 1
                else:
                    err += 1
                total_done = done + err
                collected = sum(len(r) for r in all_rows)
                # 每 30 天输出一次进度
                if total_done % 30 == 0 or total_done == total_days - 1:
                    days_done = total_done + 1  # +1 for warmup day
                    new_rows = collected - collected_prev
                    collected_prev = collected
                    print(
                        f"  进度 {days_done}/{total_days} 天 "
                        f"(OK {done+1 if total_days > 1 else done}, 空/错 {err}), "
                        f"累计 {collected} 条 (+{new_rows})"
                    )

    if not all_rows:
        raise RuntimeError("未抓取到任何真实监测站AQI数据")

    full = pd.concat(all_rows, ignore_index=True)
    full = full.dropna(subset=["pm2_5", "lat", "lon"])
    full = full.sort_values(["city", "station_name", "date"]).reset_index(drop=True)
    return full


def main():
    parser = argparse.ArgumentParser(description="下载真实监测站 AQI 历史数据")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--cities", default=None, help="逗号分隔城市名，如 北京,上海")
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        help="线程池完成后的缓冲等待秒数（并行模式下通常无需等待）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=15,
        help="并行线程数，默认 15",
    )
    parser.add_argument(
        "--out",
        default="",
        help="输出文件路径（默认 data/raw/aqi_2024.csv）",
    )
    parser.add_argument(
        "--refresh-station-catalog",
        action="store_true",
        help="强制重新下载站点清单并覆盖本地缓存",
    )
    args = parser.parse_args()

    cities = args.cities.split(",") if args.cities else list(CITY_COORDS.keys())
    cities = [c.strip() for c in cities if c.strip()]
    print(f"下载真实站点 AQI: {args.start} → {args.end}, {len(cities)} 城")

    full = build_real_station_aqi(
        cities,
        args.start,
        args.end,
        sleep_sec=args.sleep_sec,
        refresh_station_catalog=args.refresh_station_catalog,
        workers=args.workers,
    )

    out = Path(args.out) if args.out else DATA_RAW / "aqi_2024.csv"
    full.to_csv(out, index=False)
    print(f"\n已保存: {out} ({len(full)} 行)")
    print(
        f"城市: {full['city'].nunique()} 个, "
        f"站点: {full['station_name'].nunique()} 个, "
        f"日期: {full['date'].min()} -> {full['date'].max()}"
    )
    print(f"PM2.5 均值: {full['pm2_5'].mean():.2f}")


if __name__ == "__main__":
    main()
