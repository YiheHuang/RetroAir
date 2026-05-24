"""
03_fetch_aod.py —— 从 CAMS EAC4 提取站点日尺度 AOD 特征
输出: data/processed/satellite_features.csv

用法:
  python scripts/03_fetch_aod.py --start 2024-01-01 --end 2024-12-31
  python scripts/03_fetch_aod.py --start 2020-01-01 --end 2024-12-31

前置条件:
  1. 注册 ADS 账号: https://ads.atmosphere.copernicus.eu
  2. 获取 API key, 创建 ~/.cdsapirc:
     url: https://ads.atmosphere.copernicus.eu/api
     key: <你的key>
  3. 在 ADS 网页接受 cams-global-reanalysis-eac4 使用许可
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CAMS_AREA,
    CAMS_CDS_VARIABLES,
    CAMS_DATASET,
    CAMS_NC_VARIABLES,
    CAMS_TIMES,
    CITY_COORDS,
    DATA_PROCESSED,
    DATA_RAW,
    SATELLITE_FEATURES,
)

# CAMS NetCDF 中的缺失值标记
_FILL_VALUE = np.float32(3.4028234663852886e38)


def _download_cams_year(year: int, out_dir: Path) -> Path:
    """下载单年 CAMS EAC4 NetCDF，若已存在则跳过。"""
    out_path = out_dir / f"cams_eac4_china_{year}.nc"
    if out_path.exists():
        print(f"  已缓存: {out_path}")
        return out_path

    print(f"  下载 CAMS EAC4 {year} (约 185 MB)...")
    import cdsapi

    client = cdsapi.Client(quiet=True)
    request = {
        "variable": list(CAMS_CDS_VARIABLES),
        "date": f"{year}-01-01/{year}-12-31",
        "time": list(CAMS_TIMES),
        "data_format": "netcdf",
        "area": CAMS_AREA,
    }
    client.retrieve(CAMS_DATASET, request).download(str(out_path))
    print(f"  已保存: {out_path}")
    return out_path


def load_station_points(aqi_path: Path | None = None) -> list[dict]:
    """从 AQI 数据读取站点经纬度，缺失时回退到城市中心代理点。"""
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


def _replace_fill(arr: np.ndarray) -> np.ndarray:
    """将 NetCDF 缺失值替换为 NaN。"""
    arr = arr.astype(np.float64)
    arr[arr > 1e37] = np.nan
    return arr


def _daily_mean_from_ds(ds, lat: float, lon: float, date_str: str) -> dict[str, float]:
    """从 xarray Dataset 为单个 (lat, lon, date) 提取日均 AOD 值。"""
    result = {}
    for nc_var in CAMS_NC_VARIABLES:
        if nc_var not in ds:
            result[nc_var] = np.nan
            continue
        try:
            # sel 日期+最近网格点，对当天时次取均值
            day_data = ds[nc_var].sel(
                valid_time=date_str, latitude=lat, longitude=lon, method="nearest"
            )
            vals = _replace_fill(day_data.values)
            result[nc_var] = float(np.nanmean(vals))
        except Exception:
            result[nc_var] = np.nan

    # Angstrom 指数 AE(469,865)
    aod_469 = result.get("aod469", np.nan)
    aod_865 = result.get("aod865", np.nan)
    if not np.isnan(aod_469) and not np.isnan(aod_865) and aod_469 > 0 and aod_865 > 0:
        result["_ae_469_865"] = float(-np.log(aod_469 / aod_865) / np.log(469.0 / 865.0))
    else:
        result["_ae_469_865"] = np.nan

    return result


def _extract_one_station(
    city: str, station_name: str, lat: float, lon: float,
    start: str, end: str, dss: dict[int, "xr.Dataset"],
) -> list[dict]:
    """为单个站点提取全时段 AOD 特征，返回行列表。"""
    date_list = pd.date_range(start, end, freq="D")
    rows = []
    ok = 0
    for d in date_list:
        date_str = d.strftime("%Y-%m-%d")
        year = d.year
        ds = dss.get(year)
        if ds is None:
            continue
        daily = _daily_mean_from_ds(ds, lat, lon, date_str)
        feat = {
            "date": date_str,
            "city": city,
            "station_name": station_name,
            "lat": lat,
            "lon": lon,
            "aod_550": daily.get("aod550", np.nan),
            "aod_469": daily.get("aod469", np.nan),
            "aod_865": daily.get("aod865", np.nan),
            "ae_469_865": daily.get("_ae_469_865", np.nan),
            "dust_aod": daily.get("duaod550", np.nan),
            "sulphate_aod": daily.get("suaod550", np.nan),
            "bc_aod": daily.get("bcaod550", np.nan),
            "om_aod": daily.get("omaod550", np.nan),
            "ss_aod": daily.get("ssaod550", np.nan),
        }
        rows.append(feat)
        if not np.isnan(feat["aod_550"]):
            ok += 1
    return rows, ok


def main():
    parser = argparse.ArgumentParser(description="从 CAMS EAC4 提取站点 AOD 特征")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--max-stations", type=int, default=0, help="测试用，0=全部")
    parser.add_argument("--aqi", default="", help="输入的 AQI 文件路径")
    parser.add_argument("--out", default="", help="输出文件路径")
    parser.add_argument(
        "--no-download", action="store_true",
        help="跳过下载，仅从已缓存的 NetCDF 提取",
    )
    parser.add_argument("--nc-dir", default="", help="NetCDF 缓存目录")
    args = parser.parse_args()

    nc_dir = Path(args.nc_dir) if args.nc_dir else DATA_RAW / "cams_nc"
    nc_dir.mkdir(parents=True, exist_ok=True)

    start_year = int(args.start[:4])
    end_year = int(args.end[:4])

    # 步骤 1: 下载 NetCDF
    if not args.no_download:
        for year in range(start_year, end_year + 1):
            _download_cams_year(year, nc_dir)

    # 步骤 2: 加载站点
    aqi_path = Path(args.aqi) if args.aqi else DATA_RAW / "aqi_2024.csv"
    # 自动补 .csv 后缀
    if not aqi_path.exists() and aqi_path.suffix != ".csv":
        aqi_path = aqi_path.with_suffix(".csv")
    if not aqi_path.exists():
        print(f"错误: AQI 文件不存在: {aqi_path}")
        sys.exit(1)
    stations = load_station_points(aqi_path)
    if args.max_stations > 0:
        stations = stations[: args.max_stations]
    print(f"AQI 文件: {aqi_path}")
    print(f"站点数: {len(stations)}")

    # 步骤 3: 加载 NetCDF
    import xarray as xr

    dss: dict[int, "xr.Dataset"] = {}
    for year in range(start_year, end_year + 1):
        nc_path = nc_dir / f"cams_eac4_china_{year}.nc"
        if not nc_path.exists():
            print(f"错误: NetCDF 文件不存在: {nc_path}")
            sys.exit(1)
        dss[year] = xr.open_dataset(nc_path)
        print(f"已加载: {nc_path}")

    # 步骤 4: 逐站提取
    print(f"\n提取 AOD 特征: {args.start} → {args.end}")
    all_rows = []
    for i, s in enumerate(stations):
        city = s["city"]
        station_name = s["station_name"]
        lat = float(s["lat"])
        lon = float(s["lon"])
        rows, ok = _extract_one_station(city, station_name, lat, lon, args.start, args.end, dss)
        all_rows.extend(rows)
        total_days = len(rows)
        print(f"  [{i+1}/{len(stations)}] {city}/{station_name}  OK {ok}/{total_days} 天")

    if not all_rows:
        print("错误: 未提取到任何 AOD 特征!")
        sys.exit(1)

    for ds in dss.values():
        ds.close()

    df = pd.DataFrame(all_rows)
    out = Path(args.out) if args.out else DATA_PROCESSED / "satellite_features.csv"
    df.to_csv(out, index=False)
    print(f"\n已保存: {out} ({df.shape[0]} 行 x {df.shape[1]} 列)")
    coverage = df["aod_550"].notna().mean()
    print(f"AOD_550 覆盖率: {coverage:.1%}")
    for c in SATELLITE_FEATURES:
        na = df[c].isna().sum()
        if na > 0:
            print(f"  {c} 缺失: {na}/{len(df)}")


if __name__ == "__main__":
    main()
