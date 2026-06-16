"""
00_build_dataset.py —— 按年份+城市构建完整数据集

用法:
  python scripts/00_build_dataset.py --start-year 2020 --end-year 2024 --city 上海
  python scripts/00_build_dataset.py --start-year 2018 --end-year 2025 --city 北京

流程:
  1. 逐年下载 AQI → data/raw/aqi/aqi_{year}_{city}.csv
  2. 逐年下载气象 → data/raw/weather/weather_{year}_{city}.csv
  3. 逐年提取 AOD → data/processed/aod/satellite_features_{year}_{city}.csv
  4. 合并三源 → data/processed/merged/merged_{start}_{end}_{city}.csv
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# 子目录
DATA_AQI = DATA_RAW / "aqi"
DATA_WEATHER = DATA_RAW / "weather"
DATA_AOD = DATA_PROCESSED / "aod"
DATA_MERGED = DATA_PROCESSED / "merged"


def run_script(script_name: str, extra_args: list[str]):
    cmd = [sys.executable, str(SCRIPTS / script_name), *extra_args]
    print(f"\n>>> {script_name}")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"{script_name} 失败, exit={proc.returncode}")


def main():
    parser = argparse.ArgumentParser(description="按年份+城市构建数据集")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--city", required=True, help="城市名，如 北京、上海")
    parser.add_argument("--workers", type=int, default=12, help="并行线程数")
    parser.add_argument("--skip-aqi", action="store_true")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--skip-aod", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    args = parser.parse_args()

    city = args.city
    start_year = args.start_year
    end_year = args.end_year
    years = list(range(start_year, end_year + 1))

    for d in [DATA_AQI, DATA_WEATHER, DATA_AOD, DATA_MERGED]:
        d.mkdir(parents=True, exist_ok=True)

    city_slug = city

    for year in years:
        start = f"{year}-01-01"
        end = f"{year}-12-31"

        # 1) AQI
        if not args.skip_aqi:
            aqi_out = DATA_AQI / f"aqi_{year}_{city_slug}.csv"
            if aqi_out.exists():
                print(f"跳过 AQI {year} (已存在: {aqi_out})")
            else:
                run_script("01_fetch_aqi.py", [
                    "--start", start, "--end", end,
                    "--cities", city,
                    "--workers", str(args.workers),
                    "--out", str(aqi_out),
                ])

        # 2) Weather
        if not args.skip_weather:
            weather_out = DATA_WEATHER / f"weather_{year}_{city_slug}.csv"
            if weather_out.exists():
                print(f"跳过 Weather {year} (已存在: {weather_out})")
            else:
                run_script("02_fetch_weather.py", [
                    "--start", start, "--end", end,
                    "--aqi", str(DATA_AQI / f"aqi_{year}_{city_slug}.csv"),
                    "--workers", str(max(1, args.workers // 2)),
                    "--out", str(weather_out),
                ])

        # 3) AOD
        if not args.skip_aod:
            aod_out = DATA_AOD / f"satellite_features_{year}_{city_slug}.csv"
            if aod_out.exists():
                print(f"跳过 AOD {year} (已存在: {aod_out})")
            else:
                run_script("03_fetch_aod.py", [
                    "--start", start, "--end", end,
                    "--aqi", str(DATA_AQI / f"aqi_{year}_{city_slug}.csv"),
                    "--out", str(aod_out),
                ])

    # 4) Merge
    if not args.skip_merge:
        merged_out = DATA_MERGED / f"merged_{start_year}_{end_year}_{city_slug}.csv"
        if merged_out.exists():
            print(f"跳过 Merge (已存在: {merged_out})")
        else:
            aqi_files = [str(DATA_AQI / f"aqi_{y}_{city_slug}.csv") for y in years]
            weather_files = [str(DATA_WEATHER / f"weather_{y}_{city_slug}.csv") for y in years]
            aod_files = [str(DATA_AOD / f"satellite_features_{y}_{city_slug}.csv") for y in years]

            run_script("04_merge_data.py", [
                "--aqi-list", *aqi_files,
                "--weather-list", *weather_files,
                "--aod-list", *aod_files,
                "--out", str(merged_out),
            ])

    print(f"\n===== 数据集构建完成 =====")
    print(f"合并文件: {merged_out}")


if __name__ == "__main__":
    main()
