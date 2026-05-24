"""
inference.py —— 纯推理管线（无 UI 依赖），供 app.py 和命令行调用

用法:
  python scripts/inference.py --lat 39.9 --lon 116.4 --date 2024-06-15
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CAMS_NC_VARIABLES, DATA_CAMS_NC, TIME_FEATURES,
    WEATHER_CANONICAL_COLS, WEATHER_NASA_POWER_MAP,
)
from scripts.models.nn_defs import ResMLPRegressor

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "models" / "compare"
SCALER_PATH = MODEL_DIR / "inference_scaler.pkl"
MODEL_PATH = MODEL_DIR / "ResMLP_seed42.pt"

AQI_LEVELS = [
    (35, "优", "#00e400"),
    (75, "良", "#ffff00"),
    (115, "轻度污染", "#ff7e00"),
    (150, "中度污染", "#ff0000"),
    (250, "重度污染", "#99004c"),
    (float("inf"), "严重污染", "#7e0023"),
]


def fetch_weather(lat: float, lon: float, target_date: str) -> dict[str, float]:
    """NASA POWER 单日气象数据。"""
    start = pd.to_datetime(target_date).strftime("%Y%m%d")
    params = {
        "parameters": ",".join(WEATHER_NASA_POWER_MAP.values()),
        "community": "RE",
        "latitude": lat, "longitude": lon,
        "start": start, "end": start,
        "format": "JSON",
    }
    resp = requests.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        params=params, timeout=15,
    )
    resp.raise_for_status()
    values = resp.json().get("properties", {}).get("parameter", {})

    result = {}
    for col in WEATHER_CANONICAL_COLS:
        nasa_key = WEATHER_NASA_POWER_MAP[col]
        val = values.get(nasa_key, {}).get(start, None)
        result[col] = np.nan if val in (-999, -999.0, None) else float(val)
    return result


def fetch_aod(lat: float, lon: float, target_date: str) -> dict[str, float]:
    """从 CAMS EAC4 NetCDF 提取单日 AOD。"""
    d = pd.to_datetime(target_date)
    nc_path = DATA_CAMS_NC / f"cams_eac4_china_{d.year}.nc"

    # 回退到最近可用年份
    if not nc_path.exists():
        for y in range(d.year - 1, 2017, -1):
            p = DATA_CAMS_NC / f"cams_eac4_china_{y}.nc"
            if p.exists():
                nc_path = p
                break

    if not nc_path.exists():
        return {var: np.nan for var in [
            "aod550", "aod469", "aod865", "duaod550", "suaod550",
            "bcaod550", "omaod550", "ssaod550",
        ]}

    ds = xr.open_dataset(nc_path)
    raw = {}
    for nc_var in CAMS_NC_VARIABLES:
        if nc_var not in ds:
            raw[nc_var] = np.nan
            continue
        try:
            day_data = ds[nc_var].sel(
                valid_time=target_date, latitude=lat, longitude=lon, method="nearest"
            )
            vals = day_data.values.astype(np.float64)
            vals[vals > 1e37] = np.nan
            raw[nc_var] = float(np.nanmean(vals))
        except Exception:
            raw[nc_var] = np.nan
    ds.close()

    # Angstrom index
    aod469 = raw.get("aod469", np.nan)
    aod865 = raw.get("aod865", np.nan)
    if not np.isnan(aod469) and not np.isnan(aod865) and aod469 > 0 and aod865 > 0:
        ae = float(-np.log(aod469 / aod865) / np.log(469.0 / 865.0))
    else:
        ae = np.nan

    return {
        "aod_550": raw.get("aod550", np.nan),
        "aod_469": raw.get("aod469", np.nan),
        "aod_865": raw.get("aod865", np.nan),
        "ae_469_865": ae,
        "dust_aod": raw.get("duaod550", np.nan),
        "sulphate_aod": raw.get("suaod550", np.nan),
        "bc_aod": raw.get("bcaod550", np.nan),
        "om_aod": raw.get("omaod550", np.nan),
        "ss_aod": raw.get("ssaod550", np.nan),
    }


def load_predictor():
    """加载模型和定标器。"""
    with open(SCALER_PATH, "rb") as f:
        scaler_info = pickle.load(f)
    feature_cols = scaler_info["cols"]
    scaler = scaler_info["scaler"]
    y_mean = scaler_info["y_mean"]
    y_std = scaler_info["y_std"]

    model = ResMLPRegressor(len(feature_cols), hidden_dim=128, n_blocks=4, dropout=0.15)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return model, scaler, feature_cols, y_mean, y_std


def predict(lat: float, lon: float, target_date: str,
            model=None, scaler=None, feature_cols=None,
            y_mean=None, y_std=None) -> dict:
    """完整推理管线。"""
    if model is None:
        model, scaler, feature_cols, y_mean, y_std = load_predictor()

    weather = fetch_weather(lat, lon, target_date)
    aod = fetch_aod(lat, lon, target_date)
    day_of_year = pd.to_datetime(target_date).dayofyear

    feat_vec = np.zeros(len(feature_cols), dtype=np.float32)
    missing = []
    for i, c in enumerate(feature_cols):
        if c in aod and not np.isnan(aod[c]):
            feat_vec[i] = aod[c]
        elif c in weather and not np.isnan(weather[c]):
            feat_vec[i] = weather[c]
        elif c == "day_of_year":
            feat_vec[i] = day_of_year
        else:
            feat_vec[i] = scaler.mean_[i]
            missing.append(c)

    feat_scaled = scaler.transform(feat_vec.reshape(1, -1))
    with torch.no_grad():
        pm25_z = model(torch.tensor(feat_scaled, dtype=torch.float32)).item()
    # 逆标准化: z-score → 原始 PM2.5 尺度
    pm25 = pm25_z * y_std + y_mean
    pm25 = max(0.0, pm25)

    # 空气质量等级
    for thresh, level, color in AQI_LEVELS:
        if pm25 <= thresh:
            break

    return {
        "pm2_5": round(pm25, 1),
        "level": level,
        "color": color,
        "lat": lat, "lon": lon, "date": target_date,
        "weather": weather,
        "aod": aod,
        "missing_features": missing,
    }


def main():
    parser = argparse.ArgumentParser(description="UrbanAir PM2.5 推理")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    print(f"加载模型...")
    model, scaler, cols, y_mean, y_std = load_predictor()
    print(f"特征数: {len(cols)}")

    print(f"\n预测: ({args.lat:.4f}, {args.lon:.4f})  {args.date}")
    result = predict(args.lat, args.lon, args.date, model, scaler, cols, y_mean, y_std)

    print(f"\n  PM2.5 = {result['pm2_5']} μg/m³  ({result['level']})")
    print(f"\n气象:")
    for k, v in result["weather"].items():
        print(f"  {k}: {v:.2f}" if not np.isnan(v) else f"  {k}: 缺失")
    print(f"\nAOD:")
    for k, v in result["aod"].items():
        print(f"  {k}: {v:.4f}" if not np.isnan(v) else f"  {k}: 缺失")
    if result["missing_features"]:
        print(f"\n缺失特征 (中位数填充): {result['missing_features']}")


if __name__ == "__main__":
    main()
