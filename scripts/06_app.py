"""
06_app.py —— Streamlit MVP: 任意地点空气质量预测器
运行: streamlit run scripts/06_app.py

输入: 纬度 + 经度 + 日期
输出: PM2.5 预测值 + AQI 等级
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import pandas as pd
import numpy as np
import streamlit as st
import osmnx as ox
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, date
from config import (
    DATA_MODELS, CITY_COORDS, OSM_POI_TAGS, OSM_RADIUS,
    WEATHER_VARIABLES, PM25_COL,
)

st.set_page_config(page_title="UrbanAir MVP", page_icon="🌍", layout="centered")


# ============================================================
# 核心函数: 从经纬度提取特征 (无缓存 — 任意位置都不同)
# ============================================================
def extract_osm_at_point(lat: float, lon: float) -> dict:
    """在任意经纬度提取OSM缓冲特征"""
    features = {}
    for name, tags in OSM_POI_TAGS.items():
        try:
            gdf = ox.features_from_point((lat, lon), tags=tags, dist=OSM_RADIUS)
            features[f"poi_{name}_count"] = len(gdf)
        except Exception:
            features[f"poi_{name}_count"] = 0

    try:
        G = ox.graph_from_point((lat, lon), dist=OSM_RADIUS, network_type="drive")
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        features["road_total_length_m"] = (
            float(edges["length"].sum()) if "length" in edges.columns else 0.0
        )
        features["road_segment_count"] = len(edges)

        if "highway" in edges.columns:
            for hw in ["motorway", "trunk", "primary", "secondary",
                        "tertiary", "residential"]:
                mask = edges["highway"].str.contains(hw, na=False)
                features[f"road_{hw}_length_m"] = (
                    float(edges.loc[mask, "length"].sum())
                    if "length" in edges.columns else 0.0
                )
        else:
            for hw in ["motorway", "trunk", "primary", "secondary",
                        "tertiary", "residential"]:
                features[f"road_{hw}_length_m"] = 0.0
    except Exception:
        features["road_total_length_m"] = 0.0
        features["road_segment_count"] = 0
        for hw in ["motorway", "trunk", "primary", "secondary",
                     "tertiary", "residential"]:
            features[f"road_{hw}_length_m"] = 0.0

    area_km2 = (3.14159 * OSM_RADIUS ** 2) / 1e6
    features["road_density_km_per_km2"] = (
        features["road_total_length_m"] / 1000 / area_km2 if area_km2 > 0 else 0.0
    )
    total_poi = sum(features.get(f"poi_{n}_count", 0) for n in OSM_POI_TAGS)
    features["poi_total_density_per_km2"] = total_poi / area_km2 if area_km2 > 0 else 0.0

    return features


@st.cache_data(ttl=3600)
def fetch_weather_at_point(lat: float, lon: float, date_str: str) -> dict:
    """在任意经纬度获取气象数据"""
    cache = requests_cache.CachedSession(
        str(DATA_MODELS.parent.parent / ".cache" / "weather"), expire_after=3600
    )
    client = openmeteo_requests.Client(session=retry(cache, retries=2, backoff_factor=0.5))
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": date_str, "end_date": date_str,
        "daily": WEATHER_VARIABLES,
    }
    resp = client.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )
    daily = resp[0].Daily()
    weather = {}
    for i, name in enumerate(WEATHER_VARIABLES):
        vals = daily.Variables(i).ValuesAsNumpy()
        weather[name] = float(vals[0]) if len(vals) > 0 else 0.0
    return weather


# ============================================================
# 加载模型
# ============================================================
@st.cache_resource
def load_model():
    path = DATA_MODELS / "xgb_model.pkl"
    if not path.exists():
        st.error(f"模型文件不存在: {path}\n请先运行 run_pipeline.py")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


artifact = load_model()
model = artifact["model"]
scaler = artifact["scaler"]
feature_cols = artifact["feature_cols"]
metrics = artifact["metrics"]


# ============================================================
# UI
# ============================================================
st.title("🌍 UrbanAir MVP")
st.caption("任意地点空气质量预测 —— 基于 OSM + 气象 + 多源数据的 PM2.5 推断")
st.markdown("---")

# 快速选择城市 (填充经纬度)
quick_city = st.selectbox(
    "📍 快捷选点 (或手动输入下方经纬度)",
    ["-- 手动输入 --"] + list(CITY_COORDS.keys()),
)

col1, col2 = st.columns(2)
with col1:
    if quick_city != "-- 手动输入 --":
        default_lat = CITY_COORDS[quick_city][0]
    else:
        default_lat = 39.90
    lat = st.number_input("纬度 (Latitude)", min_value=18.0, max_value=55.0,
                          value=default_lat, format="%.4f",
                          help="中国大陆范围: 18~55°N")

with col2:
    if quick_city != "-- 手动输入 --":
        default_lon = CITY_COORDS[quick_city][1]
    else:
        default_lon = 116.40
    lon = st.number_input("经度 (Longitude)", min_value=73.0, max_value=135.0,
                          value=default_lon, format="%.4f",
                          help="中国大陆范围: 73~135°E")

selected_date = st.date_input(
    "📅 选择日期",
    date(2024, 6, 15),
    min_value=date(2024, 1, 1),
    max_value=date(2024, 12, 31),
)

# 预测按钮
predict_btn = st.button("🔍 预测 PM2.5", type="primary", use_container_width=True)

if predict_btn:
    date_str = selected_date.strftime("%Y-%m-%d")

    with st.spinner(f"正在提取 ({lat:.4f}, {lon:.4f}) 的多源特征..."):
        # 1. OSM 特征
        osm_feat = extract_osm_at_point(lat, lon)

        # 2. 气象特征
        weather_feat = fetch_weather_at_point(lat, lon, date_str)

        # 3. 合并 + 对齐特征列
        all_feat = {**osm_feat, **weather_feat}
        input_df = pd.DataFrame([all_feat])

        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[feature_cols].fillna(0.0)

        # 4. 预测
        X = scaler.transform(input_df)
        pm25_pred = float(model.predict(X)[0])

    # ============================================================
    # 显示结果
    # ============================================================
    st.markdown("---")
    st.subheader(f"📊 预测结果 —— ({lat:.4f}, {lon:.4f})")

    aqi_levels = [
        (0, 35, "🟢 优", "#4CAF50"),
        (35, 75, "🟡 良", "#FFC107"),
        (75, 115, "🟠 轻度污染", "#FF9800"),
        (115, 150, "🔴 中度污染", "#F44336"),
        (150, 250, "🟣 重度污染", "#9C27B0"),
        (250, 999, "🟤 严重污染", "#795548"),
    ]
    level_label, level_color = next(
        (l, c) for lo, hi, l, c in aqi_levels if pm25_pred <= hi
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("PM2.5 预测值", f"{pm25_pred:.1f} μg/m³")
    with c2:
        st.metric("AQI 等级", level_label)
    with c3:
        pm25_bp = [0, 35, 75, 115, 150, 250, 500]
        aqi_bp = [0, 50, 100, 150, 200, 300, 500]
        est_aqi = np.interp(pm25_pred, pm25_bp, aqi_bp)
        st.metric("估算 AQI", f"{est_aqi:.0f}")

    st.markdown(
        f"<div style='height:8px;background:{level_color};border-radius:4px'></div>",
        unsafe_allow_html=True,
    )

    # 模型信息
    st.markdown("---")
    st.subheader("📈 模型信息")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("算法", "XGBoost")
    with mc2:
        st.metric("训练集 R²", f"{metrics['train_r2']:.3f}")
    with mc3:
        st.metric("测试集 R²", f"{metrics['test_r2']:.3f}")
    with mc4:
        st.metric("测试 RMSE", f"{metrics['test_rmse']:.1f}")

    # 特征重要性
    st.markdown("---")
    st.subheader("🔝 特征重要性 Top-8")
    importance = pd.DataFrame(artifact["importance"]).head(8)
    for _, row in importance.iterrows():
        pct = row["importance"] / importance["importance"].max()
        st.text(f"{row['feature']:<35s} {'█' * int(pct * 40)} {row['importance']:.4f}")

    # 输入特征快照
    with st.expander("🔍 查看输入特征详情"):
        st.json({k: round(v, 4) if isinstance(v, float) else v
                  for k, v in all_feat.items()})


if __name__ == "__main__":
    print("请使用: streamlit run scripts/06_app.py")
