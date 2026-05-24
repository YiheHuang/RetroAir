"""
UrbanAir PM2.5 推理应用

用法:
  streamlit run app.py
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent))

from scripts.inference import load_predictor, predict

CITY_PRESETS = {
    "北京": (39.9042, 116.4074), "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644), "成都": (30.5728, 104.0668),
    "武汉": (30.5928, 114.3055), "西安": (34.3416, 108.9398),
    "南京": (32.0603, 118.7969), "长沙": (28.2282, 112.9388),
    "东京": (35.6762, 139.6503), "首尔": (37.5665, 126.9780),
    "新德里": (28.6139, 77.2090), "伦敦": (51.5074, -0.1278),
    "纽约": (40.7128, -74.0060), "洛杉矶": (34.0522, -118.2437),
}

DEFAULT_LAT, DEFAULT_LON = 35.0, 105.0

st.set_page_config(page_title="UrbanAir PM2.5", page_icon="🌫️", layout="wide")

model, scaler, feature_cols, y_mean, y_std = load_predictor()

# ---- session state ----
for key, default in [
    ("lat", DEFAULT_LAT), ("lon", DEFAULT_LON), ("zoom", 4),
    ("history", {}), ("last_result", None),
    ("_do_predict", False), ("_last_consumed_click", (None, None)),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🌫️ UrbanAir — PM2.5 实时推理")
st.caption(f"ResMLP · {len(feature_cols)} 特征")

col_map, col_ctrl = st.columns([2, 1])

# ========== 地图 ==========
with col_map:
    m = folium.Map(
        location=[st.session_state.lat, st.session_state.lon],
        zoom_start=st.session_state.zoom,
        tiles="OpenStreetMap",
    )
    folium.LatLngPopup().add_to(m)
    folium.ClickForMarker().add_to(m)

    # 当前选中位置
    folium.Marker(
        [st.session_state.lat, st.session_state.lon],
        icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
        tooltip="当前选中",
    ).add_to(m)

    target_date_str = st.session_state.get("_date", date.today().isoformat())
    for h in st.session_state.history.get(target_date_str, []):
        folium.CircleMarker(
            [h["lat"], h["lon"]], radius=5, color=h["color"],
            fill=True, fill_opacity=0.8,
            tooltip=f"PM2.5={h['pm2_5']:.0f}",
        ).add_to(m)

    map_data = st_folium(m, height=520, width="100%", key="the_map")

    # 地图点击 → 仅在坐标真的变化时更新并 rerun
    if map_data and map_data.get("last_clicked"):
        c = map_data["last_clicked"]
        if c.get("lat") and c.get("lng"):
            nl = round(c["lat"], 4)
            nn = round(c["lng"], 4)
            # 归一化经度到 [-180, 180]
            nn = ((nn + 180) % 360) - 180
            # 纬度截断
            nl = max(-90, min(90, nl))
            # 和上次已消费的点击比较（而非和当前坐标比较），防止重复 rerun
            prev = st.session_state.get("_last_consumed_click", (None, None))
            if (nl, nn) != prev:
                st.session_state.lat = nl
                st.session_state.lon = nn
                st.session_state.zoom = 10
                st.session_state._last_consumed_click = (nl, nn)
                st.rerun()

# ========== 控制面板 ==========
with col_ctrl:
    st.subheader("📍 位置")

    c = st.selectbox("快捷城市", [""] + list(CITY_PRESETS.keys()),
                     placeholder="选择城市...")
    if c:
        st.session_state.lat, st.session_state.lon = CITY_PRESETS[c]

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.lat = st.number_input(
            "纬度", -90.0, 90.0, st.session_state.lat, step=0.01, format="%.4f")
    with c2:
        st.session_state.lon = st.number_input(
            "经度", -180.0, 180.0, st.session_state.lon, step=0.01, format="%.4f")

    st.info(f"( {st.session_state.lat:.4f} , {st.session_state.lon:.4f} )")

    target_date = st.date_input("日期", date.today() - timedelta(days=7),
                                max_value=date.today())
    target_date_str = target_date.isoformat()
    st.session_state._date = target_date_str

    # 按钮设置 session_state 标记 (跨 st.rerun() 保持)
    if st.button("🔍 预测 PM2.5", type="primary", use_container_width=True):
        st.session_state._do_predict = True

    st.divider()

    if st.session_state.last_result is not None:
        r = st.session_state.last_result
        st.markdown(f"""
        <div style="background:#f0f2f6;border-radius:10px;padding:12px;margin:8px 0">
            <h3 style="margin:0;color:{r['color']}">PM₂.₅ = {r['pm2_5']:.1f} μg/m³</h3>
            <small>({r['level']}) · {r['date']} · ({r['lat']:.4f}, {r['lon']:.4f})</small>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📊 特征详情"):
            ca, cb = st.columns(2)
            with ca:
                st.write("**气象**")
                for k, v in r.get("weather", {}).items():
                    st.text(f"{k}: {v:.2f}" if not np.isnan(v) else f"{k}: 缺失")
            with cb:
                st.write("**AOD**")
                for k, v in r.get("aod", {}).items():
                    st.text(f"{k}: {v:.4f}" if not np.isnan(v) else f"{k}: 缺失")
            if r.get("missing_features"):
                st.warning(f"缺失特征: {r['missing_features']}")

    recs = st.session_state.history.get(target_date_str, [])
    if recs:
        st.divider()
        st.caption(f"📋 {target_date_str} · {len(recs)} 次")
        for h in recs[-5:]:
            st.caption(f"📍 {h['pm2_5']} μg/m³ · {h['level']}")

# ========== 执行预测 ==========
if st.session_state.get("_do_predict"):
    st.session_state._do_predict = False
    lat, lon = st.session_state.lat, st.session_state.lon
    with st.spinner(f"预测中 ({lat:.4f}, {lon:.4f})..."):
        try:
            r = predict(lat, lon, target_date_str,
                        model, scaler, feature_cols, y_mean, y_std)
            r["lat"] = lat
            r["lon"] = lon
            st.session_state.last_result = r
            st.session_state.history.setdefault(target_date_str, []).append({
                "lat": lat, "lon": lon,
                "pm2_5": r["pm2_5"], "level": r["level"], "color": r["color"],
            })
            st.rerun()
        except Exception as e:
            st.error(f"预测失败: {e}")
