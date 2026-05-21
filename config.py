"""
MVP 共享配置 —— 城市坐标、路径、特征列表
所有脚本从此导入，避免重复定义
"""
from pathlib import Path

# --- 路径 ---
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_MODELS = ROOT / "data" / "models"

for d in [DATA_RAW, DATA_PROCESSED, DATA_MODELS]:
    d.mkdir(parents=True, exist_ok=True)

# --- 目标城市及其近似中心坐标 ---
CITY_COORDS = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "成都": (30.5728, 104.0668),
    "武汉": (30.5928, 114.3055),
    "西安": (34.3416, 108.9398),
    "南京": (32.0603, 118.7969),
    "杭州": (30.2741, 120.1551),
    "郑州": (34.7466, 113.6254),
    "长沙": (28.2282, 112.9388),
}

# --- 空气质量变量 (Open-Meteo Air Quality API) ---
AQI_VARIABLES = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "european_aqi",
]
# PM2.5 列名 (固定已知)
PM25_COL = "pm2_5"

# --- 气象变量 (Open-Meteo Weather API) ---
WEATHER_VARIABLES = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
    "surface_pressure_mean",
    "precipitation_sum",
    "cloud_cover_mean",
]

# --- OSM POI 标签 ---
OSM_POI_TAGS = {
    "industrial": {"industrial": True},
    "fuel": {"amenity": "fuel"},
    "restaurant": {"amenity": "restaurant"},
    "school": {"amenity": "school"},
    "hospital": {"amenity": "hospital"},
    "park": {"leisure": "park"},
}

# --- OSM 缓冲区半径 (m) ---
OSM_RADIUS = 1000

# --- 训练/测试时间切分点 ---
TRAIN_CUTOFF = "2024-11-01"

# --- 排除列 (不作为特征) ---
EXCLUDE_COLS = [
    "city", "date", "lat", "lon", "station_name",
    "pm10", "nitrogen_dioxide", "ozone", "european_aqi",
]
