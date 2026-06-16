"""
RetroAir 共享配置 —— 城市坐标、路径、特征列表
基于 气象 + AOD 推断 PM2.5（空间插值任务）
所有脚本从此导入，避免重复定义
"""
from pathlib import Path

from dotenv import load_dotenv

# --- 路径 ---
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_MODELS = ROOT / "data" / "models"

# 子目录
DATA_AQI = DATA_RAW / "aqi"
DATA_WEATHER = DATA_RAW / "weather"
DATA_CAMS_NC = DATA_RAW / "cams_nc"
DATA_AOD = DATA_PROCESSED / "aod"
DATA_MERGED = DATA_PROCESSED / "merged"
DATA_FINAL = DATA_PROCESSED / "final"

for d in [DATA_RAW, DATA_PROCESSED, DATA_MODELS,
          DATA_AQI, DATA_WEATHER, DATA_CAMS_NC,
          DATA_AOD, DATA_MERGED, DATA_FINAL]:
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

# 气象标准字段 (训练统一使用)
WEATHER_CANONICAL_COLS = WEATHER_VARIABLES

# NASA POWER 对应字段映射 (作为 Open-Meteo 的兜底数据源)
WEATHER_NASA_POWER_MAP = {
    "temperature_2m_mean": "T2M",
    "relative_humidity_2m_mean": "RH2M",
    "wind_speed_10m_mean": "WS10M",
    "wind_direction_10m_dominant": "WD10M",
    "surface_pressure_mean": "PS",
    "precipitation_sum": "PRECTOTCORR",
    "cloud_cover_mean": "CLOUD_AMT",
}

# --- CAMS EAC4 气溶胶再分析数据 (Copernicus ADS, 免费) ---
# 数据集: cams-global-reanalysis-eac4
# 注册: https://ads.atmosphere.copernicus.eu
CAMS_DATASET = "cams-global-reanalysis-eac4"
CAMS_AREA = [55, 73, 18, 135]  # [北, 西, 南, 东] 覆盖全中国
# CDS API 请求用的变量名（长名）
CAMS_CDS_VARIABLES = [
    "total_aerosol_optical_depth_550nm",
    "total_aerosol_optical_depth_469nm",
    "total_aerosol_optical_depth_670nm",
    "total_aerosol_optical_depth_865nm",
    "total_aerosol_optical_depth_1240nm",
    "dust_aerosol_optical_depth_550nm",
    "sulphate_aerosol_optical_depth_550nm",
    "organic_matter_aerosol_optical_depth_550nm",
    "black_carbon_aerosol_optical_depth_550nm",
    "sea_salt_aerosol_optical_depth_550nm",
]
# NetCDF 中的变量名（简写）→ 对应 CAMS_CDS_VARIABLES 的顺序
CAMS_NC_VARIABLES = [
    "aod550", "aod469", "aod670", "aod865", "aod1240",
    "duaod550", "suaod550", "omaod550", "bcaod550", "ssaod550",
]
# CAMS 日数据取 00/06/12/18 UTC 四个时次 (EAC4 为 3 小时间隔)
CAMS_TIMES = ["00:00", "06:00", "12:00", "18:00"]

SATELLITE_FEATURES = [
    "aod_550",
    "aod_469",
    "aod_865",
    "ae_469_865",
    "dust_aod",
    "sulphate_aod",
    "bc_aod",
    "om_aod",
    "ss_aod",
]

# --- 时间特征 ---
TIME_FEATURES = ["day_of_year"]

# --- 训练/测试切分 (随机切分, 空间插值任务) ---
TEST_SIZE = 0.2
SPLIT_SEED = 42

# --- 排除列 (不作为特征) ---
EXCLUDE_COLS = [
    "city", "date", "lat", "lon", "station_name",
    "pm10", "nitrogen_dioxide", "ozone", "european_aqi",
]
