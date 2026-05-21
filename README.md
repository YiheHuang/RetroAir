# UrbanAir —— 城市环境质量多模态评估与预测系统

数据分析与挖掘 · 期末项目

## 输入输出

**输入任意经纬度 + 日期 → 实时拉取卫星/城市场景/气象三源数据 → 模型融合推断 → 输出该地点 PM2.5 预测值。** 解决监测站覆盖盲区问题：X位置空气质量到底怎么样？

---

## 问题与动机

中国168城仅有少量AQI监测站（每城3-15个），**站点之外没有空气质量数据**。但PM2.5不是随机分布的——它和周边土地利用（工厂/公园）、路网密度、气象条件高度相关。

如果我们能用**公开免费的多源数据**训练出一个能够推断任意地点的空气质量的AI模型，就能构建一张高分辨率污染地图。

---

## 三模态输入 → 模型融合 → 推断输出

```
用户输入: (纬度, 经度, 日期)
    │
    ├─ 模态1: 卫星遥感 ── Sentinel-2 10m影像 ──→ ResNet-50 ──→ [2048维]
    ├─ 模态2: 城市场景 ── OSM POI/路网/建筑 ──→ 统计特征 ──→ [~30维]
    └─ 模态3: 气象条件 ── 温/湿/风/压/降水  ──→ Open-Meteo ─→ [7维]
    │
    └──→ 多模态交叉注意力融合 (MM-CAF) ──→ PM2.5 预测值 + AQI 等级
```

**全部数据源公开免费，Python API 一键可达，单GPU可训练。**

| 模态 | 数据源 | 获取方式 | 说明 |
|------|--------|---------|------|
| 卫星遥感 | Sentinel-2 L2A | Copernicus API / GEE | 10m分辨率, 1km×1km切片 |
| 城市场景 | OpenStreetMap | `osmnx` Python库 | POI密度/路网/绿地/建筑 |
| 气象条件 | ERA5 再分析 | `openmeteo` Python库 | 日均温湿风压降水, 无需Key |
| 标签(PM2.5) | CAMS 再分析 | Open-Meteo AQI API | ~40km, 无需Key |

---

## 算法路线

### MVP（当前已实现）
- **XGBoost** 在 OSM + 气象特征上预测 PM2.5
- CPU训练30秒, Streamlit单页预测器
- 支持**任意经纬度输入**, 实时抽取OSM+气象特征后推断

### 完整版
- 加入 Sentinel-2 卫星影像 → ResNet-50 编码
- 多模态交叉注意力融合网络（MM-CAF）：卫星↔OSM↔气象三向交互
- 7种算法系统对比（RF / XGBoost / MLP / ResNet / EarlyFusion / CrossAttnFusion / LateFusion）
- Streamlit 4模块 Dashboard: AQI地图 / 模型对比 / 模态分析 / 预测器

---

## 项目结构

```
mvp/                          # 当前 MVP（已可运行）
├── run_pipeline.py           # 一键: 数据下载→训练→预测App
├── config.py                 # 城市坐标、特征定义、路径
├── scripts/
│   ├── 01_fetch_aqi.py       # 下载 PM2.5 数据
│   ├── 02_fetch_weather.py   # 下载气象数据
│   ├── 03_extract_osm.py     # 提取 OSM 缓冲特征
│   ├── 04_merge_data.py      # 三源时空对齐
│   ├── 05_train_model.py     # XGBoost 训练+评估
│   └── 06_app.py             # Streamlit 预测器（任意经纬度）
├── data/{raw,processed,models}/
└── requirements.txt

final_skeleton/               # 完整版框架（待Week 3-5填充）
├── config.yaml               # 全局配置（7模型切换）
├── train.py                  # 统一训练入口
├── app.py                    # Streamlit 4模块应用
├── models/
│   ├── baseline.py           # M1 RF + M2 XGBoost
│   ├── resnet.py             # M4 ResNet-50 编码器
│   └── fusion.py             # M5/M6/M7 多模态融合
└── utils/                    # 评估指标 + 可视化
```

---

## 5分钟快速启动

```bash
pip install -r requirements.txt
python run_pipeline.py --test-mode   # 30天/3城, 验证全管线
streamlit run scripts/06_app.py      # 打开预测器
```

---

## 团队分工

| 角色 | 核心职责 | 技术栈 |
|------|---------|--------|
| **数据工程** | 四源数据获取/清洗/对齐、特征工程Pipeline、数据版本管理 | `osmnx`, `openmeteo`, `sentinelsat`, `pandas` |
| **算法** | MM-CAF网络实现、7模型对比实验、超参数调优 | `PyTorch`, `timm`, `einops`, `XGBoost` |
| **评估分析** | 评估指标体系、统计检验、SHAP/注意力可视化、消融实验 | `sklearn`, `shap`, `plotly` |
| **工程交付** | Streamlit 4模块应用、一键运行脚本、Git管理、答辩PPT+视频 | `Streamlit`, `Folium`, `Plotly`, `Git` |

## 与文档案例的区分

不选房价预测/信用风险/犯罪预测/时间序列/推荐系统/股票分析——这些是历年高频选题。本项目融合环境科学+遥感+城市计算，三模态异构融合是刚需而非勉强拼凑，全部数据公开可复现。