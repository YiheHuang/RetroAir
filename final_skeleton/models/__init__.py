"""
最终版项目框架 —— 模块初始化与模型注册

最终版目录结构:
final/
├── config.yaml          # 全局配置 (Hydra/OmegaConf)
├── data_pipeline.py     # 统一数据管线类
├── train.py             # 统一训练入口
├── evaluate.py          # 评估+可视化
├── app.py               # Streamlit 完整应用
├── models/
│   ├── __init__.py      # 模型注册表
│   ├── baseline.py      # RF + XGBoost
│   ├── resnet.py        # ResNet-50 卫星图编码器
│   └── fusion.py        # M5/M6/M7 多模态融合
├── utils/
│   ├── __init__.py
│   ├── metrics.py       # 评估指标
│   └── visualize.py     # 可视化工具
└── run_all.sh           # 一键运行
"""
from pathlib import Path

# 注册所有可用模型
MODEL_REGISTRY = {
    "rf": "models.baseline.RandomForestModel",
    "xgboost": "models.baseline.XGBoostModel",
    "resnet": "models.resnet.ResNetEncoder",
    "early_fusion": "models.fusion.EarlyFusionNet",
    "attn_fusion": "models.fusion.CrossAttnFusionNet",
    "late_fusion": "models.fusion.LateFusionEnsemble",
}

def get_model(name: str, **kwargs):
    """从注册表获取模型实例"""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 '{name}'. 可用: {list(MODEL_REGISTRY.keys())}")
    module_path, class_name = MODEL_REGISTRY[name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(**kwargs)
