"""
train.py —— 最终版统一训练入口
用法: python train.py --config config.yaml --model attn_fusion

MVP 升级路径:
  05_train_model.py (MVP, 仅XGBoost)
    → train.py (完整版, 支持config切换7种模型)
"""
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--model", default=None, help="模型名称 (覆盖config)")
    parser.add_argument("--data", default=None, help="数据路径 (覆盖config)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: str) -> nn.Module:
    """根据config构建模型"""
    from models import get_model

    model_name = cfg["model"]["name"]
    model_cfg = cfg["model"]

    kwargs = {}
    if model_name in ["early_fusion", "attn_fusion"]:
        kwargs.update(model_cfg.get("fusion", {}))
    if model_name == "resnet":
        kwargs.update(model_cfg.get("resnet", {}))

    model = get_model(model_name, **kwargs)
    return model.to(device)


def create_dataloaders(data_path: str, cfg: dict):
    """加载数据并创建DataLoader —— 骨架, 待数据管线对接"""
    # TODO: 从 HDF5/CSV 加载对齐后的三模态数据
    # TODO: 按 config["split"] 划分训练/验证/测试
    # TODO: 卫星图模态: 在线/离线 ResNet 特征提取
    # 此处返回占位
    return None, None, None


def train_epoch(model, loader, optimizer, criterion, device):
    """单epoch训练 —— 骨架"""
    model.train()
    total_loss = 0.0
    for batch in loader:
        pass  # TODO: 实现batch训练逻辑
    return total_loss / max(len(loader), 1)


def validate(model, loader, criterion, device):
    """验证 —— 骨架"""
    model.eval()
    metrics = {}
    with torch.no_grad():
        for batch in loader:
            pass  # TODO: 实现验证逻辑
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.model:
        cfg["model"]["name"] = args.model

    device = torch.device(args.device)
    print(f"设备: {device}")
    print(f"模型: {cfg['model']['name']}")
    print(f"数据: {cfg['data']}")

    # 构建模型
    model = build_model(cfg, device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 加载数据
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data or cfg["data"], cfg
    )

    # TODO: 完整训练循环 (集成早停/checkpoint/TensorBoard)
    # TODO: 超参数搜索 (Optuna / Grid Search)
    # TODO: 消融实验 (按config切换模态组合)

    print("\n✓ 训练框架初始化完成 (模型已构建, 待数据管线对接)")


if __name__ == "__main__":
    main()
