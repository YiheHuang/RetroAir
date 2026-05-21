"""
resnet.py —— ResNet-50 卫星图像编码器 (M4)
  - EuroSAT 预训练权重加载
  - 支持冻结/微调指定层
  - 输出 2048-d 特征向量
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetEncoder(nn.Module):
    """M4: 卫星图像特征提取器

    输入: (B, 3, 224, 224) Sentinel-2 RGB 影像
    输出: (B, 2048) 特征向量
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: str = "eurosat",   # eurosat | imagenet
        freeze_layers: list = None,     # 冻结的层索引 [1,2,3]
        feature_dim: int = 2048,
    ):
        super().__init__()

        # 加载 backbone
        if pretrained == "imagenet":
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            # EuroSAT 权重: 从 HuggingFace 下载后加载
            self.backbone = resnet50(weights=None)
            # TODO: 加载 EuroSAT 预训练权重
            # state_dict = torch.load("eurosat_resnet50.pth")
            # self.backbone.load_state_dict(state_dict)

        # 移除最后的分类头
        self.encoder = nn.Sequential(*list(self.backbone.children())[:-1])
        self.feature_dim = feature_dim

        # 冻结指定层
        if freeze_layers:
            self._freeze_layers(freeze_layers)

    def _freeze_layers(self, layers: list):
        """冻结指定层索引 (1-indexed, 对应ResNet的4个layer组)"""
        layer_map = {1: [4], 2: [5], 3: [6], 4: [7]}
        for layer_idx in layers:
            for child_idx in layer_map.get(layer_idx, []):
                for param in list(self.backbone.children())[child_idx].parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) 卫星图像
        Returns:
            (B, feature_dim) 特征向量
        """
        feat = self.encoder(x)          # (B, 2048, 1, 1)
        feat = torch.flatten(feat, 1)   # (B, 2048)
        return feat


# 模块自测
if __name__ == "__main__":
    encoder = ResNetEncoder(pretrained="imagenet", freeze_layers=[1, 2, 3])
    dummy = torch.randn(4, 3, 224, 224)
    out = encoder(dummy)
    print(f"输入: {dummy.shape} → 输出: {out.shape}")
    print(f"可训练参数: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")
    print(f"总参数: {sum(p.numel() for p in encoder.parameters()):,}")
