"""
fusion.py —— 多模态融合模型 (M5/M6/M7)

  M5 EarlyFusionNet: 特征级融合 (拼接 → MLP)
  M6 CrossAttnFusionNet: 交叉注意力融合
  M7 LateFusionEnsemble: 决策级融合 (Stacking)

所有融合模型接收三个模态的嵌入向量:
  - img_feat:  (B, 2048) ResNet编码的卫星图特征
  - osm_feat:  (B, 30)   OSM缓冲区统计特征
  - met_feat:  (B, 7)    气象条件特征
输出:
  - pm25:      (B, 1)    PM2.5回归预测
  - aqi_cls:   (B, 6)    AQI等级分类logits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# M5: 早融合 (Early Fusion)
# ============================================================
class EarlyFusionNet(nn.Module):
    """特征级融合: 各模态编码器输出拼接后送入MLP"""

    def __init__(
        self,
        img_dim: int = 2048,
        osm_dim: int = 30,
        met_dim: int = 7,
        hidden_dim: int = 256,
        mlp_hidden: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [768, 512, 256, 128]

        # 模态投影
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.osm_proj = nn.Sequential(
            nn.Linear(osm_dim, hidden_dim // 4),   # 30→64
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
        )
        self.met_proj = nn.Sequential(
            nn.Linear(met_dim, hidden_dim // 8),   # 7→32
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 8),
        )

        total_dim = hidden_dim + hidden_dim // 4 + hidden_dim // 8

        # MLP 预测头
        layers = []
        prev_dim = total_dim
        for h in mlp_hidden:
            layers += [
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        self.mlp = nn.Sequential(*layers)

        # 双输出头
        self.reg_head = nn.Linear(mlp_hidden[-1], 1)     # PM2.5
        self.cls_head = nn.Linear(mlp_hidden[-1], 6)     # AQI等级

    def forward(self, img_feat, osm_feat, met_feat):
        img = self.img_proj(img_feat)
        osm = self.osm_proj(osm_feat)
        met = self.met_proj(met_feat)

        fused = torch.cat([img, osm, met], dim=1)
        hidden = self.mlp(fused)

        return self.reg_head(hidden).squeeze(-1), self.cls_head(hidden)


# ============================================================
# M6: 交叉注意力融合 (Cross-Attention Fusion)
# ============================================================
class CrossAttentionBlock(nn.Module):
    """单路交叉注意力: Q从A来, K/V从B来"""

    def __init__(self, dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        # query: (B, 1, dim)   key_value: (B, 1, dim)
        attn_out, _ = self.attn(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_out))


class CrossAttnFusionNet(nn.Module):
    """交叉注意力融合: 模态间显式交互"""

    def __init__(
        self,
        img_dim: int = 2048,
        osm_dim: int = 30,
        met_dim: int = 7,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_hidden: list = None,
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [768, 512, 256, 128]

        # 模态投影 → 统一维度
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.ReLU(), nn.LayerNorm(hidden_dim),
        )
        self.osm_proj = nn.Sequential(
            nn.Linear(osm_dim, hidden_dim),
            nn.ReLU(), nn.LayerNorm(hidden_dim),
        )
        self.met_proj = nn.Sequential(
            nn.Linear(met_dim, hidden_dim),
            nn.ReLU(), nn.LayerNorm(hidden_dim),
        )

        # 交叉注意力块 (3对有向交互)
        self.cross_img_osm = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.cross_osm_met = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.cross_met_img = CrossAttentionBlock(hidden_dim, num_heads, dropout)

        # MLP头
        layers = []
        prev_dim = hidden_dim * 3
        for h in mlp_hidden:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(),
                       nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev_dim = h
        self.mlp = nn.Sequential(*layers)

        self.reg_head = nn.Linear(mlp_hidden[-1], 1)
        self.cls_head = nn.Linear(mlp_hidden[-1], 6)

    def forward(self, img_feat, osm_feat, met_feat):
        # 投影
        img = self.img_proj(img_feat).unsqueeze(1)   # (B, 1, H)
        osm = self.osm_proj(osm_feat).unsqueeze(1)
        met = self.met_proj(met_feat).unsqueeze(1)

        # 交叉注意力
        enhanced_img = self.cross_img_osm(img, osm)   # 卫星→OSM
        enhanced_osm = self.cross_osm_met(osm, met)   # OSM→气象
        enhanced_met = self.cross_met_img(met, img)   # 气象→卫星

        # 融合
        fused = torch.cat([
            enhanced_img.squeeze(1),
            enhanced_osm.squeeze(1),
            enhanced_met.squeeze(1),
        ], dim=1)

        hidden = self.mlp(fused)
        return self.reg_head(hidden).squeeze(-1), self.cls_head(hidden)


# ============================================================
# M7: 晚融合 (Late Fusion / Stacking)
# ============================================================
class LateFusionEnsemble:
    """决策级融合: 每模态独立模型, XGBoost做元学习器"""

    def __init__(self, meta_model=None):
        from xgboost import XGBRegressor
        # 基础学习器: 每模态一个
        self.img_model = None   # Ridge (在fit中创建)
        self.osm_model = XGBRegressor(n_estimators=100, max_depth=4)
        self.met_model = XGBRegressor(n_estimators=100, max_depth=3)
        self.meta_model = meta_model or XGBRegressor(n_estimators=50, max_depth=3)

    def fit(self, img_feat, osm_feat, met_feat, y):
        """
        img_feat: (N, 2048) numpy
        osm_feat: (N, 30)   numpy
        met_feat: (N, 7)    numpy
        y:         (N,)     numpy
        """
        # 阶段1: 训练基础学习器
        if isinstance(img_feat, torch.Tensor):
            img_feat = img_feat.cpu().numpy()
        if isinstance(osm_feat, torch.Tensor):
            osm_feat = osm_feat.cpu().numpy()
        if isinstance(met_feat, torch.Tensor):
            met_feat = met_feat.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        # 图像模态用 Ridge 回归 (2048维 → 1维)
        from sklearn.linear_model import Ridge
        self.img_model = Ridge(alpha=1.0)
        self.img_model.fit(img_feat, y)
        pred1 = self.img_model.predict(img_feat)

        # OSM
        self.osm_model.fit(osm_feat, y)
        pred2 = self.osm_model.predict(osm_feat)

        # 气象
        self.met_model.fit(met_feat, y)
        pred3 = self.met_model.predict(met_feat)

        # 阶段2: 训练元学习器
        meta_X = np.column_stack([pred1, pred2, pred3])
        self.meta_model.fit(meta_X, y)

        return self

    def predict(self, img_feat, osm_feat, met_feat):
        if isinstance(img_feat, torch.Tensor):
            img_feat = img_feat.cpu().numpy()
        if isinstance(osm_feat, torch.Tensor):
            osm_feat = osm_feat.cpu().numpy()
        if isinstance(met_feat, torch.Tensor):
            met_feat = met_feat.cpu().numpy()

        pred1 = self.img_model.predict(img_feat)
        pred2 = self.osm_model.predict(osm_feat)
        pred3 = self.met_model.predict(met_feat)

        meta_X = np.column_stack([pred1, pred2, pred3])
        return self.meta_model.predict(meta_X)


# ============================================================
# 模块自测
# ============================================================
if __name__ == "__main__":
    print("=== M5: Early Fusion ===")
    m5 = EarlyFusionNet()
    img = torch.randn(4, 2048)
    osm = torch.randn(4, 30)
    met = torch.randn(4, 7)
    reg, cls = m5(img, osm, met)
    print(f"PM2.5: {reg.shape}, AQI: {cls.shape}")
    print(f"参数: {sum(p.numel() for p in m5.parameters()):,}")

    print("\n=== M6: Cross-Attention Fusion ===")
    m6 = CrossAttnFusionNet()
    reg, cls = m6(img, osm, met)
    print(f"PM2.5: {reg.shape}, AQI: {cls.shape}")
    print(f"参数: {sum(p.numel() for p in m6.parameters()):,}")

    print("\n=== M7: Late Fusion ===")
    m7 = LateFusionEnsemble()
    m7.fit(img.numpy(), osm.numpy(), met.numpy(), np.random.randn(4))
    pred = m7.predict(img.numpy(), osm.numpy(), met.numpy())
    print(f"PM2.5: {pred.shape}")
