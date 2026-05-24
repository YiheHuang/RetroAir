"""MLP + Residual + FT-Transformer + DCN-V2 —— 表格数据的深度模型"""
import math
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """带 LayerNorm 的残差块"""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class ResMLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_blocks: int = 4, dropout: float = 0.15):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        z = self.in_proj(x)
        z = self.blocks(z)
        return self.head(z).squeeze(1)


class FTTransformerRegressor(nn.Module):
    """Feature Token Transformer —— 每个数值特征化为 token，self-attention 驱动交互。

    Y. Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021.
    """
    def __init__(self, input_dim: int, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 3, dropout: float = 0.15):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 特征 tokenizer: 每个标量特征 → d_model 向量（共享投影）
        self.tokenizer = nn.Linear(1, d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder (PreNorm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
            dim_feedforward=d_model * 4,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 预测头
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, F) → (B, F, 1) → (B, F, d_model)
        x = x.unsqueeze(-1)
        x = self.tokenizer(x)

        # Prepend [CLS]
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+F, d_model)

        # Self-attention
        x = self.transformer(x)

        # [CLS] → prediction
        return self.head(x[:, 0, :]).squeeze(1)


class CrossLayer(nn.Module):
    """DCN-V2 Cross 层: x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.W = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # x0: (B, F), xl: (B, F)
        return x0 * self.W(xl) + xl


class DCNV2Regressor(nn.Module):
    """Deep & Cross Network V2 —— 显式特征交叉 + 深度网络并行。

    R. Wang et al., "DCN V2: Improved Deep & Cross Network", WWW 2021.
    """
    def __init__(self, input_dim: int, deep_dim: int = 128,
                 n_cross_layers: int = 3, n_deep_layers: int = 3,
                 dropout: float = 0.15):
        super().__init__()

        # Cross Network: 显式特征交叉
        self.cross_layers = nn.ModuleList(
            [CrossLayer(input_dim) for _ in range(n_cross_layers)]
        )

        # Deep Network: 标准 MLP
        deep_layers = []
        prev = input_dim
        for _ in range(n_deep_layers):
            deep_layers.extend([
                nn.Linear(prev, deep_dim),
                nn.ReLU(),
                nn.BatchNorm1d(deep_dim),
                nn.Dropout(dropout),
            ])
            prev = deep_dim
        self.deep_net = nn.Sequential(*deep_layers)

        # 组合预测头
        combined_dim = input_dim + deep_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Cross pathway
        x_cross = x
        x0 = x
        for layer in self.cross_layers:
            x_cross = layer(x0, x_cross)

        # Deep pathway
        x_deep = self.deep_net(x)

        # Combine
        combined = torch.cat([x_cross, x_deep], dim=1)
        return self.head(combined).squeeze(1)
