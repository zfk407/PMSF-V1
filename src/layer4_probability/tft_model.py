"""
TFT (Temporal Fusion Transformer) 时间序列模型
寻找长期影响，自动调整不同时间窗口的权重
简化实现：基于注意力机制的时序特征编码 + MLP预测
"""
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True

    class SimpleTFT(nn.Module):
        """简化版TFT网络：变量选择 + 门控融合 + MLP"""

        def __init__(self, n_features: int, hidden_size: int):
            super().__init__()
            self.input_proj = nn.Linear(n_features, hidden_size)
            self.var_attention = nn.Sequential(
                nn.Linear(n_features, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, n_features),
                nn.Softmax(dim=1)
            )
            self.grn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(0.1)
            )
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid()
            )
            self.output = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

        def forward(self, x):
            attn_weights = self.var_attention(x)
            x_weighted = x * attn_weights
            h = self.input_proj(x_weighted)
            grn_out = self.grn(h)
            gate = self.gate(h)
            h = self.layer_norm(h + gate * grn_out)
            return self.output(h)

except ImportError:
    HAS_TORCH = False
    SimpleTFT = None


class TFTModel:
    """TFT时序融合模型（简化版）"""

    def __init__(self, config: dict):
        self.cfg = config["probability_models"]["tft"]
        self.hidden_size = self.cfg.get("hidden_size", 64)
        self.model = None
        self.feature_cols = None
        self.window_sizes = [5, 10, 30]  # 多尺度窗口

    def fit(self, train_df: pd.DataFrame, epochs: int = 20):
        """训练TFT模型"""
        if train_df.empty or "label" not in train_df.columns:
            print("[TFT] 训练数据无效")
            return

        self.feature_cols = [c for c in train_df.columns
                             if c not in ("issue", "number", "label")]

        if not HAS_TORCH:
            print("[TFT] 未安装torch，使用频率基线")
            return

        try:
            X = train_df[self.feature_cols].fillna(0).values.astype(np.float32)
            y = train_df["label"].values.astype(np.float32)

            n_features = X.shape[1]
            self.model = SimpleTFT(n_features, self.hidden_size)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.BCELoss()

            X_t = torch.FloatTensor(X)
            y_t = torch.FloatTensor(y).unsqueeze(1)

            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                pred = self.model(X_t)
                loss = criterion(pred, y_t)
                loss.backward()
                optimizer.step()

            print(f"[TFT] 训练完成，最终loss: {loss.item():.4f}")
        except Exception as e:
            print(f"[TFT] 训练失败: {e}，使用频率基线")
            self.model = None

    def predict_proba(self, test_df: pd.DataFrame) -> dict:
        """预测每个号码的出现概率"""
        if test_df.empty:
            return {num: 1 / 35 for num in range(1, 36)}

        if self.model is not None and HAS_TORCH and self.feature_cols:
            try:
                X = test_df[self.feature_cols].fillna(0).values.astype(np.float32)
                X_t = torch.FloatTensor(X)
                self.model.eval()
                with torch.no_grad():
                    probs = self.model(X_t).numpy().flatten()
                result = {}
                for i, row in test_df.iterrows():
                    result[int(row["number"])] = float(max(probs[i], 1e-6))
                total = sum(result.values())
                if total > 0:
                    result = {k: v / total for k, v in result.items()}
                return result
            except Exception as e:
                print(f"[TFT] 预测失败: {e}")

        # 兜底：时序趋势加权
        result = {}
        for _, row in test_df.iterrows():
            num = int(row["number"])
            # 短期频率 + 趋势
            short_freq = row.get("freq_short", 1 / 35)
            trend = row.get("trend", 0)
            score = short_freq * (1 + 0.5 * np.tanh(trend * 5))
            result[num] = max(score, 0.001)
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result
