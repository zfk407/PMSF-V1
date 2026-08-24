"""
CatBoost 类别变量处理模型
优势：自动处理区域、状态、尾数等类别变量
输入: 号码特征（含类别特征）
输出: 号码出现概率
"""
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


class CatBoostModel:
    """CatBoost概率模型"""

    def __init__(self, config: dict):
        self.cfg = config["probability_models"]["catboost"]
        self.model = None
        self.feature_cols = None
        self.cat_features = ["zone", "tail"]  # 类别特征

    def fit(self, train_df: pd.DataFrame):
        """训练模型"""
        if train_df.empty or "label" not in train_df.columns:
            print("[CatBoost] 训练数据无效")
            return

        self.feature_cols = [c for c in train_df.columns
                             if c not in ("issue", "number", "label")]
        # 确保类别特征存在
        cat_cols = [c for c in self.cat_features if c in train_df.columns]

        X = train_df[self.feature_cols].copy()
        y = train_df["label"].values

        # 类别特征转为字符串
        for c in cat_cols:
            X[c] = X[c].astype(str)

        if HAS_CAT:
            try:
                cat_indices = [self.feature_cols.index(c) for c in cat_cols if c in self.feature_cols]
                self.model = CatBoostClassifier(
                    iterations=self.cfg.get("iterations", 300),
                    depth=self.cfg.get("depth", 6),
                    learning_rate=self.cfg.get("learning_rate", 0.05),
                    verbose=self.cfg.get("verbose", False),
                    random_seed=42,
                    loss_function="Logloss",
                    eval_metric="Logloss"
                )
                self.model.fit(X, y, cat_features=cat_indices if cat_indices else None)
                print(f"[CatBoost] 训练完成，类别特征: {cat_cols}")
            except Exception as e:
                print(f"[CatBoost] 训练失败: {e}，使用频率基线")
                self.model = None
        else:
            print("[CatBoost] 未安装catboost，使用频率基线")

    def predict_proba(self, test_df: pd.DataFrame) -> dict:
        """预测每个号码的出现概率"""
        if test_df.empty:
            return {num: 1 / 35 for num in range(1, 36)}

        if self.model is not None and HAS_CAT and self.feature_cols:
            try:
                X = test_df[self.feature_cols].copy()
                cat_cols = [c for c in self.cat_features if c in X.columns]
                for c in cat_cols:
                    X[c] = X[c].astype(str)
                probs = self.model.predict_proba(X)[:, 1]
                result = {}
                for i, row in test_df.iterrows():
                    result[int(row["number"])] = float(probs[i])
                total = sum(result.values())
                if total > 0:
                    result = {k: v / total for k, v in result.items()}
                return result
            except Exception as e:
                print(f"[CatBoost] 预测失败: {e}")

        # 兜底
        result = {}
        for _, row in test_df.iterrows():
            num = int(row["number"])
            freq = row.get("freq_mid", 1 / 35)
            result[num] = max(freq, 0.001)
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result
