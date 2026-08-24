"""
XGBoost 号码概率排序模型
输入: 单个号码的特征向量（遗漏、冷热、尾数、区域、状态、关系）
输出: 该号码下一期出现的贡献概率
"""
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class XGBoostModel:
    """XGBoost概率模型"""

    def __init__(self, config: dict):
        self.cfg = config["probability_models"]["xgboost"]
        self.model = None
        self.feature_cols = None

    def fit(self, train_df: pd.DataFrame):
        """训练模型，train_df包含特征列和label列"""
        if train_df.empty or "label" not in train_df.columns:
            print("[XGB] 训练数据无效")
            return

        self.feature_cols = [c for c in train_df.columns
                             if c not in ("issue", "number", "label")]
        X = train_df[self.feature_cols].fillna(0).values
        y = train_df["label"].values

        if HAS_XGB:
            try:
                self.model = xgb.XGBClassifier(
                    n_estimators=self.cfg.get("n_estimators", 300),
                    max_depth=self.cfg.get("max_depth", 6),
                    learning_rate=self.cfg.get("learning_rate", 0.05),
                    subsample=self.cfg.get("subsample", 0.8),
                    colsample_bytree=self.cfg.get("colsample_bytree", 0.8),
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
                self.model.fit(X, y)
                print(f"[XGB] 训练完成，特征数: {len(self.feature_cols)}")
            except Exception as e:
                print(f"[XGB] 训练失败: {e}，使用频率基线")
                self.model = None
        else:
            print("[XGB] 未安装xgboost，使用频率基线")

    def predict_proba(self, test_df: pd.DataFrame) -> dict:
        """
        预测每个号码的出现概率
        返回: {number: probability}
        """
        if test_df.empty:
            return {num: 1 / 35 for num in range(1, 36)}

        if self.model is not None and HAS_XGB and self.feature_cols:
            try:
                X = test_df[self.feature_cols].fillna(0).values
                probs = self.model.predict_proba(X)[:, 1]
                result = {}
                for i, row in test_df.iterrows():
                    result[int(row["number"])] = float(probs[i])
                # 归一化
                total = sum(result.values())
                if total > 0:
                    result = {k: v / total for k, v in result.items()}
                return result
            except Exception as e:
                print(f"[XGB] 预测失败: {e}")

        # 兜底：基于频率的基线概率
        return self._frequency_baseline(test_df)

    def _frequency_baseline(self, df: pd.DataFrame) -> dict:
        """频率基线：用近期频率作为概率"""
        result = {}
        for _, row in df.iterrows():
            num = int(row["number"])
            freq = row.get("freq_short", 1 / 35)
            result[num] = max(freq, 0.001)
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result

    def get_feature_importance(self) -> dict:
        """获取特征重要性"""
        if self.model is not None and HAS_XGB and self.feature_cols:
            importance = self.model.feature_importances_
            return dict(zip(self.feature_cols, importance.tolist()))
        return {}
