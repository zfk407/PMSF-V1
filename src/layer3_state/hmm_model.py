"""
HMM 隐藏马尔可夫模型
寻找状态变化路径，判断下一状态概率
输入: 冷热、遗漏、配对、区域、尾数、跨度等指标
输出: STATE_A/B/C 概率
"""
import numpy as np
import pandas as pd

try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False


class HMMStateModel:
    """HMM状态识别模型"""

    def __init__(self, config: dict):
        self.cfg = config["state_models"]["hmm"]
        self.n_components = self.cfg.get("n_components", 3)
        self.model = None
        self.feature_cols = [
            "hot_continuation", "cold_recovery", "zone_hhi",
            "pair_stability", "avg_miss", "sum_deviation"
        ]

    def fit(self, state_indicators: pd.DataFrame):
        """训练HMM模型"""
        if state_indicators.empty or len(state_indicators) < self.n_components * 2:
            print("[HMM] 数据不足，使用默认状态概率")
            return

        X = state_indicators[self.feature_cols].fillna(0).values

        if HAS_HMMLEARN:
            try:
                self.model = hmm.GaussianHMM(
                    n_components=self.n_components,
                    covariance_type=self.cfg.get("covariance_type", "full"),
                    n_iter=self.cfg.get("n_iter", 200),
                    random_state=42
                )
                self.model.fit(X)
                # 将HMM隐状态映射到 A/B/C（按热号延续率均值排序）
                self._map_states(X)
                print(f"[HMM] 训练完成，{self.n_components}个状态")
            except Exception as e:
                print(f"[HMM] 训练失败: {e}，使用规则判定")
                self.model = None
        else:
            print("[HMM] 未安装hmmlearn，使用规则判定")

    def _map_states(self, X: np.ndarray):
        """将HMM隐状态映射到 A(热)/B(冷)/C(拓展)"""
        if self.model is None:
            return
        states = self.model.predict(X)
        # 按每个状态的hot_continuation均值排序
        state_hot = {}
        for s in range(self.n_components):
            mask = states == s
            if mask.sum() > 0:
                state_hot[s] = X[mask, 0].mean()  # 第0列是hot_continuation
            else:
                state_hot[s] = 0.5
        # 排序：最热->A, 最冷->B, 中间->C
        sorted_states = sorted(state_hot.items(), key=lambda x: x[1], reverse=True)
        self.state_map = {}
        labels = ["A", "C", "B"]  # 热、中(拓展)、冷
        for i, (s, _) in enumerate(sorted_states):
            self.state_map[s] = labels[i] if i < len(labels) else "C"

    def predict_next(self, state_indicators: pd.DataFrame) -> dict:
        """
        预测下一期状态概率
        返回: {"A": prob, "B": prob, "C": prob}
        """
        if state_indicators.empty:
            return {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}

        if self.model is not None and HAS_HMMLEARN:
            try:
                X = state_indicators[self.feature_cols].fillna(0).values
                # 获取当前状态
                current_state = self.model.predict(X[-1:])[0]
                # 转移矩阵获取下一状态概率
                transmat = self.model.transmat_
                next_probs_raw = transmat[current_state]
                # 映射到 A/B/C
                result = {"A": 0.0, "B": 0.0, "C": 0.0}
                for s, prob in enumerate(next_probs_raw):
                    label = self.state_map.get(s, "C")
                    result[label] += prob
                # 归一化
                total = sum(result.values())
                if total > 0:
                    result = {k: v / total for k, v in result.items()}
                return result
            except Exception as e:
                print(f"[HMM] 预测失败: {e}")

        # 兜底：基于最近指标的规则判定
        last = state_indicators.iloc[-1].to_dict()
        return self._rule_predict(last)

    def _rule_predict(self, indicators: dict) -> dict:
        """规则兜底预测"""
        hot = indicators.get("hot_continuation", 0.5)
        cold = indicators.get("cold_recovery", 0.5)
        miss = indicators.get("avg_miss", 5)
        # 简单softmax
        score_a = hot * 2 + (1 - min(miss / 10, 1))
        score_b = (1 - hot) * 2 + min(miss / 10, 1)
        score_c = cold * 2 + 0.5
        scores = np.array([score_a, score_b, score_c])
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()
        return {"A": float(probs[0]), "B": float(probs[1]), "C": float(probs[2])}
