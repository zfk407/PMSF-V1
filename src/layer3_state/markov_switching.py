"""
Markov Switching 马尔可夫切换模型
处理不同状态下不同分布：纠缠态偏向热号延续，拓展态偏向冷号恢复
用于对号码概率做状态条件化修正
"""
import numpy as np
import pandas as pd


class MarkovSwitchingModel:
    """马尔可夫切换模型（状态依赖分布）"""

    def __init__(self, config: dict):
        self.cfg = config["state_models"]["markov_switching"]
        self.n_regimes = self.cfg.get("n_regimes", 3)
        self.regime_labels = ["A", "B", "C"]
        # 各状态下的号码分布参数
        self.regime_number_probs = {}  # {regime: {number: prob}}
        self.regime_feature_means = {}  # {regime: feature_mean_vector}
        self.transition_matrix = None
        self.feature_cols = [
            "hot_continuation", "cold_recovery", "zone_hhi",
            "pair_stability", "avg_miss", "sum_deviation"
        ]

    def fit(self, state_indicators: pd.DataFrame, history_df: pd.DataFrame,
            state_sequence: list = None):
        """
        训练Markov Switching模型
        学习各状态下的号码出现分布
        """
        if state_indicators.empty or history_df.empty:
            print("[MS] 数据不足")
            return

        front_cols = ["front01", "front02", "front03", "front04", "front05"]
        n = len(state_indicators)

        # 生成状态序列
        if state_sequence is None:
            from ..layer2_rules.three_states import ThreeStateSystem
            tss = ThreeStateSystem({"rules": {"three_states": {"state_duration_mean": 5}}})
            state_sequence = [tss.rule_based_state(row.to_dict())
                              for _, row in state_indicators.iterrows()]

        # 对齐历史数据（state_indicators从第5期开始）
        aligned_history = history_df.iloc[-n:].reset_index(drop=True) if len(history_df) >= n else history_df

        # 1. 各状态下的号码出现概率
        for regime in self.regime_labels:
            mask = np.array([s == regime for s in state_sequence])
            regime_nums = []
            for i in range(len(aligned_history)):
                if i < len(mask) and mask[i]:
                    row = aligned_history.iloc[i]
                    regime_nums.extend([int(row[c]) for c in front_cols])
            if regime_nums:
                counts = pd.Series(regime_nums).value_counts(normalize=True)
                self.regime_number_probs[regime] = {
                    num: float(counts.get(num, 0.001)) for num in range(1, 36)
                }
                # 归一化
                total = sum(self.regime_number_probs[regime].values())
                self.regime_number_probs[regime] = {
                    k: v / total for k, v in self.regime_number_probs[regime].items()
                }
            else:
                self.regime_number_probs[regime] = {num: 1 / 35 for num in range(1, 36)}

        # 2. 各状态下的特征均值
        X = state_indicators[self.feature_cols].fillna(0).values
        for regime in self.regime_labels:
            mask = np.array([s == regime for s in state_sequence])
            if mask.sum() > 0:
                self.regime_feature_means[regime] = X[mask].mean(axis=0)
            else:
                self.regime_feature_means[regime] = X.mean(axis=0)

        # 3. 转移矩阵
        self.transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        label_to_idx = {"A": 0, "B": 1, "C": 2}
        for i in range(len(state_sequence) - 1):
            from_idx = label_to_idx.get(state_sequence[i], 2)
            to_idx = label_to_idx.get(state_sequence[i + 1], 2)
            self.transition_matrix[from_idx, to_idx] += 1
        self.transition_matrix += 0.1
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)

        print(f"[MS] 训练完成，{len(self.regime_number_probs)}个状态分布已学习")

    def get_regime_number_probs(self, regime: str) -> dict:
        """获取指定状态下的号码概率分布"""
        return self.regime_number_probs.get(regime, {num: 1 / 35 for num in range(1, 36)})

    def apply_regime_bias(self, base_probs: dict, regime: str, weight: float = 0.3) -> dict:
        """
        将状态条件分布融合到基础概率中
        weight: 状态分布的融合权重（0=不融合，1=完全用状态分布）
        """
        regime_probs = self.get_regime_number_probs(regime)
        result = {}
        for num in range(1, 36):
            base = base_probs.get(num, 1 / 35)
            rp = regime_probs.get(num, 1 / 35)
            result[num] = base * (1 - weight) + rp * weight
        # 归一化
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result

    def infer_regime_from_features(self, features: dict) -> str:
        """根据特征向量推断最可能的状态（基于距离最近的状态特征均值）"""
        if not self.regime_feature_means:
            return "C"
        x = np.array([features.get(col, 0.5) for col in self.feature_cols])
        best_regime = "C"
        best_dist = float("inf")
        for regime, mean in self.regime_feature_means.items():
            dist = np.sum((x - mean) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_regime = regime
        return best_regime
