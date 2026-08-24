"""
HSMM 隐半马尔可夫模型（PMSF主状态模型）
相比HMM增加状态持续时间建模，学习"纠缠状态通常持续5期左右"等规律
输入: 冷热、遗漏、配对、区域、尾数、跨度
输出: STATE_A/B/C 概率 + 状态持续期数估计
"""
import numpy as np
import pandas as pd
from collections import defaultdict


class HSMMStateModel:
    """HSMM状态识别模型（含持续时间建模）"""

    def __init__(self, config: dict):
        self.cfg = config["state_models"]["hsmm"]
        self.n_states = self.cfg.get("n_states", 3)
        self.max_duration = self.cfg.get("max_duration", 15)
        self.state_labels = ["A", "B", "C"]
        # 模型参数
        self.initial_prob = None       # 初始状态概率
        self.transition_prob = None    # 状态转移矩阵
        self.duration_dist = None      # 各状态持续时间分布
        self.emission_params = None    # 发射参数（高斯均值/方差）
        self.feature_cols = [
            "hot_continuation", "cold_recovery", "zone_hhi",
            "pair_stability", "avg_miss", "sum_deviation"
        ]

    def fit(self, state_indicators: pd.DataFrame, state_sequence: list = None):
        """
        训练HSMM模型
        state_sequence: 可选的已知状态序列（来自规则判定），用于监督式参数估计
        """
        if state_indicators.empty or len(state_indicators) < 10:
            print("[HSMM] 数据不足")
            return

        X = state_indicators[self.feature_cols].fillna(0).values
        n_samples = len(X)

        # 如果没有提供状态序列，用规则判定生成
        if state_sequence is None:
            state_sequence = self._rule_state_sequence(state_indicators)

        # 确保状态序列长度匹配
        state_sequence = state_sequence[-n_samples:] if len(state_sequence) >= n_samples else state_sequence

        # 1. 初始状态概率
        self.initial_prob = np.zeros(self.n_states)
        for s in state_sequence[:5]:
            idx = self._label_to_idx(s)
            self.initial_prob[idx] += 1
        self.initial_prob = self.initial_prob / self.initial_prob.sum()

        # 2. 转移概率矩阵
        self.transition_prob = np.zeros((self.n_states, self.n_states))
        for i in range(len(state_sequence) - 1):
            from_idx = self._label_to_idx(state_sequence[i])
            to_idx = self._label_to_idx(state_sequence[i + 1])
            self.transition_prob[from_idx, to_idx] += 1
        # 加平滑
        self.transition_prob += 0.1
        row_sums = self.transition_prob.sum(axis=1, keepdims=True)
        self.transition_prob = self.transition_prob / row_sums

        # 3. 持续时间分布（按状态统计连续驻留期数）
        self.duration_dist = np.zeros((self.n_states, self.max_duration + 1))
        current_state = state_sequence[0]
        current_duration = 1
        for i in range(1, len(state_sequence)):
            if state_sequence[i] == current_state:
                current_duration += 1
            else:
                idx = self._label_to_idx(current_state)
                d = min(current_duration, self.max_duration)
                self.duration_dist[idx, d] += 1
                current_state = state_sequence[i]
                current_duration = 1
        # 最后一段
        idx = self._label_to_idx(current_state)
        d = min(current_duration, self.max_duration)
        self.duration_dist[idx, d] += 1
        # 归一化 + 平滑
        self.duration_dist += 0.01
        dur_sums = self.duration_dist.sum(axis=1, keepdims=True)
        self.duration_dist = self.duration_dist / dur_sums

        # 4. 发射参数（各状态下特征的高斯均值和方差）
        self.emission_params = []
        for s_idx in range(self.n_states):
            mask = np.array([self._label_to_idx(s) == s_idx for s in state_sequence])
            if mask.sum() > 0:
                mean = X[mask].mean(axis=0)
                var = X[mask].var(axis=0) + 1e-6
            else:
                mean = X.mean(axis=0)
                var = X.var(axis=0) + 1e-6
            self.emission_params.append({"mean": mean, "var": var})

        # 计算平均持续时间
        self.avg_durations = {}
        for s_idx, label in enumerate(self.state_labels):
            durations = np.arange(1, self.max_duration + 1)
            probs = self.duration_dist[s_idx, 1:]
            self.avg_durations[label] = float(np.sum(durations * probs) / probs.sum())

        print(f"[HSMM] 训练完成，平均持续时间: {self.avg_durations}")

    def _label_to_idx(self, label: str) -> int:
        mapping = {"A": 0, "B": 1, "C": 2}
        return mapping.get(label, 2)

    def _rule_state_sequence(self, indicators_df: pd.DataFrame) -> list:
        """用规则生成状态序列"""
        from ..layer2_rules.three_states import ThreeStateSystem
        tss = ThreeStateSystem({"rules": {"three_states": {"state_duration_mean": 5}}})
        sequence = []
        for _, row in indicators_df.iterrows():
            sequence.append(tss.rule_based_state(row.to_dict()))
        return sequence

    def _emission_logprob(self, x: np.ndarray, state_idx: int) -> float:
        """计算观测x在状态state_idx下的对数发射概率（高斯）"""
        mean = self.emission_params[state_idx]["mean"]
        var = self.emission_params[state_idx]["var"]
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * var) + (x - mean) ** 2 / var)
        return log_prob

    def predict_next(self, state_indicators: pd.DataFrame) -> dict:
        """
        预测下一期状态概率，考虑当前状态已持续时间
        返回: {"A": prob, "B": prob, "C": prob, "current_state": str, "current_duration": int}
        """
        if state_indicators.empty or self.transition_prob is None:
            return {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3,
                    "current_state": "C", "current_duration": 1}

        X = state_indicators[self.feature_cols].fillna(0).values

        # 用Viterbi风格确定当前状态和持续时间
        # 简化：从后往前找当前连续状态
        state_sequence = self._rule_state_sequence(state_indicators)
        current_state = state_sequence[-1]
        current_duration = 1
        for i in range(len(state_sequence) - 2, -1, -1):
            if state_sequence[i] == current_state:
                current_duration += 1
            else:
                break

        current_idx = self._label_to_idx(current_state)

        # 持续时间危险率：已持续d期后，下一期转移的概率
        d = min(current_duration, self.max_duration)
        # P(持续>d | 持续>=d) = sum_{k>d} P(k) / sum_{k>=d} P(k)
        survive_prob = self.duration_dist[current_idx, d:].sum() / (self.duration_dist[current_idx, d - 1:].sum() + 1e-8)
        # 下一期仍留在当前状态的概率
        stay_prob = max(0.05, min(0.95, survive_prob))
        # 转移概率（考虑持续时间修正）
        next_probs = self.transition_prob[current_idx].copy()
        next_probs[current_idx] = stay_prob  # 修正停留概率
        # 其他状态按原转移比例分配
        other_sum = next_probs.sum() - next_probs[current_idx]
        if other_sum > 0:
            for s in range(self.n_states):
                if s != current_idx:
                    next_probs[s] = (1 - stay_prob) * (self.transition_prob[current_idx, s] / other_sum)

        # 结合当前观测的发射概率做修正
        last_x = X[-1]
        emission_weights = np.array([
            np.exp(self._emission_logprob(last_x, s)) for s in range(self.n_states)
        ])
        emission_weights = emission_weights / (emission_weights.sum() + 1e-10)
        # 融合转移先验和发射似然
        fused = next_probs * 0.6 + emission_weights * 0.4
        fused = fused / fused.sum()

        result = {}
        for s_idx, label in enumerate(self.state_labels):
            result[label] = float(fused[s_idx])
        result["current_state"] = current_state
        result["current_duration"] = current_duration
        result["avg_durations"] = self.avg_durations

        return result

    def get_state_duration_estimate(self, state: str) -> float:
        """获取某状态的平均持续期数"""
        return self.avg_durations.get(state, 5.0)
