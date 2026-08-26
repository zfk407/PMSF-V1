"""
概率融合层 PMSF Ensemble
多个模型输出不简单平均，建立加权融合公式
权重通过滚动回测动态调整

最终评分 = 0.25*XGBoost + 0.20*CatBoost + 0.15*GNN + 0.15*TFT
         + 0.10*HSMM + 0.10*Bayesian + 0.05*彭湃规则修正
"""
import numpy as np
import pandas as pd
from collections import defaultdict


class ProbabilityFusion:
    """多模型概率融合器"""

    def __init__(self, config: dict):
        self.cfg = config
        weights = config["probability_models"]["fusion_weights"]
        self.weights = {
            "xgboost": weights.get("xgboost", 0.28),
            "catboost": weights.get("catboost", 0.15),
            "gnn": weights.get("gnn", 0.05),
            "tft": weights.get("tft", 0.05),
            "hsmm": weights.get("hsmm", 0.15),
            "bayesian": weights.get("bayesian", 0.15),
            "freq_line": weights.get("freq_line", 0.12),
            "pengpai_rule": weights.get("pengpai_rule", 0.05)
        }
        self.front_nums = list(range(1, 36))
        # 回测性能记录（用于动态权重调整）
        self.model_performance = defaultdict(list)

    def fuse(self, model_outputs: dict, state_probs: dict = None,
             rule_bias: dict = None) -> dict:
        """
        融合多个模型的概率输出
        model_outputs: {model_name: {number: prob}}
        state_probs: HSMM状态概率 {"A": p, "B": p, "C": p}
        rule_bias: 彭湃规则修正 {number: bias_factor}
        返回: {number: fused_probability}
        """
        # 收集所有可用模型
        available_models = [m for m in self.weights if m in model_outputs]
        if not available_models:
            return {num: 1 / 35 for num in self.front_nums}

        # 重新归一化可用模型的权重
        total_weight = sum(self.weights[m] for m in available_models)
        if total_weight > 0:
            norm_weights = {m: self.weights[m] / total_weight for m in available_models}
        else:
            norm_weights = {m: 1 / len(available_models) for m in available_models}

        # 加权融合
        fused = {}
        for num in self.front_nums:
            score = 0.0
            for model_name in available_models:
                prob = model_outputs[model_name].get(num, 1 / 35)
                score += norm_weights[model_name] * prob
            fused[num] = score

        # 贝叶斯后验修正（基于状态先验）
        if state_probs:
            fused = self._bayesian_correction(fused, state_probs)

        # 彭湃规则修正
        if rule_bias:
            for num in self.front_nums:
                bias = rule_bias.get(num, 1.0)
                fused[num] *= bias

        # 归一化
        total = sum(fused.values())
        if total > 0:
            fused = {k: v / total for k, v in fused.items()}

        return fused

    def _bayesian_correction(self, probs: dict, state_probs: dict) -> dict:
        """
        贝叶斯后验修正
        P(号码|状态) = P(状态|号码) * P(号码) / P(状态)
        简化：根据状态概率对号码概率做加权调整
        """
        # 状态对号码类型的偏向 (2026-08 深挖回测校准: 原始幅度过大,
        # B态/ C态下冷号加权实际负效(B-0.42/C-0.65), 已温和化)
        state_bias = {
            "A": {"hot": 1.08, "cold": 0.96},   # 纠缠热态：热号略升
            "B": {"hot": 0.99, "cold": 1.02},   # 终止冷态：近中性(实证热号仍偏多)
            "C": {"hot": 1.02, "cold": 1.04}    # 拓展回补态：仅极温和冷号倾向
        }

        result = {}
        for num, p in probs.items():
            factor = 1.0
            for state, sp in state_probs.items():
                if state in state_bias:
                    # 简单判断冷热（基于概率分位）
                    is_hot = p > np.median(list(probs.values()))
                    is_cold = p < np.percentile(list(probs.values()), 30)
                    if is_hot:
                        factor *= state_bias[state]["hot"] ** sp
                    if is_cold:
                        factor *= state_bias[state]["cold"] ** sp
            result[num] = p * factor

        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result

    def update_weights_from_backtest(self, model_hit_rates: dict):
        """
        根据回测命中率动态调整权重
        model_hit_rates: {model_name: hit_rate}
        """
        if not model_hit_rates:
            return

        # 记录性能
        for model, rate in model_hit_rates.items():
            self.model_performance[model].append(rate)

        # 基于近期平均命中率调整权重（指数加权）
        performance = {}
        for model, rates in self.model_performance.items():
            if rates:
                # 取最近10次的指数加权平均
                recent = rates[-10:]
                weights_arr = np.exp(np.linspace(-1, 0, len(recent)))
                performance[model] = float(np.average(recent, weights=weights_arr))
            else:
                performance[model] = 0.5

        # 性能加权（性能好的模型权重提升）
        if performance:
            perf_values = np.array(list(performance.values()))
            perf_values = perf_values - perf_values.min() + 0.01  # 平移到正
            perf_weights = perf_values / perf_values.sum()

            # 原始权重和性能权重各占50%
            for i, model in enumerate(performance.keys()):
                if model in self.weights:
                    original = self.weights[model]
                    self.weights[model] = original * 0.7 + perf_weights[i] * 0.3

            # 重新归一化
            total = sum(self.weights.values())
            if total > 0:
                self.weights = {k: v / total for k, v in self.weights.items()}

    def get_current_weights(self) -> dict:
        """获取当前融合权重"""
        return dict(self.weights)

    def rank_numbers(self, fused_probs: dict, top_k: int = 10) -> list:
        """按融合概率排序，返回top_k号码"""
        sorted_nums = sorted(fused_probs.items(), key=lambda x: x[1], reverse=True)
        return [(num, prob) for num, prob in sorted_nums[:top_k]]
