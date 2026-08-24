"""
三态系统（彭湃核心规则之三）
STATE-A: 纠缠热态 - 热号延续、配对活跃、结构稳定
STATE-B: 终止冷态 - 热号退出、关系断裂、遗漏增加
STATE-C: 拓展回补态 - 冷号进入、区域扩散、新关系形成

系统先判断当前处于什么状态，再在状态空间内优化选号
"""
import numpy as np
import pandas as pd
from collections import defaultdict


class ThreeStateSystem:
    """三态系统定义与规则判定"""

    STATE_A = "A"  # 纠缠热态
    STATE_B = "B"  # 终止冷态
    STATE_C = "C"  # 拓展回补态

    STATE_NAMES = {
        "A": "纠缠热态",
        "B": "终止冷态",
        "C": "拓展回补态"
    }

    def __init__(self, config: dict):
        self.cfg = config
        self.duration_mean = config["rules"]["three_states"]["state_duration_mean"]
        self.front_cols = ["front01", "front02", "front03", "front04", "front05"]

    def compute_state_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为每一期计算状态指标向量
        指标: 热号延续率、冷号回补率、区域集中度、配对稳定度、遗漏变化
        """
        if len(df) < 10:
            return pd.DataFrame()

        records = []
        # 预计算每期号码集合
        issue_nums = []
        for _, row in df.iterrows():
            issue_nums.append(set(int(row[c]) for c in self.front_cols))

        for idx in range(5, len(df)):
            current = issue_nums[idx]
            prev5 = issue_nums[max(0, idx - 5):idx]
            prev10 = issue_nums[max(0, idx - 10):idx]

            # 1. 热号延续率：当前期号码中，在前5期出现过>=2次的比例
            hot_count = 0
            for num in current:
                appear = sum(1 for s in prev5 if num in s)
                if appear >= 2:
                    hot_count += 1
            hot_continuation = hot_count / 5.0

            # 2. 冷号回补率：当前期号码中，在前10期出现<=1次的比例
            cold_count = 0
            for num in current:
                appear = sum(1 for s in prev10 if num in s)
                if appear <= 1:
                    cold_count += 1
            cold_recovery = cold_count / 5.0

            # 3. 区域集中度（HHI指数）
            zones = [self._num_zone(n) for n in current]
            zone_counts = pd.Series(zones).value_counts(normalize=True)
            zone_hhi = float((zone_counts ** 2).sum())

            # 4. 配对稳定度：当前期号码对中，在前5期共现过的比例
            pair_total = 0
            pair_stable = 0
            current_list = sorted(current)
            for i in range(len(current_list)):
                for j in range(i + 1, len(current_list)):
                    pair_total += 1
                    a, b = current_list[i], current_list[j]
                    for s in prev5:
                        if a in s and b in s:
                            pair_stable += 1
                            break
            pair_stability = pair_stable / pair_total if pair_total else 0

            # 5. 平均遗漏变化
            current_miss = []
            for num in current:
                miss = 0
                for past_idx in range(idx - 1, -1, -1):
                    if num in issue_nums[past_idx]:
                        break
                    miss += 1
                current_miss.append(miss)
            avg_miss = float(np.mean(current_miss))

            # 6. 和值偏离度（相对近10期均值）
            recent_sums = df.iloc[max(0, idx - 10):idx]["sum_front"].values
            sum_deviation = abs(df.iloc[idx]["sum_front"] - np.mean(recent_sums)) / (np.std(recent_sums) + 1e-8)

            records.append({
                "issue": df.iloc[idx]["issue"],
                "hot_continuation": hot_continuation,
                "cold_recovery": cold_recovery,
                "zone_hhi": zone_hhi,
                "pair_stability": pair_stability,
                "avg_miss": avg_miss,
                "sum_deviation": float(sum_deviation)
            })

        return pd.DataFrame(records)

    @staticmethod
    def _num_zone(num: int) -> int:
        if num <= 9:
            return 1
        elif num <= 18:
            return 2
        elif num <= 27:
            return 3
        else:
            return 4

    def rule_based_state(self, indicators: dict) -> str:
        """
        基于彭湃规则的硬判定状态
        纠缠热态(A): 热号延续高 + 配对稳定 + 区域集中
        终止冷态(B): 热号延续低 + 遗漏增加 + 配对断裂
        拓展回补态(C): 冷号回补高 + 区域扩散 + 和值偏离
        """
        hot = indicators.get("hot_continuation", 0.5)
        cold = indicators.get("cold_recovery", 0.5)
        hhi = indicators.get("zone_hhi", 0.4)
        pair = indicators.get("pair_stability", 0.3)
        miss = indicators.get("avg_miss", 5)
        sum_dev = indicators.get("sum_deviation", 1.0)

        # A态评分
        score_a = hot * 0.3 + pair * 0.3 + (1 - hhi + 0.5) * 0.2 + (1 - min(miss / 10, 1)) * 0.2
        # B态评分
        score_b = (1 - hot) * 0.3 + (1 - pair) * 0.3 + min(miss / 10, 1) * 0.2 + (1 - cold) * 0.2
        # C态评分
        score_c = cold * 0.3 + (1 - hhi + 0.5) * 0.2 + min(sum_dev / 3, 1) * 0.25 + min(miss / 10, 1) * 0.25

        scores = {"A": score_a, "B": score_b, "C": score_c}
        return max(scores, key=scores.get)

    def get_state_bias(self, state: str) -> dict:
        """
        根据状态返回选号偏向
        返回各类型号码的权重偏置
        """
        if state == "A":  # 纠缠热态
            return {
                "hot_weight": 1.5,      # 热号加权
                "cold_weight": 0.5,     # 冷号降权
                "pair_weight": 1.3,     # 配对加权
                "zone_focus": True,     # 区域集中
                "trend_follow": True    # 趋势跟随
            }
        elif state == "B":  # 终止冷态
            return {
                "hot_weight": 0.6,
                "cold_weight": 1.2,
                "pair_weight": 0.7,
                "zone_focus": False,
                "trend_follow": False
            }
        else:  # C 拓展回补态
            return {
                "hot_weight": 0.9,
                "cold_weight": 1.6,     # 冷号回补加权
                "pair_weight": 1.0,
                "zone_focus": False,    # 区域扩散
                "trend_follow": False,
                "recovery_bonus": 1.4   # 回补额外加成
            }

    def apply_state_bias_to_probabilities(self, probs: dict, state: str,
                                            number_attrs: dict = None) -> dict:
        """
        将状态偏置应用到号码概率上
        probs: {number: probability}
        number_attrs: {number: {"is_hot": bool, "is_cold": bool, ...}}
        """
        bias = self.get_state_bias(state)
        result = {}
        for num, p in probs.items():
            factor = 1.0
            if number_attrs and num in number_attrs:
                attrs = number_attrs[num]
                if attrs.get("is_hot"):
                    factor *= bias.get("hot_weight", 1.0)
                if attrs.get("is_cold"):
                    factor *= bias.get("cold_weight", 1.0)
            result[num] = p * factor
        # 归一化
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result
