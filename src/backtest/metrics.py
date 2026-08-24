"""
评价指标系统
不使用单纯"中了几次"，采用：
1. 覆盖率 - 预测号码覆盖实际号码的比例
2. Top-K命中率 - 模型前K号码中实际出现几个
3. 结构命中 - 预测的奇偶/大小/四区结构是否成立
4. 状态命中 - 预测的状态是否与实际表现一致
"""
import numpy as np
import pandas as pd


class BacktestMetrics:
    """回测评价指标"""

    def __init__(self, config: dict):
        self.cfg = config["backtest"]
        self.top_k = self.cfg.get("top_k", 10)
        self.results = []

    def record(self, issue: str, predicted_numbers: list, actual_numbers: list,
               predicted_structure: dict = None, actual_structure: dict = None,
               predicted_state: str = None, actual_state: str = None,
               group_label: str = None):
        """记录一次预测结果"""
        record = {
            "issue": issue,
            "group": group_label,
            "predicted": predicted_numbers,
            "actual": actual_numbers,
            "coverage": self.coverage(predicted_numbers, actual_numbers),
            "top_k_hit": self.top_k_hit(predicted_numbers, actual_numbers),
            "structure_hit": self.structure_hit(predicted_structure, actual_structure),
            "state_hit": self.state_hit(predicted_state, actual_state),
            "exact_hit": len(set(predicted_numbers) & set(actual_numbers))
        }
        self.results.append(record)
        return record

    def coverage(self, predicted: list, actual: list) -> float:
        """
        覆盖率：预测号码集合中包含多少个实际号码
        例如预测20个号码，实际5个，覆盖3个 = 3/5 = 0.6
        """
        if not actual:
            return 0.0
        hit = len(set(predicted) & set(actual))
        return hit / len(actual)

    def top_k_hit(self, predicted_ranked: list, actual: list, k: int = None) -> int:
        """
        Top-K命中率：模型排名前K的号码中，实际出现几个
        predicted_ranked: 按概率从高到低排序的号码列表
        """
        if k is None:
            k = self.top_k
        top_k_nums = predicted_ranked[:k]
        return len(set(top_k_nums) & set(actual))

    def structure_hit(self, predicted: dict, actual: dict) -> dict:
        """
        结构命中率
        比较奇偶、大小、四区、和值、跨度结构
        """
        if not predicted or not actual:
            return {"odd_even": False, "big_small": False, "zone": False, "overall": False}

        result = {}
        # 奇偶
        result["odd_even"] = predicted.get("odd_even") == actual.get("odd_even")
        # 大小
        result["big_small"] = predicted.get("big_small") == actual.get("big_small")
        # 四区
        result["zone"] = predicted.get("zone") == actual.get("zone")
        # 和值范围
        pred_sum = predicted.get("sum")
        actual_sum = actual.get("sum")
        if pred_sum and actual_sum:
            result["sum_close"] = abs(pred_sum - actual_sum) <= 15
        else:
            result["sum_close"] = False
        # 总体
        result["overall"] = all([
            result["odd_even"], result["big_small"], result["zone"]
        ])
        return result

    def state_hit(self, predicted_state: str, actual_state: str) -> bool:
        """状态命中率"""
        if not predicted_state or not actual_state:
            return False
        return predicted_state == actual_state

    def summary(self) -> dict:
        """生成回测总结报告"""
        if not self.results:
            return {"message": "无回测数据"}

        df = pd.DataFrame(self.results)

        summary = {
            "total_predictions": len(df),
            "avg_coverage": float(df["coverage"].mean()),
            "avg_top_k_hit": float(df["top_k_hit"].mean()),
            "avg_exact_hit": float(df["exact_hit"].mean()),
            "max_exact_hit": int(df["exact_hit"].max()),
            "hit_3_plus": int((df["exact_hit"] >= 3).sum()),
            "hit_4_plus": int((df["exact_hit"] >= 4).sum()),
            "hit_5": int((df["exact_hit"] == 5).sum()),
        }

        # 结构命中统计
        structure_hits = [r["structure_hit"] for r in self.results if r["structure_hit"]]
        if structure_hits:
            struct_df = pd.DataFrame(structure_hits)
            summary["structure_odd_even_rate"] = float(struct_df["odd_even"].mean())
            summary["structure_big_small_rate"] = float(struct_df["big_small"].mean())
            summary["structure_zone_rate"] = float(struct_df["zone"].mean())
            summary["structure_overall_rate"] = float(struct_df["overall"].mean())

        # 状态命中
        state_hits = [r["state_hit"] for r in self.results if r["state_hit"] is not None]
        if state_hits:
            summary["state_hit_rate"] = float(np.mean(state_hits))

        # 按组统计
        if "group" in df.columns and df["group"].notna().any():
            group_stats = {}
            for group in df["group"].dropna().unique():
                group_df = df[df["group"] == group]
                group_stats[group] = {
                    "count": len(group_df),
                    "avg_coverage": float(group_df["coverage"].mean()),
                    "avg_exact_hit": float(group_df["exact_hit"].mean()),
                    "hit_3_plus": int((group_df["exact_hit"] >= 3).sum())
                }
            summary["by_group"] = group_stats

        return summary

    def get_model_hit_rates(self) -> dict:
        """获取各模型命中率（用于动态权重调整）"""
        if not self.results:
            return {}
        # 简化：用覆盖率作为模型性能指标
        return {
            "overall": float(np.mean([r["coverage"] for r in self.results]))
        }

    def reset(self):
        """重置回测结果"""
        self.results = []
