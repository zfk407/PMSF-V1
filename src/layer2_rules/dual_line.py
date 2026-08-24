"""
双线系统（彭湃核心规则之一）
将大乐透按期号奇偶拆分为单线(奇数期)和双线(偶数期)，分别学习节奏
禁止单线预测双线、双线预测单线，避免信息污染
"""
import pandas as pd
import numpy as np


class DualLineSystem:
    """双线系统"""

    def __init__(self, config: dict):
        self.cfg = config
        self.enabled = config["rules"]["dual_line"]["enabled"]

    @staticmethod
    def _issue_parity(issue: str) -> str:
        """判断期号奇偶（取期号最后一位）"""
        try:
            last_digit = int(str(issue)[-1])
            return "single" if last_digit % 2 == 1 else "double"
        except (ValueError, IndexError):
            return "unknown"

    def split(self, df: pd.DataFrame) -> dict:
        """
        将数据拆分为单线/双线两个子集
        返回: {"single": df_single, "double": df_double, "all": df}
        """
        if not self.enabled:
            return {"single": df, "double": df, "all": df}

        df = df.copy()
        df["line_type"] = df["issue"].apply(self._issue_parity)
        single = df[df["line_type"] == "single"].reset_index(drop=True)
        double = df[df["line_type"] == "double"].reset_index(drop=True)
        return {
            "single": single,
            "double": double,
            "all": df
        }

    def get_line_for_issue(self, issue: str) -> str:
        """获取指定期号所属线"""
        return self._issue_parity(issue)

    def get_line_data_for_prediction(self, df: pd.DataFrame, target_issue: str) -> pd.DataFrame:
        """
        根据目标期号，返回对应线的历史数据
        例如预测偶数期，只用偶数期历史数据训练
        """
        if not self.enabled:
            return df
        line = self._issue_parity(target_issue)
        split_data = self.split(df)
        return split_data.get(line, df)

    def analyze_line_rhythm(self, df: pd.DataFrame) -> dict:
        """
        分析单线/双线的节奏差异
        返回各线的统计特征
        """
        split_data = self.split(df)
        result = {}
        front_cols = ["front01", "front02", "front03", "front04", "front05"]
        for line_name, line_df in split_data.items():
            if line_df.empty:
                continue
            # 号码频率
            all_nums = []
            for _, row in line_df.iterrows():
                all_nums.extend([int(row[c]) for c in front_cols])
            freq = pd.Series(all_nums).value_counts(normalize=True).to_dict()
            # 和值统计
            sum_stats = {
                "mean": float(line_df["sum_front"].mean()),
                "std": float(line_df["sum_front"].std()),
                "min": int(line_df["sum_front"].min()),
                "max": int(line_df["sum_front"].max())
            }
            # 奇偶结构分布
            odd_even_dist = line_df["odd_even"].value_counts(normalize=True).to_dict()
            result[line_name] = {
                "count": len(line_df),
                "number_freq": freq,
                "sum_stats": sum_stats,
                "odd_even_dist": odd_even_dist
            }
        return result
