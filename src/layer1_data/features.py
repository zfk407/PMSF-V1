"""
特征工程模块
为每个号码在每一期生成特征向量：遗漏、冷热、尾数、区域、状态、关系等
"""
import numpy as np
import pandas as pd
from collections import defaultdict


class FeatureEngineer:
    """特征工程器"""

    def __init__(self, config: dict):
        self.cfg = config
        self.front_range = range(
            config["data"]["front_range"][0],
            config["data"]["front_range"][1] + 1
        )
        self.w_short = config["features"]["window_short"]
        self.w_mid = config["features"]["window_mid"]
        self.w_long = config["features"]["window_long"]
        self.hot_th = config["features"]["hot_threshold"]
        self.cold_th = config["features"]["cold_threshold"]

    def build_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建逐号码逐期的特征数据集
        返回列: issue, number, label(0/1), 特征列...
        """
        front_cols = ["front01", "front02", "front03", "front04", "front05"]
        records = []
        # 预计算每期的号码集合
        issue_numbers = {}
        for _, row in df.iterrows():
            nums = set(int(row[c]) for c in front_cols)
            issue_numbers[row["issue"]] = nums

        issues = df["issue"].tolist()

        for idx, issue in enumerate(issues):
            # 历史窗口（不包含当前期）
            history = issues[:idx]
            if len(history) < self.w_long:
                continue  # 数据不足跳过
            current_nums = issue_numbers[issue]

            for num in self.front_range:
                feat = self._compute_number_features(num, history, issue_numbers, idx)
                feat["issue"] = issue
                feat["number"] = num
                feat["label"] = 1 if num in current_nums else 0
                records.append(feat)

        result = pd.DataFrame(records)
        return result

    def _compute_number_features(self, num: int, history: list,
                                  issue_numbers: dict, current_idx: int) -> dict:
        """计算单个号码的特征"""
        feat = {}
        # 基础属性
        feat["num"] = num
        feat["tail"] = num % 10
        feat["is_odd"] = 1 if num % 2 == 1 else 0
        feat["is_big"] = 1 if num >= 18 else 0
        # 区域
        if num <= 9:
            feat["zone"] = 1
        elif num <= 18:
            feat["zone"] = 2
        elif num <= 27:
            feat["zone"] = 3
        else:
            feat["zone"] = 4

        # 遗漏值（距上次出现的期数）
        miss = 0
        for past_issue in reversed(history):
            if num in issue_numbers[past_issue]:
                break
            miss += 1
        feat["miss"] = miss
        feat["miss_log"] = np.log1p(miss)

        # 各窗口出现频率
        for w_name, w in [("short", self.w_short), ("mid", self.w_mid), ("long", self.w_long)]:
            window = history[-w:] if len(history) >= w else history
            count = sum(1 for iss in window if num in issue_numbers[iss])
            freq = count / len(window) if window else 0
            feat[f"freq_{w_name}"] = freq
            feat[f"count_{w_name}"] = count

        # 冷热标记
        long_freq = feat.get("freq_long", 0)
        feat["is_hot"] = 1 if long_freq >= self.hot_th else 0
        feat["is_cold"] = 1 if long_freq <= self.cold_th else 0

        # 近期趋势（短期频率 - 长期频率）
        feat["trend"] = feat.get("freq_short", 0) - feat.get("freq_long", 0)

        # 平均遗漏间隔
        appearances = [i for i, iss in enumerate(history) if num in issue_numbers[iss]]
        if len(appearances) >= 2:
            gaps = np.diff(appearances)
            feat["avg_gap"] = float(np.mean(gaps))
            feat["std_gap"] = float(np.std(gaps))
        else:
            feat["avg_gap"] = len(history)
            feat["std_gap"] = 0.0

        # 邻号活跃度（num-1, num+1 的近期频率）
        neighbor_freq = 0
        for neighbor in [num - 1, num + 1]:
            if 1 <= neighbor <= 35:
                window = history[-self.w_short:]
                cnt = sum(1 for iss in window if neighbor in issue_numbers[iss])
                neighbor_freq += cnt / len(window) if window else 0
        feat["neighbor_freq"] = neighbor_freq

        # 同尾数活跃度
        tail_freq = 0
        tail_nums = [n for n in self.front_range if n % 10 == num % 10 and n != num]
        for tn in tail_nums:
            window = history[-self.w_short:]
            cnt = sum(1 for iss in window if tn in issue_numbers[iss])
            tail_freq += cnt / len(window) if window else 0
        feat["tail_freq"] = tail_freq

        return feat

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """获取特征列名（排除 issue, number, label）"""
        exclude = {"issue", "number", "label"}
        return [c for c in df.columns if c not in exclude]

    def build_current_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为"下一期"构建特征（使用全部历史数据）
        用于实际预测
        """
        front_cols = ["front01", "front02", "front03", "front04", "front05"]
        issue_numbers = {}
        for _, row in df.iterrows():
            nums = set(int(row[c]) for c in front_cols)
            issue_numbers[row["issue"]] = nums
        issues = df["issue"].tolist()
        current_idx = len(issues)

        records = []
        for num in self.front_range:
            feat = self._compute_number_features(num, issues, issue_numbers, current_idx)
            feat["issue"] = "NEXT"
            feat["number"] = num
            feat["label"] = -1  # 未知
            records.append(feat)
        return pd.DataFrame(records)
