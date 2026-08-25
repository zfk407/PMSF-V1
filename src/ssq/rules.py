"""
彭湃双色球规则层 (核心)
实现彭湃双色球全部规则：
1. 隔期双线公理 - 按期号奇偶拆分独立线路
2. 恒值34配对 - 33个红球两两配对，和值恒为34
3. 过渡号机制 - 主固定(单线31/双线13) + 副动态双枢纽
4. 蓝代红规则 - 蓝球出号等价于对应红球点位计数
5. 点位周期判定 - 短期累计3次=小周期
6. 组内纠缠 - Lift>=1.25 有效纠缠热组
7. 组外拓展 - 热点向相邻组别传导
8. 隔期返点 - 1-5阶指数衰减加权(lambda=0.35)
9. 蓝补红 - 蓝球带动同号红球组别
10. 尾数定律 - 小尾数(1-5)不会长期全断
"""
import numpy as np
import pandas as pd
from collections import defaultdict


class SsqPengpaiRules:
    """彭湃双色球规则引擎"""

    # 恒值34配对：16组 + 1独立过渡号
    PAIR_GROUPS = [
        (1, 2), (3, 4), (5, 6), (7, 8),
        (9, 10), (11, 12), (13, 14), (15, 16),
        (17, 18), (19, 20), (21, 22), (23, 24),
        (25, 26), (27, 28), (29, 30), (31, 32),
    ]
    ANCHOR_NUM = 33  # 独立边界过渡号

    # 主过渡号：单期线31 / 双期线13
    MAIN_TRANSITION = {"single": 31, "double": 13}

    def __init__(self):
        self.red_cols = ["red01", "red02", "red03", "red04", "red05", "red06"]
        # 号码 -> 组索引映射
        self.num_to_group = {}
        for gi, (a, b) in enumerate(self.PAIR_GROUPS):
            self.num_to_group[a] = gi
            self.num_to_group[b] = gi
        self.num_to_group[self.ANCHOR_NUM] = len(self.PAIR_GROUPS)  # 33号独立组

    # ---------- 1. 双线公理 ----------
    @staticmethod
    def issue_parity(issue: str) -> str:
        """判断期号奇偶线：取期号最后一位"""
        try:
            last_digit = int(str(issue)[-1])
            return "single" if last_digit % 2 == 1 else "double"
        except (ValueError, IndexError):
            return "unknown"

    def split_lines(self, df: pd.DataFrame) -> dict:
        """按期号奇偶拆分为单期线/双期线"""
        df = df.copy()
        df["line_type"] = df["issue"].apply(self.issue_parity)
        return {
            "single": df[df["line_type"] == "single"].reset_index(drop=True),
            "double": df[df["line_type"] == "double"].reset_index(drop=True),
            "all": df
        }

    # ---------- 2. 分组配对 ----------
    def get_group(self, num: int) -> int:
        """返回号码所属组索引（0-15为配对组，16为独立过渡号组）"""
        return self.num_to_group.get(num, -1)

    def get_pair(self, num: int) -> int:
        """返回号码的恒值34配对号码（33号无配对返回None）"""
        if num == self.ANCHOR_NUM:
            return None
        return 34 - num

    # ---------- 3. 过渡号机制 ----------
    def get_main_transition(self, line: str) -> int:
        """主过渡号（固定）"""
        return self.MAIN_TRANSITION.get(line, 31)

    def detect_sub_transition(self, df: pd.DataFrame, window: int = 30,
                              top_n: int = 3) -> list:
        """
        副过渡号（动态识别）：
        位于两组热点纠缠衔接位 + 贝叶斯分Top10 + 近3期有隔期出号
        """
        recent = df.tail(window).copy()
        if recent.empty:
            return []

        # 贝叶斯分Top10号码
        bayes_probs = self.bayesian_number_probs(df)
        top10 = set(int(n) for n, _ in
                    sorted(bayes_probs.items(), key=lambda x: x[1], reverse=True)[:10])

        # 组热点度
        group_hot = self.compute_group_hotness(recent)

        # 近3期隔期出号（在当前期号的同线期）
        candidates = []
        for num in range(1, 34):
            if num in top10:
                continue  # 需要是衔接位号码，而非纯高频
            g = self.get_group(num)
            # 所在组及相邻组的热度
            group_heat = 0
            for dg in [g - 1, g, g + 1]:
                if dg in group_hot:
                    group_heat = max(group_heat, group_hot[dg])
            if group_heat <= 0:
                continue
            candidates.append((num, group_heat))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [int(n) for n, _ in candidates[:top_n]]

    # ---------- 4. 蓝代红 ----------
    def blue_as_red(self, blue_num: int) -> int:
        """蓝球号码等价于同号红球点位（1-16直接对应，17-33对应超出部分忽略）"""
        if 1 <= blue_num <= 33:
            return blue_num
        return None

    def apply_blue_impact(self, df: pd.DataFrame, num_counts: dict) -> dict:
        """蓝代红：蓝球出号等价于对应红球点位计数+1，改变组冷热周期"""
        counts = dict(num_counts)
        for _, row in df.iterrows():
            b = int(row["blue"])
            red_point = self.blue_as_red(b)
            if red_point:
                counts[red_point] = counts.get(red_point, 0) + 1
        return counts

    # ---------- 5. 点位周期判定 ----------
    def compute_point_cycles(self, df: pd.DataFrame, window: int = 15) -> dict:
        """
        点位周期：短期累计出现3次=完成小周期(可暂停)；>=3次=走热雏形
        返回: {group_idx: count_in_short_window}
        """
        recent = df.tail(window)
        group_counts = defaultdict(int)
        for _, row in recent.iterrows():
            for c in self.red_cols:
                g = self.get_group(int(row[c]))
                group_counts[g] += 1
        return dict(group_counts)

    def get_cycle_status(self, df: pd.DataFrame, window: int = 8,
                         include_blue: bool = True) -> dict:
        """
        点位组周期状态：rest(暂停) / hot(走热雏形) / normal
        短期窗口8期（每组期望约8*6/17≈2.8次），累计>=3次=走热雏形
        include_blue=True 时蓝代红：蓝球等价于对应红球点位计数+1，改变组冷热周期
        """
        recent = df.tail(window)
        group_counts = defaultdict(int)
        for _, row in recent.iterrows():
            for c in self.red_cols:
                g = self.get_group(int(row[c]))
                group_counts[g] += 1
            if include_blue:
                # 蓝代红：蓝球出号等价于对应红球点位计数+1
                bp = self.blue_as_red(int(row["blue"]))
                if bp:
                    bg = self.get_group(bp)
                    group_counts[bg] += 1
        status = {}
        for g in range(len(self.PAIR_GROUPS) + 1):  # 含独立组16
            cnt = group_counts.get(g, 0)
            if cnt >= 3:
                status[g] = "hot"      # 走热雏形
            elif cnt == 3:
                status[g] = "rest"     # 完成小周期，可暂停
            else:
                status[g] = "normal"
        return status

    # ---------- 6. 组内纠缠（关联规则） ----------
    def compute_entanglement(self, df: pd.DataFrame, min_lift: float = 1.25,
                             window: int = 200) -> dict:
        """
        组内纠缠：同一组两号码连续隔期交替开出 => 热点组
        用 Lift 量化：Lift>=1.25 为有效纠缠热组
        返回: {group_idx: {lift, support, is_hot}}
        """
        recent = df.tail(window)
        total = len(recent)
        if total == 0:
            return {}

        # 每组出现次数
        group_freq = defaultdict(int)
        for _, row in recent.iterrows():
            seen = set()
            for c in self.red_cols:
                g = self.get_group(int(row[c]))
                seen.add(g)
            for g in seen:
                group_freq[g] += 1

        # 组内两个号码共现次数
        result = {}
        for gi, (a, b) in enumerate(self.PAIR_GROUPS):
            a_cnt = 0
            b_cnt = 0
            ab_cnt = 0
            for _, row in recent.iterrows():
                nums = set(int(row[c]) for c in self.red_cols)
                a_in = a in nums
                b_in = b in nums
                if a_in:
                    a_cnt += 1
                if b_in:
                    b_cnt += 1
                if a_in and b_in:
                    ab_cnt += 1
            if a_cnt == 0 or b_cnt == 0:
                continue
            # Lift = P(AB) / (P(A)*P(B))
            p_a = a_cnt / total
            p_b = b_cnt / total
            p_ab = ab_cnt / total
            lift = p_ab / (p_a * p_b) if p_a * p_b > 0 else 0
            support = ab_cnt / total
            # 交替纠缠：最近10期内a/b是否交替出现
            recent10 = recent.tail(10)
            alt_occur = []
            for _, row in recent10.iterrows():
                nums = set(int(row[c]) for c in self.red_cols)
                if a in nums or b in nums:
                    alt_occur.append(1 if a in nums else 0)
            alternation = 0
            if len(alt_occur) >= 3:
                changes = sum(1 for i in range(1, len(alt_occur))
                              if alt_occur[i] != alt_occur[i - 1])
                alternation = changes / (len(alt_occur) - 1)
            result[gi] = {
                "lift": round(lift, 4),
                "support": round(support, 4),
                "is_hot": lift >= min_lift and alternation >= 0.4,
                "alternation": round(alternation, 4)
            }
        return result

    # ---------- 7. 组外拓展 ----------
    def compute_extension(self, df: pd.DataFrame, window: int = 30) -> dict:
        """
        组外拓展：热点组热度向相邻组传导（最多延伸1-2组）
        冷号启动多来自热点拓展
        返回: {group_idx: extension_score}
        """
        recent = df.tail(window)
        group_counts = defaultdict(int)
        for _, row in recent.iterrows():
            seen = set()
            for c in self.red_cols:
                seen.add(self.get_group(int(row[c])))
            for g in seen:
                group_counts[g] += 1

        total = sum(group_counts.values())
        if total == 0:
            return {}
        base_freq = {g: c / total for g, c in group_counts.items()}
        # 每组期望频率（均匀分布约1/17）
        expected = 1 / 17.0
        extension = {}
        for g in range(len(self.PAIR_GROUPS) + 1):
            freq = base_freq.get(g, 0)
            # 本组热度
            own_heat = max(0, freq - expected)
            # 相邻组传导（热源向两侧各传导1-2组）
            ext = 0
            for delta in [-2, -1, 1, 2]:
                ng = g + delta
                if ng in base_freq:
                    neighbor_heat = max(0, base_freq[ng] - expected)
                    ext += neighbor_heat * (0.5 ** abs(delta))
            extension[g] = round(own_heat + 0.4 * ext, 4)
        return extension

    # ---------- 8. 隔期返点（指数衰减） ----------
    @staticmethod
    def _order_weight(order: int, lam: float = 0.35) -> float:
        """1-5阶隔期指数衰减加权，近阶权重大"""
        return np.exp(-lam * (order - 1))

    def compute_rebound(self, df: pd.DataFrame, target_issue: str) -> dict:
        """
        隔期返点：临期隔期(上一同线期)基本必出1-2个重号；
        三隔期(往前第三个同线期)必出1个返点号
        量化：1-5阶隔期指数衰减加权(λ=0.35)
        返回: {number: rebound_score}
        """
        line = self.issue_parity(target_issue)
        lines = self.split_lines(df)
        line_df = lines.get(line, pd.DataFrame())
        if line_df.empty:
            return {}

        # 同线期序列（按期号升序）
        issues = line_df["issue"].tolist()
        if target_issue in issues:
            idx = issues.index(target_issue)
            same_line_history = line_df.iloc[:idx]
        else:
            same_line_history = line_df

        # 记录每期号码
        period_numbers = []
        for _, row in same_line_history.iterrows():
            period_numbers.append(set(int(row[c]) for c in self.red_cols))

        n = len(period_numbers)
        if n < 5:
            return {}

        # 1-5阶隔期返点
        rebound = defaultdict(float)
        for order in range(1, 6):
            if n - order < 0:
                break
            past_nums = period_numbers[n - order]
            w = self._order_weight(order)
            for num in past_nums:
                rebound[num] += w

        # 归一化
        max_score = max(rebound.values()) if rebound else 1
        return {k: round(v / max_score, 4) for k, v in rebound.items()}

    # ---------- 9. 蓝补红 ----------
    def compute_blue_boost(self, df: pd.DataFrame, window: int = 50) -> dict:
        """
        蓝补红：蓝球号码对应带动同号红球组别开出
        返回: {red_number: boost_score}
        """
        recent = df.tail(window)
        boost = defaultdict(float)
        for _, row in recent.iterrows():
            b = int(row["blue"])
            red_point = self.blue_as_red(b)
            if red_point:
                # 蓝球出现 => 同号红球点位活跃度+1，且带动配对号
                boost[red_point] += 1
                pair = self.get_pair(red_point)
                if pair:
                    boost[pair] += 0.3  # 配对联动
                # 带动所在组
                g = self.get_group(red_point)
                for gi, (a, b2) in enumerate(self.PAIR_GROUPS):
                    if gi == g:
                        boost[a] += 0.15
                        boost[b2] += 0.15
        max_score = max(boost.values()) if boost else 1
        return {k: round(v / max_score, 4) for k, v in boost.items()}

    # ---------- 10. 尾数定律 ----------
    def check_tail_law(self, df: pd.DataFrame, window: int = 100) -> dict:
        """
        尾数定律：单个号码可冷，但同一尾数不会长期全冷；
        小尾数(1-5尾)通常不会全断
        返回: {tail: 最近出现期数距离(0=最近开出, 越大越冷)}
        """
        recent = df.tail(window)
        last_seen = {}
        for i, (_, row) in enumerate(reversed(list(recent.iterrows()))):
            for c in self.red_cols:
                num = int(row[c])
                t = num % 10
                if t not in last_seen:
                    last_seen[t] = i
        # 补齐未出现的尾数
        for t in range(10):
            last_seen.setdefault(t, window)
        return last_seen

    # ---------- 概率计算 ----------
    def bayesian_number_probs(self, df: pd.DataFrame,
                              window: int = 100,
                              prior_alpha: float = 2.0) -> dict:
        """
        贝叶斯号码概率：Beta先验平滑
        P(n) = (count_n + alpha) / (total + alpha*33)
        """
        recent = df.tail(window)
        counts = defaultdict(int)
        for _, row in recent.iterrows():
            for c in self.red_cols:
                counts[int(row[c])] += 1
        total = sum(counts.values())
        probs = {}
        for n in range(1, 34):
            cnt = counts.get(n, 0)
            probs[n] = (cnt + prior_alpha) / (total + prior_alpha * 33)
        return probs

    def compute_group_hotness(self, df: pd.DataFrame, window: int = 30) -> dict:
        """组热点度：每组近window期出现频率"""
        recent = df.tail(window)
        n = len(recent)
        if n == 0:
            return {}
        group_counts = defaultdict(int)
        for _, row in recent.iterrows():
            seen = set()
            for c in self.red_cols:
                seen.add(self.get_group(int(row[c])))
            for g in seen:
                group_counts[g] += 1
        expected = n * 6 / 17.0  # 每组期望出现次数（6个号均匀分布）
        return {g: round(c / expected, 4) for g, c in group_counts.items()}

    def full_analysis(self, df: pd.DataFrame, target_issue: str) -> dict:
        """
        完整彭湃规则分析，输出结构化结果
        ★ 双线公理贯彻：除主/副过渡号判定外，所有红球信号只基于同线期历史，
        禁止跨线混看（彭湃第一铁律）
        """
        line = self.issue_parity(target_issue)
        main_trans = self.get_main_transition(line)

        # 同线期数据（预测只看同属性历史期数）
        lines = self.split_lines(df)
        line_df = lines.get(line, df)

        # 全部信号基于同线期
        sub_trans = self.detect_sub_transition(line_df)
        entanglement = self.compute_entanglement(line_df)
        extension = self.compute_extension(line_df)
        rebound = self.compute_rebound(df, target_issue)  # 内部已按同线期
        blue_boost = self.compute_blue_boost(line_df)
        tail_law = self.check_tail_law(line_df)
        group_hot = self.compute_group_hotness(line_df)
        cycle_status = self.get_cycle_status(line_df, include_blue=True)  # 蓝代红计入周期
        bayes = self.bayesian_number_probs(line_df)

        # 热点组列表
        hot_groups = [g for g, v in entanglement.items() if v["is_hot"]]

        return {
            "line": line,
            "line_periods": int(len(line_df)),   # 同线期数
            "main_transition": main_trans,
            "sub_transitions": sub_trans,
            "entanglement": entanglement,
            "hot_groups": hot_groups,
            "extension": extension,
            "rebound": rebound,
            "blue_boost": blue_boost,
            "tail_law": tail_law,
            "group_hotness": group_hot,
            "cycle_status": cycle_status,
            "bayes_probs": bayes,
            "pair_groups": self.PAIR_GROUPS,
            "anchor_num": self.ANCHOR_NUM
        }
