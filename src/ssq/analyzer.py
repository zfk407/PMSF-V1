"""
双色球分析器
将彭湃规则层 + 统计概率 + 组合优化融合，输出2组红6+蓝1推荐
严谨科学：贝叶斯概率、Lift关联规则、指数衰减返点、结构统计过滤
"""
import numpy as np
from collections import defaultdict
from .rules import SsqPengpaiRules


class SsqAnalyzer:
    """双色球分析器"""

    RED_COLS = ["red01", "red02", "red03", "red04", "red05", "red06"]

    def __init__(self):
        self.rules = SsqPengpaiRules()

    # ---------- 号码综合概率 ----------
    def compute_red_probs(self, df, target_issue: str) -> dict:
        """
        红球综合概率 = 贝叶斯基础概率 * 彭湃规则修正
        修正因子：配对组热度、过渡号、纠缠、拓展、返点、蓝补红、尾数
        """
        analysis = self.rules.full_analysis(df, target_issue)
        bayes = analysis["bayes_probs"]

        # 各规则信号归一化
        rebound = analysis["rebound"]
        blue_boost = analysis["blue_boost"]
        extension = analysis["extension"]
        group_hot = analysis["group_hotness"]
        entanglement = analysis["entanglement"]
        hot_groups = set(analysis["hot_groups"])
        main_trans = analysis["main_transition"]
        sub_trans = set(analysis["sub_transitions"])
        cycle_status = analysis["cycle_status"]
        tail_law = analysis["tail_law"]

        probs = {}
        for num in range(1, 34):
            base = bayes.get(num, 1 / 33)
            factor = 1.0
            g = self.rules.get_group(num)

            # 1) 配对组热度加权
            gh = group_hot.get(g, 1.0)
            factor *= (0.6 + 0.4 * gh)

            # 2) 组内纠缠（热点组加权）
            if g in hot_groups:
                factor *= 1.25
            elif g in entanglement and entanglement[g]["is_hot"]:
                factor *= 1.25

            # 3) 组外拓展传导
            ext = extension.get(g, 0)
            factor *= (1 + 1.5 * max(0, ext))

            # 4) 隔期返点
            rb = rebound.get(num, 0)
            factor *= (1 + 0.6 * rb)

            # 5) 蓝补红
            bb = blue_boost.get(num, 0)
            factor *= (1 + 0.4 * bb)

            # 6) 过渡号加权（主+副）
            if num == main_trans:
                factor *= 1.5
            if num in sub_trans:
                factor *= 1.3

            # 7) 点位周期：走热雏形加权，暂停期降权
            cs = cycle_status.get(g, "normal")
            if cs == "hot":
                factor *= 1.2
            elif cs == "rest":
                factor *= 0.75

            # 8) 尾数定律：长期未出的尾数给与回补加权
            tail = num % 10
            cold = tail_law.get(tail, 100)
            if cold > 15:  # 尾数超过15期未出，回补加权
                factor *= (1 + 0.2 * min(cold / 30.0, 0.5))

            # 9) 配对联动：配对号近期活跃则本号受益
            pair = self.rules.get_pair(num)
            if pair:
                pair_bayes = bayes.get(pair, 1 / 33)
                factor *= (1 + 0.3 * (pair_bayes / (1 / 33) - 1))

            probs[num] = base * factor

        # 归一化为概率分布
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    # ---------- 蓝球概率 ----------
    def compute_blue_probs(self, df, target_issue: str, window: int = 100) -> dict:
        """蓝球概率：频率 + 遗漏 + 尾数平滑（Beta先验）"""
        line = self.rules.issue_parity(target_issue)
        lines = self.rules.split_lines(df)
        line_df = lines.get(line, df.tail(window))

        # 同线期蓝球
        blues = [int(r["blue"]) for _, r in line_df.tail(window).iterrows()]
        counts = defaultdict(int)
        for b in blues:
            counts[b] += 1
        n = len(blues)

        # 蓝球遗漏
        miss = {}
        for b in range(1, 17):
            m = 0
            for past in reversed(blues):
                if past == b:
                    break
                m += 1
            miss[b] = m if n > 0 else 0

        probs = {}
        alpha = 2.0  # Beta先验
        for b in range(1, 17):
            cnt = counts.get(b, 0)
            freq_prob = (cnt + alpha) / (n + alpha * 16)
            # 遗漏加权（遗漏越大越可能回补，但要有上限）
            miss_w = min(miss[b] / 25.0, 0.8)
            probs[b] = freq_prob * (1 + 0.35 * miss_w)

        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    # ---------- 状态判定 ----------
    def detect_state(self, df, target_issue: str) -> dict:
        """
        状态判定（彭湃双色球三态）：
        A-纠缠热态(热点组多、纠缠活跃) / B-终止冷态(热号退出、遗漏上升) / C-拓展回补态(冷号进入、区域扩散)
        """
        analysis = self.rules.full_analysis(df, target_issue)
        entanglement = analysis["entanglement"]
        extension = analysis["extension"]
        rebound = analysis["rebound"]
        tail_law = analysis["tail_law"]
        cycle_status = analysis["cycle_status"]

        # 指标
        hot_group_count = len(analysis["hot_groups"])
        entanglement_strength = sum(v["lift"] for v in entanglement.values() if v["is_hot"])
        extension_energy = sum(max(0, v) for v in extension.values())
        rebound_energy = sum(rebound.values()) / max(1, len(rebound))
        cold_tail_count = sum(1 for v in tail_law.values() if v > 15)
        rest_groups = sum(1 for s in cycle_status.values() if s == "rest")

        # 规则打分
        score_a = 0.35 * hot_group_count + 0.25 * min(entanglement_strength, 5) + 0.4 * min(rest_groups, 5)
        score_c = 0.4 * min(extension_energy, 4) + 0.3 * rebound_energy + 0.3 * min(cold_tail_count / 3, 1)
        score_b = max(0, 3.0 - score_a - score_c) * 0.8 + 0.3

        scores = {"A": round(score_a, 4), "B": round(score_b, 4), "C": round(score_c, 4)}
        total = sum(scores.values())
        probs = {k: v / total for k, v in scores.items()}
        state = max(probs, key=lambda k: probs[k])
        state_names = {"A": "纠缠热态", "B": "终止冷态", "C": "拓展回补态"}
        return {
            "state": state,
            "state_name": state_names[state],
            "probabilities": probs,
            "hot_group_count": hot_group_count,
            "cold_tail_count": cold_tail_count
        }

    # ---------- 结构过滤 ----------
    @staticmethod
    def _structure_check(reds: list) -> bool:
        """科学结构过滤：奇偶、大小、三区、和值、跨度"""
        # 奇偶：避免全奇/全偶（历史上6:0或0:6出现率<2%）
        odds = sum(1 for n in reds if n % 2 == 1)
        if odds == 0 or odds == 6:
            return False

        # 大小（1-16小，17-33大）：避免极端
        bigs = sum(1 for n in reds if n >= 17)
        if bigs == 0 or bigs == 6:
            return False

        # 三区覆盖（1-11, 12-22, 23-33）：至少覆盖2个区
        z1 = sum(1 for n in reds if 1 <= n <= 11)
        z2 = sum(1 for n in reds if 12 <= n <= 22)
        z3 = sum(1 for n in reds if 23 <= n <= 33)
        covered = sum(1 for z in (z1, z2, z3) if z > 0)
        if covered < 2:
            return False

        # 和值范围（双色球6红和值常见60-150，均值约102）
        s = sum(reds)
        if s < 55 or s > 165:
            return False

        # 跨度（历史常见15-32，极端值过滤）
        span = max(reds) - min(reds)
        if span < 12 or span > 31:
            return False

        # 连号检查：避免4连号及以上（历史罕见）
        sorted_reds = sorted(reds)
        max_consecutive = 1
        cur = 1
        for i in range(1, 6):
            if sorted_reds[i] == sorted_reds[i - 1] + 1:
                cur += 1
                max_consecutive = max(max_consecutive, cur)
            else:
                cur = 1
        if max_consecutive >= 4:
            return False

        return True

    # ---------- 蒙特卡洛采样 ----------
    def monte_carlo_sample(self, red_probs: dict, blue_probs: dict,
                           n_sim: int = 20000) -> list:
        """概率加权蒙特卡洛采样，生成候选组合"""
        red_nums = list(range(1, 34))
        red_weights = np.array([red_probs[n] for n in red_nums])
        red_weights = red_weights / red_weights.sum()

        blue_nums = list(range(1, 17))
        blue_weights = np.array([blue_probs[b] for b in blue_nums])
        blue_weights = blue_weights / blue_weights.sum()

        rng = np.random.default_rng(20260825)
        candidates = []
        for _ in range(n_sim):
            reds = tuple(sorted(rng.choice(red_nums, size=6, replace=False, p=red_weights)))
            if not self._structure_check(reds):
                continue
            blue = int(rng.choice(blue_nums, p=blue_weights))
            # 组合得分 = 红球概率和 + 蓝球概率
            score = sum(red_probs[n] for n in reds) + blue_probs[blue]
            candidates.append((reds, blue, score))
        return candidates

    # ---------- 组合优化（2组差异化） ----------
    def optimize_groups(self, df, target_issue: str, n_sim: int = 20000) -> list:
        """输出2组：A组模型共识(纯概率) + B组彭湃强化(规则匹配)"""
        red_probs = self.compute_red_probs(df, target_issue)
        blue_probs = self.compute_blue_probs(df, target_issue)

        # 预先计算彭湃规则信号（只算一次，避免循环内重复计算）
        analysis = self.rules.full_analysis(df, target_issue)
        main_trans = analysis["main_transition"]
        hot_groups = set(analysis["hot_groups"])

        candidates = self.monte_carlo_sample(red_probs, blue_probs, n_sim)
        if not candidates:
            # 兜底：直接用Top概率
            top_reds = tuple(sorted([n for n, _ in
                                     sorted(red_probs.items(), key=lambda x: x[1], reverse=True)[:6]]))
            top_blue = max(blue_probs, key=blue_probs.get)
            candidates = [(top_reds, top_blue, 0.0)]

        # A组：概率最大化
        best_a = max(candidates, key=lambda x: x[2])

        # B组：与A组差异化 + 规则匹配（过渡号/配对强化）
        def pengpai_bias_score(cand):
            reds, blue, score = cand
            bias = 0.0
            for n in reds:
                if n == main_trans:
                    bias += 0.15
                g = self.rules.get_group(n)
                if g in hot_groups:
                    bias += 0.08
                # 配对完整度：若配对号也在则加分
                pair = self.rules.get_pair(n)
                if pair and pair in reds:
                    bias += 0.05
            return score + bias

        # 差异化：与A组至少差3个红球
        a_reds = set(best_a[0])
        candidates_b = [c for c in candidates if len(set(c[0]) & a_reds) <= 3]
        if not candidates_b:
            candidates_b = candidates
        best_b = max(candidates_b, key=pengpai_bias_score)

        # 结果结构
        result = []
        for label, name, desc, (reds, blue, score) in [
            ("A", "模型共识组", "贝叶斯概率+彭湃规则综合评分最高，稳定性最强", best_a),
            ("B", "彭湃强化组", "过渡号/配对/纠缠规则加权，与A组差异化", best_b)
        ]:
            odd = sum(1 for n in reds if n % 2 == 1)
            big = sum(1 for n in reds if n >= 17)
            z1 = sum(1 for n in reds if 1 <= n <= 11)
            z2 = sum(1 for n in reds if 12 <= n <= 22)
            z3 = sum(1 for n in reds if 23 <= n <= 33)
            result.append({
                "label": label,
                "name": name,
                "description": desc,
                "front": [int(n) for n in reds],
                "blue": int(blue),
                "fitness": round(float(score), 6),
                "structure": {
                    "odd_even": f"{odd}奇{6 - odd}偶",
                    "big_small": f"{big}大{6 - big}小",
                    "zone": f"{z1}-{z2}-{z3}",
                    "sum": int(sum(reds)),
                    "span": int(max(reds) - min(reds)),
                    "passed": True
                }
            })
        return result, red_probs, blue_probs

    # ---------- 完整预测输出 ----------
    def predict(self, df, target_issue: str = None) -> dict:
        """完整预测流程"""
        if target_issue is None:
            latest = df["issue"].iloc[-1]
            try:
                year = int(latest[:2])
                seq = int(latest[2:])
                next_seq = seq + 1
                if next_seq > 154:
                    next_seq = 1
                    year += 1
                target_issue = f"{year:02d}{next_seq:03d}"
            except Exception:
                target_issue = "NEXT"

        # 规则全量分析
        analysis = self.rules.full_analysis(df, target_issue)

        # 状态
        state_info = self.detect_state(df, target_issue)

        # 优化2组
        groups, red_probs, blue_probs = self.optimize_groups(df, target_issue)

        # Top号码
        top_reds = sorted(red_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        top_blues = sorted(blue_probs.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "system": "ZHIHUI-DLT",
            "game": "ssq",
            "game_name": "双色球",
            "target_issue": target_issue,
            "generate_time": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_state": {
                "state": state_info["state"],
                "state_name": state_info["state_name"],
                "probabilities": {k: round(float(v), 4) for k, v in state_info["probabilities"].items()},
                "hot_group_count": state_info["hot_group_count"],
                "cold_tail_count": state_info["cold_tail_count"]
            },
            "rules": {
                "line": analysis["line"],
                "main_transition": int(analysis["main_transition"]),
                "sub_transitions": [int(n) for n in analysis["sub_transitions"]],
                "hot_groups": [int(g) for g in analysis["hot_groups"]],
                "pair_groups": [[str(a).zfill(2), str(b).zfill(2)] for a, b in analysis["pair_groups"]],
                "anchor_num": int(analysis["anchor_num"]),
                "top_rebound": [[int(n), round(float(p), 4)] for n, p in
                                sorted(analysis["rebound"].items(), key=lambda x: x[1], reverse=True)[:10]]
            },
            "groups": groups,
            "top10_reds": [{"number": int(n), "probability": round(float(p), 5)} for n, p in top_reds],
            "top5_blues": [{"number": int(n), "probability": round(float(p), 5)} for n, p in top_blues],
            "disclaimer": "本系统基于历史数据统计分析，双色球开奖为独立随机事件，结果仅供参考，不构成投注建议。"
        }
