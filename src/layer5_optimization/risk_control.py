"""
风险控制器
- 与历史最多重复N个号
- 组间最小多样性
- 极端结构预警
"""
import numpy as np


class RiskController:
    """组合风险控制器"""

    def __init__(self, config: dict):
        self.cfg = config["optimization"]["risk_control"]
        self.max_repeat = self.cfg.get("max_repeat_with_history", 3)
        self.min_diversity = self.cfg.get("min_diversity", 0.3)

    def validate(self, groups: list, history_combos: list = None) -> dict:
        """
        验证一组组合的风险
        groups: [(front_tuple, back_tuple, score), ...]
        history_combos: 历史前区组合列表
        返回: {"passed": bool, "details": [...], "warnings": [...]}
        """
        details = []
        warnings = []
        all_passed = True

        for i, (front, back, score) in enumerate(groups):
            group_detail = {
                "group": chr(65 + i),  # A, B, C, D
                "front": list(front),
                "back": list(back),
                "risks": []
            }

            # 1. 历史重复检查
            if history_combos:
                max_overlap = 0
                for hist in history_combos[-50:]:
                    overlap = len(set(front) & set(hist))
                    max_overlap = max(max_overlap, overlap)
                if max_overlap > self.max_repeat:
                    group_detail["risks"].append(
                        f"与历史最多重复{max_overlap}个号（阈值{self.max_repeat}）"
                    )
                    warnings.append(f"{chr(65+i)}组: 历史重复{max_overlap}个号")
                    all_passed = False

            # 2. 极端结构检查
            structure_risk = self._check_extreme_structure(front)
            if structure_risk:
                group_detail["risks"].append(structure_risk)
                warnings.append(f"{chr(65+i)}组: {structure_risk}")

            # 3. 连号检查
            consecutive = self._check_consecutive(front)
            if consecutive:
                group_detail["risks"].append(consecutive)

            details.append(group_detail)

        # 4. 组间多样性检查
        if len(groups) >= 2:
            diversity_warnings = self._check_group_diversity(groups)
            warnings.extend(diversity_warnings)
            if diversity_warnings:
                all_passed = False

        return {
            "passed": all_passed,
            "details": details,
            "warnings": warnings
        }

    def _check_extreme_structure(self, front: tuple) -> str:
        """检查极端结构"""
        # 全奇/全偶
        odds = sum(1 for n in front if n % 2 == 1)
        if odds == 0 or odds == 5:
            return f"极端奇偶结构（{odds}奇{5-odds}偶）"

        # 全大/全小
        bigs = sum(1 for n in front if n >= 18)
        if bigs == 0 or bigs == 5:
            return f"极端大小结构（{bigs}大{5-bigs}小）"

        # 单区集中
        zones = [0, 0, 0, 0]
        for n in front:
            if n <= 9:
                zones[0] += 1
            elif n <= 18:
                zones[1] += 1
            elif n <= 27:
                zones[2] += 1
            else:
                zones[3] += 1
        if max(zones) >= 4:
            return f"区域过度集中（{max(zones)}个号在同一区）"

        # 和值极端
        s = sum(front)
        if s < 50 or s > 150:
            return f"和值极端（{s}）"

        return ""

    def _check_consecutive(self, front: tuple) -> str:
        """检查连号"""
        sorted_front = sorted(front)
        max_consecutive = 1
        current = 1
        for i in range(1, len(sorted_front)):
            if sorted_front[i] == sorted_front[i - 1] + 1:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 1
        if max_consecutive >= 4:
            return f"存在{max_consecutive}连号"
        return ""

    def _check_group_diversity(self, groups: list) -> list:
        """检查组间多样性"""
        warnings = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                front_i = set(groups[i][0])
                front_j = set(groups[j][0])
                overlap = len(front_i & front_j)
                diversity = 1 - overlap / 5.0
                if diversity < self.min_diversity:
                    warnings.append(
                        f"{chr(65+i)}组与{chr(65+j)}组重叠{overlap}个号，多样性{diversity:.2f}"
                    )
        return warnings

    def filter_risky(self, candidates: list, history_combos: list = None) -> list:
        """从候选中过滤掉高风险组合"""
        filtered = []
        for front, back, score in candidates:
            # 历史重复检查
            if history_combos:
                max_overlap = 0
                for hist in history_combos[-50:]:
                    overlap = len(set(front) & set(hist))
                    max_overlap = max(max_overlap, overlap)
                if max_overlap > self.max_repeat:
                    continue

            # 极端结构检查
            if self._check_extreme_structure(front):
                continue

            filtered.append((front, back, score))
        return filtered
