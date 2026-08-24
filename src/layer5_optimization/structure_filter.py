"""
结构过滤器
对候选组合进行结构合理性过滤：
- 奇偶结构（避免5奇0偶等极端）
- 大小结构（避免全大全小）
- 四区覆盖（至少覆盖2个区）
- 和值范围（70-130）
- 跨度范围（15-34）
"""
import numpy as np


class StructureFilter:
    """组合结构过滤器"""

    def __init__(self, config: dict):
        self.cfg = config["optimization"]["structure_filter"]
        self.odd_even_allowed = self.cfg.get("odd_even_allowed",
                                               [[2, 3], [3, 2], [1, 4], [4, 1]])
        self.big_small_allowed = self.cfg.get("big_small_allowed",
                                                [[2, 3], [3, 2], [1, 4], [4, 1]])
        self.min_zone_coverage = self.cfg.get("min_zone_coverage", 2)
        self.sum_range = self.cfg.get("sum_range", [70, 130])
        self.span_range = self.cfg.get("span_range", [15, 34])

    def filter(self, candidates: list) -> list:
        """
        过滤候选组合
        candidates: [(front_tuple, back_tuple, score), ...]
        返回过滤后的列表
        """
        filtered = []
        for front, back, score in candidates:
            if self.check(front):
                filtered.append((front, back, score))
        return filtered

    def check(self, front: tuple) -> bool:
        """检查单个前区组合是否通过所有结构过滤"""
        return (
            self._check_odd_even(front) and
            self._check_big_small(front) and
            self._check_zone_coverage(front) and
            self._check_sum(front) and
            self._check_span(front)
        )

    def _check_odd_even(self, front: tuple) -> bool:
        """奇偶结构检查"""
        odds = sum(1 for n in front if n % 2 == 1)
        evens = 5 - odds
        return [odds, evens] in self.odd_even_allowed

    def _check_big_small(self, front: tuple) -> bool:
        """大小结构检查（大>=18）"""
        bigs = sum(1 for n in front if n >= 18)
        smalls = 5 - bigs
        return [bigs, smalls] in self.big_small_allowed

    def _check_zone_coverage(self, front: tuple) -> bool:
        """四区覆盖检查"""
        zones = set()
        for n in front:
            if n <= 9:
                zones.add(1)
            elif n <= 18:
                zones.add(2)
            elif n <= 27:
                zones.add(3)
            else:
                zones.add(4)
        return len(zones) >= self.min_zone_coverage

    def _check_sum(self, front: tuple) -> bool:
        """和值范围检查"""
        s = sum(front)
        return self.sum_range[0] <= s <= self.sum_range[1]

    def _check_span(self, front: tuple) -> bool:
        """跨度范围检查"""
        span = max(front) - min(front)
        return self.span_range[0] <= span <= self.span_range[1]

    def get_structure_info(self, front: tuple) -> dict:
        """获取组合的结构信息"""
        odds = sum(1 for n in front if n % 2 == 1)
        bigs = sum(1 for n in front if n >= 18)
        zones = []
        for n in front:
            if n <= 9:
                zones.append(1)
            elif n <= 18:
                zones.append(2)
            elif n <= 27:
                zones.append(3)
            else:
                zones.append(4)
        zone_counts = [zones.count(i) for i in range(1, 5)]
        return {
            "odd_even": f"{odds}奇{5 - odds}偶",
            "big_small": f"{bigs}大{5 - bigs}小",
            "zone": "-".join(map(str, zone_counts)),
            "sum": sum(front),
            "span": max(front) - min(front),
            "passed": self.check(front)
        }
