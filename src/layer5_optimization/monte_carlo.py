"""
Monte Carlo 蒙特卡洛模拟
概率加权随机生成5前区+2后区组合，不是纯随机
从大量模拟中筛选高概率候选
"""
import numpy as np
import random
from tqdm import tqdm


class MonteCarloSampler:
    """蒙特卡洛采样器"""

    def __init__(self, config: dict):
        self.cfg = config["optimization"]["monte_carlo"]
        self.n_simulations = self.cfg.get("n_simulations", 1000000)
        self.n_candidates = self.cfg.get("n_candidates", 10000)
        self.front_nums = list(range(1, 36))
        self.back_nums = list(range(1, 13))

    def sample(self, front_probs: dict, back_probs: dict = None,
               n_simulations: int = None) -> list:
        """
        概率加权蒙特卡洛采样
        front_probs: {number: probability} 前区概率
        back_probs: {number: probability} 后区概率（None则均匀）
        返回: 候选组合列表 [(front_tuple, back_tuple, score), ...]
        """
        if n_simulations is None:
            n_simulations = min(self.n_simulations, 100000)  # 实际运行时限制

        # 构建概率数组
        front_probs_arr = np.array([front_probs.get(n, 1 / 35) for n in self.front_nums])
        front_probs_arr = front_probs_arr / front_probs_arr.sum()

        if back_probs:
            back_probs_arr = np.array([back_probs.get(n, 1 / 12) for n in self.back_nums])
        else:
            back_probs_arr = np.ones(12) / 12
        back_probs_arr = back_probs_arr / back_probs_arr.sum()

        candidates = []
        seen = set()

        # 批量采样（向量化加速）
        batch_size = min(n_simulations, 50000)
        n_batches = (n_simulations + batch_size - 1) // batch_size

        for _ in range(n_batches):
            current_batch = min(batch_size, n_simulations - len(candidates))
            if current_batch <= 0:
                break

            # 前区：无放回抽样5个（概率加权）
            for _ in range(current_batch):
                front = self._weighted_sample_without_replacement(
                    self.front_nums, front_probs_arr, 5
                )
                back = self._weighted_sample_without_replacement(
                    self.back_nums, back_probs_arr, 2
                )
                front_tuple = tuple(sorted(front))
                back_tuple = tuple(sorted(back))
                key = (front_tuple, back_tuple)

                if key not in seen:
                    seen.add(key)
                    # 计算组合得分
                    score = self._combo_score(front_tuple, front_probs)
                    candidates.append((front_tuple, back_tuple, score))

        # 按得分排序，保留top候选
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:self.n_candidates]

    def _weighted_sample_without_replacement(self, items: list, probs: np.ndarray,
                                               k: int) -> list:
        """概率加权无放回抽样"""
        remaining_probs = probs.copy()
        remaining_items = list(items)
        selected = []
        for _ in range(k):
            if sum(remaining_probs) <= 0:
                # 均匀选
                idx = random.randint(0, len(remaining_items) - 1)
            else:
                p = remaining_probs / remaining_probs.sum()
                idx = np.random.choice(len(remaining_items), p=p)
            selected.append(remaining_items.pop(idx))
            remaining_probs = np.delete(remaining_probs, idx)
        return selected

    def _combo_score(self, front: tuple, probs: dict) -> float:
        """组合得分 = 号码概率乘积（对数求和避免下溢）"""
        score = sum(np.log(probs.get(n, 1 / 35) + 1e-10) for n in front)
        return float(score)

    def sample_with_copula(self, front_probs: dict, copula_model,
                            n_simulations: int = None) -> list:
        """
        结合Copula依赖的采样
        先选一个种子号码，然后根据条件概率选后续号码
        """
        if n_simulations is None:
            n_simulations = min(self.n_simulations, 50000)

        candidates = []
        seen = set()

        for _ in range(n_simulations):
            # 第一个号码按基础概率
            first = self._weighted_sample_without_replacement(
                self.front_nums,
                np.array([front_probs.get(n, 1 / 35) for n in self.front_nums]),
                1
            )[0]
            selected = [first]

            # 后续号码根据Copula条件概率
            for _ in range(4):
                cond_probs = {}
                for num in self.front_nums:
                    if num not in selected:
                        cond_p = copula_model.get_conditional_prob(num, selected)
                        base_p = front_probs.get(num, 1 / 35)
                        cond_probs[num] = base_p * 0.5 + cond_p * 0.5
                nums = list(cond_probs.keys())
                probs_arr = np.array([cond_probs[n] for n in nums])
                probs_arr = probs_arr / probs_arr.sum()
                next_num = np.random.choice(nums, p=probs_arr)
                selected.append(next_num)

            front_tuple = tuple(sorted(selected))
            back = self._weighted_sample_without_replacement(
                self.back_nums, np.ones(12) / 12, 2
            )
            back_tuple = tuple(sorted(back))
            key = (front_tuple, back_tuple)

            if key not in seen:
                seen.add(key)
                score = self._combo_score(front_tuple, front_probs)
                candidates.append((front_tuple, back_tuple, score))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:self.n_candidates]
