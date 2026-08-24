"""
遗传算法组合优化
从候选组合中优化，目标：
最大化：概率 + 彭湃状态匹配 + 结构合理 + 稳定性
降低：异常集中 + 历史重复
"""
import numpy as np
import random
from copy import deepcopy


class GeneticOptimizer:
    """遗传算法组合优化器"""

    def __init__(self, config: dict):
        self.cfg = config["optimization"]["genetic_algorithm"]
        self.population_size = self.cfg.get("population_size", 500)
        self.generations = self.cfg.get("generations", 100)
        self.mutation_rate = self.cfg.get("mutation_rate", 0.1)
        self.crossover_rate = self.cfg.get("crossover_rate", 0.8)
        self.elite_size = self.cfg.get("elite_size", 20)
        self.front_nums = list(range(1, 36))
        self.back_nums = list(range(1, 13))

    def optimize(self, candidates: list, front_probs: dict,
                 state: str = "C", number_graph=None,
                 history_combos: list = None,
                 target_count: int = 4) -> list:
        """
        遗传算法优化，选出最优的target_count组组合
        candidates: [(front_tuple, back_tuple, score), ...]
        front_probs: {number: prob}
        state: 当前状态 A/B/C
        number_graph: 号码关系网络图
        history_combos: 历史组合列表（用于去重）
        target_count: 目标输出组数
        返回: [(front_tuple, back_tuple, fitness), ...]
        """
        if not candidates:
            return []

        # 如果候选数不够，直接返回top
        if len(candidates) <= target_count:
            return [(c[0], c[1], c[2]) for c in candidates[:target_count]]

        # 初始化种群（从候选中抽样 + 随机生成）
        population = self._init_population(candidates, front_probs)

        # 进化
        best_fitness_history = []
        for gen in range(self.generations):
            # 评估适应度
            fitness_scores = [
                self._fitness(ind, front_probs, state, number_graph, history_combos)
                for ind in population
            ]

            # 记录最佳
            best_idx = np.argmax(fitness_scores)
            best_fitness_history.append(fitness_scores[best_idx])

            # 选择 + 交叉 + 变异
            new_population = self._evolve(population, fitness_scores)
            population = new_population

            # 提前收敛判断
            if len(best_fitness_history) >= 10:
                recent = best_fitness_history[-10:]
                if max(recent) - min(recent) < 1e-4:
                    break

        # 最终评估
        final_scores = [
            self._fitness(ind, front_probs, state, number_graph, history_combos)
            for ind in population
        ]

        # 选择多样性最高的target_count组
        scored = list(zip(population, final_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # 确保组间多样性
        selected = self._select_diverse(scored, target_count)
        return selected

    def _init_population(self, candidates: list, front_probs: dict) -> list:
        """初始化种群"""
        population = []
        # 一半来自高概率候选
        n_from_candidates = min(self.population_size // 2, len(candidates))
        for i in range(n_from_candidates):
            front, back, _ = candidates[i]
            population.append({"front": list(front), "back": list(back)})

        # 一半随机生成（概率加权）
        probs_arr = np.array([front_probs.get(n, 1 / 35) for n in self.front_nums])
        probs_arr = probs_arr / probs_arr.sum()
        for _ in range(self.population_size - len(population)):
            front = sorted(np.random.choice(self.front_nums, 5, replace=False,
                                              p=probs_arr).tolist())
            back = sorted(np.random.choice(self.back_nums, 2, replace=False).tolist())
            population.append({"front": front, "back": back})

        return population

    def _fitness(self, individual: dict, front_probs: dict, state: str,
                 number_graph, history_combos: list) -> float:
        """
        适应度函数
        最大化：概率得分 + 状态匹配 + 关系得分 + 结构合理性
        惩罚：历史重复 + 号码集中
        """
        front = individual["front"]
        score = 0.0

        # 1. 概率得分（对数概率和）
        prob_score = sum(np.log(front_probs.get(n, 1 / 35) + 1e-10) for n in front)
        score += prob_score * 0.3

        # 2. 状态匹配
        state_bonus = self._state_match(front, state, front_probs)
        score += state_bonus * 0.25

        # 3. 关系网络得分（配对/纠缠）
        if number_graph is not None:
            relation_score = self._relation_score(front, number_graph)
            score += relation_score * 0.2

        # 4. 结构合理性（分散度）
        diversity = self._diversity_score(front)
        score += diversity * 0.15

        # 5. 历史重复惩罚
        if history_combos:
            repeat_penalty = self._repeat_penalty(front, history_combos)
            score -= repeat_penalty * 0.1

        return float(score)

    def _state_match(self, front: list, state: str, front_probs: dict) -> float:
        """状态匹配得分"""
        probs = [front_probs.get(n, 1 / 35) for n in front]
        median_prob = np.median(list(front_probs.values()))

        if state == "A":  # 纠缠热态：偏好热号集中
            hot_count = sum(1 for p in probs if p > median_prob * 1.2)
            return hot_count / 5.0
        elif state == "B":  # 终止冷态：偏好中等号码
            mid_count = sum(1 for p in probs
                            if median_prob * 0.8 < p < median_prob * 1.2)
            return mid_count / 5.0
        else:  # C 拓展回补态：偏好冷号
            cold_count = sum(1 for p in probs if p < median_prob * 0.8)
            return cold_count / 5.0

    def _relation_score(self, front: list, number_graph) -> float:
        """关系网络得分：号码间关系强度"""
        score = 0.0
        pairs = 0
        for i in range(len(front)):
            for j in range(i + 1, len(front)):
                rel = number_graph.get_relation_score(front[i], front[j])
                score += rel
                pairs += 1
        return score / pairs if pairs > 0 else 0

    def _diversity_score(self, front: list) -> float:
        """号码分散度得分"""
        # 区域覆盖
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
        zone_score = len(zones) / 4.0

        # 跨度得分
        span = max(front) - min(front)
        span_score = min(span / 30.0, 1.0)

        return (zone_score + span_score) / 2

    def _repeat_penalty(self, front: list, history_combos: list) -> float:
        """历史重复惩罚"""
        max_overlap = 0
        front_set = set(front)
        for hist in history_combos[-100:]:  # 只看最近100期
            overlap = len(front_set & set(hist))
            max_overlap = max(max_overlap, overlap)
        return max_overlap / 5.0  # 最多惩罚1.0

    def _evolve(self, population: list, fitness_scores: list) -> list:
        """进化一代：选择、交叉、变异"""
        # 精英保留
        scored = list(zip(population, fitness_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        elites = [deepcopy(ind) for ind, _ in scored[:self.elite_size]]

        # 轮盘赌选择
        total_fitness = sum(fitness_scores)
        if total_fitness <= 0:
            probs = np.ones(len(population)) / len(population)
        else:
            probs = np.array(fitness_scores) / total_fitness
            probs = np.maximum(probs, 0)
            probs = probs / probs.sum()

        new_population = elites[:]
        while len(new_population) < self.population_size:
            # 选择父母
            parent1 = population[np.random.choice(len(population), p=probs)]
            parent2 = population[np.random.choice(len(population), p=probs)]

            # 交叉
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = deepcopy(random.choice([parent1, parent2]))

            # 变异
            if random.random() < self.mutation_rate:
                child = self._mutate(child)

            new_population.append(child)

        return new_population[:self.population_size]

    def _crossover(self, parent1: dict, parent2: dict) -> dict:
        """交叉操作：合并父母的号码池再抽样"""
        all_front = list(set(parent1["front"] + parent2["front"]))
        all_back = list(set(parent1["back"] + parent2["back"]))

        if len(all_front) >= 5:
            front = sorted(random.sample(all_front, 5))
        else:
            # 补充随机号码
            remaining = [n for n in self.front_nums if n not in all_front]
            front = sorted(all_front + random.sample(remaining, 5 - len(all_front)))

        if len(all_back) >= 2:
            back = sorted(random.sample(all_back, 2))
        else:
            remaining = [n for n in self.back_nums if n not in all_back]
            back = sorted(all_back + random.sample(remaining, 2 - len(all_back)))

        return {"front": front, "back": back}

    def _mutate(self, individual: dict) -> dict:
        """变异操作：随机替换一个号码"""
        ind = deepcopy(individual)
        # 前区变异
        if random.random() < 0.7:
            idx = random.randint(0, 4)
            current = ind["front"][idx]
            available = [n for n in self.front_nums if n not in ind["front"]]
            if available:
                ind["front"][idx] = random.choice(available)
            ind["front"].sort()
        # 后区变异
        else:
            idx = random.randint(0, 1)
            available = [n for n in self.back_nums if n not in ind["back"]]
            if available:
                ind["back"][idx] = random.choice(available)
            ind["back"].sort()
        return ind

    def _select_diverse(self, scored: list, target_count: int) -> list:
        """选择多样性最高的target_count组"""
        selected = []
        remaining = scored.copy()

        # 先选得分最高的
        if remaining:
            best_ind, best_score = remaining.pop(0)
            selected.append((tuple(best_ind["front"]), tuple(best_ind["back"]), best_score))

        # 依次选择与已选组合差异最大的
        while len(selected) < target_count and remaining:
            best_candidate = None
            best_diversity = -1

            for ind, score in remaining:
                front_set = set(ind["front"])
                # 计算与已选组合的最小重叠（多样性）
                min_overlap = 5
                for sel_front, _, _ in selected:
                    overlap = len(front_set & set(sel_front))
                    min_overlap = min(min_overlap, overlap)
                diversity = 5 - min_overlap  # 差异越大越好
                # 综合得分：多样性 * 0.6 + 适应度 * 0.4
                combined = diversity * 0.6 + (score / 100) * 0.4
                if combined > best_diversity:
                    best_diversity = combined
                    best_candidate = (ind, score)

            if best_candidate:
                ind, score = best_candidate
                selected.append((tuple(ind["front"]), tuple(ind["back"]), score))
                remaining.remove(best_candidate)
            else:
                break

        return selected
