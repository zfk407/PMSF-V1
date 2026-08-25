"""
PMSF-V1 彭湃大乐透多尺度状态融合系统 - 主入口
完整运行流程：数据 -> 规则 -> 状态 -> 概率 -> 优化 -> 输出
"""
import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.layer1_data.database import DltDatabase
from src.layer1_data.fetcher import DltDataFetcher
from src.layer1_data.features import FeatureEngineer
from src.layer2_rules.dual_line import DualLineSystem
from src.layer2_rules.number_graph import NumberGraph
from src.layer2_rules.three_states import ThreeStateSystem
from src.layer3_state.hmm_model import HMMStateModel
from src.layer3_state.hsmm_model import HSMMStateModel
from src.layer3_state.markov_switching import MarkovSwitchingModel
from src.layer4_probability.xgb_model import XGBoostModel
from src.layer4_probability.catboost_model import CatBoostModel
from src.layer4_probability.tft_model import TFTModel
from src.layer4_probability.gnn_model import GNNModel
from src.layer4_probability.copula_model import CopulaModel
from src.layer4_probability.fusion import ProbabilityFusion
from src.layer5_optimization.monte_carlo import MonteCarloSampler
from src.layer5_optimization.structure_filter import StructureFilter
from src.layer5_optimization.genetic_algorithm import GeneticOptimizer
from src.layer5_optimization.risk_control import RiskController
from src.backtest.walk_forward import WalkForwardBacktester
from src.backtest.metrics import BacktestMetrics
from src.output.generator import OutputGenerator


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PMSFSystem:
    """PMSF-V1 系统主类"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        np.random.seed(self.config["system"]["random_seed"])

        # 初始化各层组件
        self.db = DltDatabase(self.config["data"]["db_path"])
        self.fetcher = DltDataFetcher(self.config["data"]["raw_dir"])
        self.fe = FeatureEngineer(self.config)
        self.dual_line = DualLineSystem(self.config)
        self.number_graph = NumberGraph(self.config)
        self.three_states = ThreeStateSystem(self.config)
        self.hmm = HMMStateModel(self.config)
        self.hsmm = HSMMStateModel(self.config)
        self.ms = MarkovSwitchingModel(self.config)
        self.xgb = XGBoostModel(self.config)
        self.catboost = CatBoostModel(self.config)
        self.tft = TFTModel(self.config)
        self.gnn = GNNModel(self.config)
        self.copula = CopulaModel(self.config)
        self.fusion = ProbabilityFusion(self.config)
        self.mc = MonteCarloSampler(self.config)
        self.struct_filter = StructureFilter(self.config)
        self.ga = GeneticOptimizer(self.config)
        self.risk = RiskController(self.config)
        self.output_gen = OutputGenerator(self.config)

        self.data = None
        self.feature_df = None

    def load_data(self, use_web: bool = True, use_mock: bool = True) -> pd.DataFrame:
        """第一层：加载数据"""
        print("\n" + "=" * 60)
        print("  第一层：数据基础层")
        print("=" * 60)

        # 先尝试从数据库加载
        if self.db.count() > 100:
            print(f"[数据] 从数据库加载 {self.db.count()} 期历史数据")
            self.data = self.db.load_all()
        else:
            # 从网络/CSV/模拟获取
            self.data = self.fetcher.get_data(use_web=use_web, use_mock_fallback=use_mock)
            if not self.data.empty:
                # 存入数据库
                self.db.insert_batch(self.data)
                print(f"[数据] 已存入数据库，共 {len(self.data)} 期")

        if self.data is None or self.data.empty:
            raise RuntimeError("无法获取数据，请检查网络或数据文件")

        print(f"[数据] 数据范围: {self.data['issue'].iloc[0]} ~ {self.data['issue'].iloc[-1]}")
        print(f"[数据] 最新期号: {self.data['issue'].iloc[-1]}")
        return self.data

    def build_features(self) -> pd.DataFrame:
        """第一层：特征工程"""
        print("\n[特征] 开始构建号码特征...")
        self.feature_df = self.fe.build_dataset(self.data)
        print(f"[特征] 特征数据集: {len(self.feature_df)} 条记录, "
              f"{len(self.feature_df.columns)} 列")
        return self.feature_df

    def run_rules_layer(self):
        """第二层：彭湃规则层"""
        print("\n" + "=" * 60)
        print("  第二层：彭湃规则层")
        print("=" * 60)

        # 双线系统分析
        print("[规则] 双线系统分析...")
        line_rhythm = self.dual_line.analyze_line_rhythm(self.data)
        for line, stats in line_rhythm.items():
            if line != "all":
                print(f"  {line}线: {stats['count']}期, "
                      f"和值均值={stats['sum_stats']['mean']:.1f}")

        # 号码关系网络
        print("[规则] 构建号码关系网络...")
        self.number_graph.build_from_history(self.data, window=200)
        print(f"  网络节点: {len(self.number_graph.graph.nodes())}, "
              f"边: {len(self.number_graph.graph.edges())}")

        # 三态指标
        print("[规则] 计算三态系统指标...")
        self.state_indicators = self.three_states.compute_state_indicators(self.data)
        print(f"  状态指标: {len(self.state_indicators)} 期")

        return line_rhythm

    def run_state_layer(self) -> dict:
        """第三层：状态识别层"""
        print("\n" + "=" * 60)
        print("  第三层：状态识别层")
        print("=" * 60)

        if self.state_indicators.empty:
            self.state_indicators = self.three_states.compute_state_indicators(self.data)

        # HMM
        print("[状态] 训练HMM...")
        self.hmm.fit(self.state_indicators)
        hmm_probs = self.hmm.predict_next(self.state_indicators)
        print(f"  HMM预测: A={hmm_probs['A']:.2%} B={hmm_probs['B']:.2%} C={hmm_probs['C']:.2%}")

        # HSMM（主模型）
        print("[状态] 训练HSMM（主状态模型）...")
        self.hsmm.fit(self.state_indicators)
        hsmm_result = self.hsmm.predict_next(self.state_indicators)
        print(f"  HSMM预测: A={hsmm_result['A']:.2%} B={hsmm_result['B']:.2%} C={hsmm_result['C']:.2%}")
        print(f"  当前状态: {hsmm_result.get('current_state', '?')} "
              f"(已持续{hsmm_result.get('current_duration', '?')}期)")

        # Markov Switching
        print("[状态] 训练Markov Switching...")
        state_sequence = [self.three_states.rule_based_state(row.to_dict())
                          for _, row in self.state_indicators.iterrows()]
        self.ms.fit(self.state_indicators, self.data, state_sequence)

        # 综合状态（以HSMM为主）
        self.current_state = max(["A", "B", "C"],
                                  key=lambda s: hsmm_result.get(s, 0))
        self.state_probs = {k: hsmm_result.get(k, 1/3) for k in ["A", "B", "C"]}

        state_names = {"A": "纠缠热态", "B": "终止冷态", "C": "拓展回补态"}
        print(f"\n[状态] 综合判定: {state_names[self.current_state]} ({self.current_state})")

        return {
            "state": self.current_state,
            "state_probs": self.state_probs,
            "hsmm": hsmm_result,
            "hmm": hmm_probs
        }

    def run_probability_layer(self) -> dict:
        """第四层：概率模型层"""
        print("\n" + "=" * 60)
        print("  第四层：概率模型层")
        print("=" * 60)

        if self.feature_df is None:
            self.build_features()

        # 当前期特征
        current_features = self.fe.build_current_features(self.data)
        model_outputs = {}

        # XGBoost
        print("[概率] 训练XGBoost...")
        self.xgb.fit(self.feature_df)
        xgb_probs = self.xgb.predict_proba(current_features)
        model_outputs["xgboost"] = xgb_probs
        top5 = sorted(xgb_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  XGBoost Top5: {[f'{n}({p:.3f})' for n, p in top5]}")

        # CatBoost
        print("[概率] 训练CatBoost...")
        self.catboost.fit(self.feature_df)
        cat_probs = self.catboost.predict_proba(current_features)
        model_outputs["catboost"] = cat_probs

        # TFT
        print("[概率] 训练TFT...")
        self.tft.fit(self.feature_df, epochs=15)
        tft_probs = self.tft.predict_proba(current_features)
        model_outputs["tft"] = tft_probs

        # GNN
        print("[概率] 训练GNN...")
        self.gnn.fit(self.number_graph.graph)
        gnn_probs = self.gnn.predict_proba()
        model_outputs["gnn"] = gnn_probs

        # Copula
        print("[概率] 拟合Copula...")
        self.copula.fit(self.data, window=200)

        # HSMM状态概率作为模型输入
        model_outputs["hsmm"] = self._state_to_number_probs()
        model_outputs["bayesian"] = self._bayesian_prior()

        # 彭湃规则修正
        rule_bias = self._pengpai_rule_bias()

        # 融合
        print("\n[概率] 多模型融合...")
        self.fused_probs = self.fusion.fuse(
            model_outputs=model_outputs,
            state_probs=self.state_probs,
            rule_bias=rule_bias
        )

        top10 = sorted(self.fused_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  融合Top10: {[f'{n:02d}' for n, _ in top10]}")
        print(f"  融合权重: {self.fusion.get_current_weights()}")

        return self.fused_probs

    def _state_to_number_probs(self) -> dict:
        """将状态概率转换为号码概率（用于融合）"""
        # 使用Markov Switching的状态条件分布
        return self.ms.get_regime_number_probs(self.current_state)

    def _bayesian_prior(self) -> dict:
        """贝叶斯先验：基于长期频率"""
        front_cols = ["front01", "front02", "front03", "front04", "front05"]
        counts = {n: 0 for n in range(1, 36)}
        for _, row in self.data.tail(100).iterrows():
            for c in front_cols:
                counts[int(row[c])] += 1
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def _pengpai_rule_bias(self) -> dict:
        """彭湃规则修正偏置"""
        bias = {n: 1.0 for n in range(1, 36)}
        # 双线偏置：根据当前期号线型，调整对应线的活跃号码
        # 纠缠偏置：关系网络中心性高的号码加权
        for num in range(1, 36):
            entanglement = self.number_graph.get_entanglement_score(num)
            if self.current_state == "A":  # 纠缠热态：高纠缠加权
                bias[num] *= 1 + 0.2 * np.tanh(entanglement * 5)
            elif self.current_state == "C":  # 拓展态：低纠缠（冷号）加权
                bias[num] *= 1 + 0.2 * (1 - np.tanh(entanglement * 5))
        return bias

    def run_optimization_layer(self) -> list:
        """第五层：组合优化层（4组独立优化目标）"""
        print("\n" + "=" * 60)
        print("  第五层：组合优化层")
        print("=" * 60)

        # 历史组合（用于去重）
        front_cols = ["front01", "front02", "front03", "front04", "front05"]
        history_combos = []
        for _, row in self.data.iterrows():
            history_combos.append(tuple(sorted(int(row[c]) for c in front_cols)))

        # 1. 蒙特卡洛采样（大候选池）
        print("[优化] 蒙特卡洛采样...")
        n_sim = min(self.config["optimization"]["monte_carlo"]["n_simulations"], 80000)
        candidates = self.mc.sample(self.fused_probs, n_simulations=n_sim)
        print(f"  生成候选: {len(candidates)} 组")

        # 2. 结构过滤
        print("[优化] 结构过滤...")
        filtered = self.struct_filter.filter(candidates)
        print(f"  通过结构过滤: {len(filtered)} 组 (过滤率: {1 - len(filtered)/len(candidates):.1%})")

        # 3. 风险过滤
        print("[优化] 风险过滤...")
        safe_candidates = self.risk.filter_risky(filtered, history_combos)
        print(f"  通过风险过滤: {len(safe_candidates)} 组")

        if len(safe_candidates) < 200:
            print("[优化] 候选不足，使用结构过滤结果")
            safe_candidates = filtered[:2000] if len(filtered) >= 200 else filtered

        # 4. 为2组分别构建概率偏置并优化
        self.optimized_groups = []

        # --- A组：模型共识组（纯概率最大化）---
        print("[优化] A组 - 模型共识组（纯概率最大化）...")
        group_a = self._optimize_group(
            safe_candidates, self.fused_probs, "A",
            history_combos, exclude_groups=[]
        )
        self.optimized_groups.append(group_a)

        # --- B组：彭湃强化组（规则匹配+关系网络）---
        print("[优化] B组 - 彭湃强化组（规则匹配+纠缠配对）...")
        pengpai_probs = self._pengpai_enhanced_probs()
        group_b = self._optimize_group(
            safe_candidates, pengpai_probs, "B",
            history_combos, exclude_groups=[group_a]
        )
        self.optimized_groups.append(group_b)

        print(f"  优化完成，输出 {len(self.optimized_groups)} 组")

        # 5. 风控验证
        print("[优化] 风控验证...")
        self.risk_report = self.risk.validate(self.optimized_groups, history_combos)
        if self.risk_report["warnings"]:
            for w in self.risk_report["warnings"]:
                print(f"  [预警] {w}")
        else:
            print("  风控通过，无预警")

        return self.optimized_groups

    def _optimize_group(self, candidates: list, probs: dict, group_label: str,
                         history_combos: list, exclude_groups: list = None) -> tuple:
        """
        为单组优化，确保与已选组有足够差异
        返回: (front_tuple, back_tuple, fitness)
        """
        if exclude_groups is None:
            exclude_groups = []

        # 过滤掉与已选组重叠过多的候选（至少差2个号，即最多3个重复）
        if exclude_groups:
            filtered_candidates = []
            for front, back, score in candidates:
                too_similar = False
                for excl_front, _, _ in exclude_groups:
                    overlap = len(set(front) & set(excl_front))
                    if overlap >= 3:  # 最多允许2个重复，确保至少差3个号
                        too_similar = True
                        break
                if not too_similar:
                    filtered_candidates.append((front, back, score))
            if len(filtered_candidates) >= 30:
                candidates = filtered_candidates

        # 用遗传算法优化
        ga_result = self.ga.optimize(
            candidates=candidates,
            front_probs=probs,
            state=self.current_state,
            number_graph=self.number_graph,
            history_combos=history_combos,
            target_count=1
        )

        if ga_result:
            return ga_result[0]

        # 兜底：从候选中选概率最高的
        if candidates:
            best = max(candidates, key=lambda x: x[2])
            return best
        # 终极兜底
        top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        front = tuple(sorted([n for n, _ in top5]))
        back = tuple(sorted(np.random.choice(range(1, 13), 2, replace=False).tolist()))
        return (front, back, 0.0)

    def _pengpai_enhanced_probs(self) -> dict:
        """彭湃强化概率：纠缠+配对+关系网络加权"""
        probs = {}
        for num in range(1, 36):
            base = self.fused_probs.get(num, 1 / 35)
            # 关系网络中心性（纠缠）
            entanglement = self.number_graph.get_entanglement_score(num)
            # 与Top5号码的平均关系强度（配对）
            top5 = sorted(self.fused_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            pair_score = 0
            for top_num, _ in top5:
                if top_num != num:
                    pair_score += self.number_graph.get_relation_score(num, top_num)
            pair_score = pair_score / 5 if top5 else 0
            # 彭湃强化：基础概率 * (1 + 0.3*纠缠 + 0.3*配对)
            enhanced = base * (1 + 0.3 * np.tanh(entanglement * 5) + 0.3 * np.tanh(pair_score * 5))
            probs[num] = enhanced
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def _cold_expansion_probs(self) -> dict:
        """冷态拓展概率：冷号+遗漏大+回补加权"""
        probs = {}
        # 获取当前特征中的遗漏信息
        current_features = self.fe.build_current_features(self.data)
        miss_dict = {}
        for _, row in current_features.iterrows():
            miss_dict[int(row["number"])] = row.get("miss", 5)

        median_prob = np.median(list(self.fused_probs.values()))
        for num in range(1, 36):
            base = self.fused_probs.get(num, 1 / 35)
            miss = miss_dict.get(num, 5)
            # 冷号（低于中位数概率）加权
            is_cold = base < median_prob * 0.8
            # 遗漏大加权
            miss_bonus = min(miss / 15.0, 1.0)
            # 拓展概率：冷号*1.5 + 遗漏加成
            if is_cold:
                enhanced = base * (1.5 + 0.5 * miss_bonus)
            else:
                enhanced = base * (0.7 + 0.3 * miss_bonus)
            probs[num] = enhanced
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def _exploration_probs(self, exclude_front: tuple) -> dict:
        """探索概率：与A组差异最大化+中等概率区域"""
        probs = {}
        exclude_set = set(exclude_front)
        median_prob = np.median(list(self.fused_probs.values()))
        for num in range(1, 36):
            base = self.fused_probs.get(num, 1 / 35)
            if num in exclude_set:
                # A组已有的号码大幅降权
                probs[num] = base * 0.2
            else:
                # 中等概率区域加权（避免太冷也避免太热）
                deviation = abs(base - median_prob) / (median_prob + 1e-8)
                mid_bonus = max(0, 1 - deviation)
                probs[num] = base * (0.8 + 0.6 * mid_bonus)
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def generate_output(self, target_issue: str = None) -> dict:
        """生成最终输出"""
        print("\n" + "=" * 60)
        print("  最终输出")
        print("=" * 60)

        if target_issue is None:
            # 推断下一期号
            latest = self.data["issue"].iloc[-1]
            try:
                year = int(latest[:2])
                seq = int(latest[2:])
                next_seq = seq + 1
                if next_seq > 150:
                    next_seq = 1
                    year += 1
                target_issue = f"{year:02d}{next_seq:03d}"
            except Exception:
                target_issue = "NEXT"

        output = self.output_gen.generate(
            optimized_groups=self.optimized_groups,
            fused_probs=self.fused_probs,
            state=self.current_state,
            state_probs=self.state_probs,
            structure_filter=self.struct_filter,
            risk_report=self.risk_report,
            target_issue=target_issue
        )

        # 打印文本输出
        text = self.output_gen.format_text(output)
        print(text)

        # 保存
        json_path, txt_path = self.output_gen.save(output)
        print(f"\n[输出] 结果已保存:")
        print(f"  JSON: {json_path}")
        print(f"  文本: {txt_path}")

        return output

    def run_full_pipeline(self, target_issue: str = None, use_web: bool = True) -> dict:
        """运行完整流程"""
        print("\n" + "#" * 60)
        print("  PMSF-V1 彭湃大乐透多尺度状态融合系统")
        print(f"  Version: {self.config['system']['version']}")
        print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("#" * 60)

        # 五层流程
        self.load_data(use_web=use_web)
        self.build_features()
        self.run_rules_layer()
        self.run_state_layer()
        self.run_probability_layer()
        self.run_optimization_layer()
        output = self.generate_output(target_issue)

        print("\n" + "#" * 60)
        print("  系统运行完成")
        print("#" * 60)

        return output

    def run_backtest(self, n_test: int = 30) -> dict:
        """运行滚动回测"""
        print("\n" + "#" * 60)
        print("  PMSF-V1 滚动回测")
        print("#" * 60)

        self.load_data(use_web=False)
        if self.data is None or len(self.data) < self.config["backtest"]["train_min_size"]:
            print("[回测] 数据不足")
            return {}

        backtester = WalkForwardBacktester(self.config)
        result = backtester.run(self.data, n_test=n_test, verbose=True)

        print("\n" + "=" * 60)
        print("  回测结果总结")
        print("=" * 60)
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")

        return result


def main():
    parser = argparse.ArgumentParser(description="PMSF-V1 彭湃大乐透多尺度状态融合系统")
    parser.add_argument("--mode", choices=["predict", "backtest"], default="predict",
                        help="运行模式: predict=预测下一期, backtest=滚动回测")
    parser.add_argument("--issue", type=str, default=None, help="目标期号")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--no-web", action="store_true", help="不使用网络抓取数据")
    parser.add_argument("--n-test", type=int, default=30, help="回测期数")

    args = parser.parse_args()

    # 切换到项目目录
    os.chdir(PROJECT_ROOT)

    system = PMSFSystem(config_path=args.config)

    if args.mode == "predict":
        system.run_full_pipeline(target_issue=args.issue, use_web=not args.no_web)
    elif args.mode == "backtest":
        system.run_backtest(n_test=args.n_test)


if __name__ == "__main__":
    main()
