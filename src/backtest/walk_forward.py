"""
Walk Forward Validation 滚动回测系统
训练 07001-10000 -> 预测 10001 -> 加入10001 -> 预测10002 -> 循环
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..layer1_data.features import FeatureEngineer
from ..layer2_rules.three_states import ThreeStateSystem
from ..layer3_state.hsmm_model import HSMMStateModel
from ..layer4_probability.xgb_model import XGBoostModel
from ..layer4_probability.fusion import ProbabilityFusion
from ..layer5_optimization.monte_carlo import MonteCarloSampler
from ..layer5_optimization.structure_filter import StructureFilter
from ..layer5_optimization.genetic_algorithm import GeneticOptimizer
from .metrics import BacktestMetrics


class WalkForwardBacktester:
    """滚动回测器"""

    def __init__(self, config: dict):
        self.cfg = config
        self.train_min_size = config["backtest"]["train_min_size"]
        self.step_size = config["backtest"]["step_size"]
        self.metrics = BacktestMetrics(config)

    def run(self, df: pd.DataFrame, n_test: int = 50, verbose: bool = True) -> dict:
        """
        运行滚动回测
        df: 全部历史数据
        n_test: 回测期数
        返回: 回测总结
        """
        if len(df) < self.train_min_size + n_test:
            n_test = max(0, len(df) - self.train_min_size)
            if n_test <= 0:
                return {"error": "数据不足，无法回测"}

        total = len(df)
        start_idx = total - n_test
        front_cols = ["front01", "front02", "front03", "front04", "front05"]

        iterator = range(start_idx, total)
        if verbose:
            iterator = tqdm(iterator, desc="滚动回测")

        for test_idx in iterator:
            # 训练集：0 ~ test_idx-1
            train_df = df.iloc[:test_idx].reset_index(drop=True)
            test_row = df.iloc[test_idx]
            issue = test_row["issue"]
            actual_front = [int(test_row[c]) for c in front_cols]

            try:
                # 简化回测：只跑核心流程（特征+XGB+融合+采样）
                result = self._single_predict(train_df)
                if result:
                    predicted_ranked = result["ranked_numbers"]
                    predicted_state = result.get("state")

                    # 记录指标
                    # 修复: 用Top-K作为预测集合, 避免35全集导致coverage/exact_hit恒为100%/5
                    top_k = self.cfg["backtest"].get("top_k", 10)
                    predicted_set = list(predicted_ranked)[:top_k]
                    self.metrics.record(
                        issue=issue,
                        predicted_numbers=predicted_set,
                        actual_numbers=actual_front,
                        predicted_state=predicted_state,
                        actual_state=None,
                        group_label="ALL"
                    )
            except Exception as e:
                if verbose:
                    print(f"[回测] {issue} 预测失败: {e}")
                continue

        return self.metrics.summary()

    def _single_predict(self, train_df: pd.DataFrame) -> dict:
        """单次预测（简化版，用于回测加速）"""
        # 1. 特征工程
        fe = FeatureEngineer(self.cfg)
        feature_df = fe.build_dataset(train_df)
        if feature_df.empty:
            return None

        # 2. 状态识别
        tss = ThreeStateSystem(self.cfg)
        indicators = tss.compute_state_indicators(train_df)
        if indicators.empty:
            state = "C"
            state_probs = {"A": 1/3, "B": 1/3, "C": 1/3}
        else:
            hsmm = HSMMStateModel(self.cfg)
            hsmm.fit(indicators)
            state_result = hsmm.predict_next(indicators)
            state = max(["A", "B", "C"], key=lambda s: state_result.get(s, 0))
            state_probs = {k: state_result.get(k, 1/3) for k in ["A", "B", "C"]}

        # 3. XGBoost预测
        xgb = XGBoostModel(self.cfg)
        xgb.fit(feature_df)

        # 当前特征
        current_features = fe.build_current_features(train_df)
        xgb_probs = xgb.predict_proba(current_features)

        # 4. 融合（简化：只用XGB + 状态偏置）
        fusion = ProbabilityFusion(self.cfg)
        fused = fusion.fuse(
            model_outputs={"xgboost": xgb_probs},
            state_probs=state_probs
        )

        # 5. 排序
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        ranked_numbers = [num for num, _ in ranked]

        return {
            "ranked_numbers": ranked_numbers,
            "fused_probs": fused,
            "state": state,
            "state_probs": state_probs
        }
