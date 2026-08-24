"""
Copula 模型
研究号码之间的非线性依赖关系
分析不同条件下号码共同出现的结构
输出：号码对的依赖强度矩阵
"""
import numpy as np
import pandas as pd
from scipy import stats


class CopulaModel:
    """Copula非线性依赖模型"""

    def __init__(self, config: dict):
        self.cfg = config["probability_models"]["copula"]
        self.family = self.cfg.get("family", "gaussian")
        self.dependency_matrix = None  # 35x35 依赖矩阵
        self.front_nums = list(range(1, 36))
        self.marginal_params = {}  # 每个号码的边缘分布参数

    def fit(self, df: pd.DataFrame, window: int = None):
        """
        拟合Copula模型
        学习号码之间的非线性依赖结构
        """
        front_cols = ["front01", "front02", "front03", "front04", "front05"]
        if window and len(df) > window:
            df = df.tail(window).reset_index(drop=True)

        n = len(df)
        if n < 20:
            print("[Copula] 数据不足，使用均匀依赖")
            self.dependency_matrix = np.eye(35) * 0.5 + 0.1
            np.fill_diagonal(self.dependency_matrix, 1.0)
            return

        # 构建二元出现矩阵 (n_samples x 35)
        binary_matrix = np.zeros((n, 35))
        for i, row in df.iterrows():
            for c in front_cols:
                num = int(row[c])
                if 1 <= num <= 35:
                    binary_matrix[i, num - 1] = 1

        # 计算每个号码的边缘分布（出现概率）
        for idx, num in enumerate(self.front_nums):
            p = binary_matrix[:, idx].mean()
            self.marginal_params[num] = {"p": float(p)}

        # 计算Gaussian Copula相关矩阵
        # 对二元变量使用四象限相关（phi系数）
        self.dependency_matrix = np.zeros((35, 35))
        for i in range(35):
            for j in range(i + 1, 35):
                x = binary_matrix[:, i]
                y = binary_matrix[:, j]
                # 联合概率
                p_xy = (x * y).mean()
                p_x = x.mean()
                p_y = y.mean()
                # Phi系数（二元变量的Pearson相关）
                denom = np.sqrt(p_x * (1 - p_x) * p_y * (1 - p_y))
                if denom > 1e-8:
                    phi = (p_xy - p_x * p_y) / denom
                else:
                    phi = 0.0
                # 转换为Gaussian Copula的rho
                rho = np.clip(phi, -0.9, 0.9)
                self.dependency_matrix[i, j] = rho
                self.dependency_matrix[j, i] = rho

        np.fill_diagonal(self.dependency_matrix, 1.0)
        # 确保半正定
        self.dependency_matrix = self._make_psd(self.dependency_matrix)

        print(f"[Copula] 拟合完成，平均依赖强度: {np.mean(np.abs(self.dependency_matrix[self.dependency_matrix < 1])):.4f}")

    def _make_psd(self, matrix: np.ndarray) -> np.ndarray:
        """确保矩阵半正定"""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def get_dependency(self, num_a: int, num_b: int) -> float:
        """获取两个号码的依赖强度"""
        if self.dependency_matrix is None:
            return 0.0
        i, j = num_a - 1, num_b - 1
        return float(self.dependency_matrix[i, j])

    def get_conditional_prob(self, num: int, given_nums: list) -> float:
        """
        计算在给定号码已出现的条件下，num出现的条件概率
        基于Gaussian Copula的条件分布
        """
        if self.dependency_matrix is None or not given_nums:
            return self.marginal_params.get(num, {}).get("p", 5 / 35)

        idx = num - 1
        given_indices = [n - 1 for n in given_nums if 1 <= n <= 35 and n != num]

        if not given_indices:
            return self.marginal_params.get(num, {}).get("p", 5 / 35)

        # 简化条件概率：基于平均依赖度调整
        avg_dep = np.mean([self.dependency_matrix[idx, gi] for gi in given_indices])
        base_p = self.marginal_params.get(num, {}).get("p", 5 / 35)

        # 正依赖提升概率，负依赖降低概率
        adjusted = base_p * (1 + 0.3 * avg_dep)
        return float(np.clip(adjusted, 0.001, 0.9))

    def get_pair_dependency_top(self, num: int, top_k: int = 5) -> list:
        """获取与指定号码依赖最强的top_k号码"""
        if self.dependency_matrix is None:
            return []
        idx = num - 1
        deps = [(self.front_nums[j], float(self.dependency_matrix[idx, j]))
                for j in range(35) if j != idx]
        deps.sort(key=lambda x: abs(x[1]), reverse=True)
        return deps[:top_k]

    def apply_copula_bias(self, base_probs: dict, selected_nums: list = None) -> dict:
        """
        将Copula依赖应用到概率上
        如果已选某些号码，调整剩余号码的条件概率
        """
        if selected_nums is None or len(selected_nums) == 0:
            return base_probs

        result = {}
        for num in range(1, 36):
            if num in selected_nums:
                result[num] = base_probs.get(num, 1 / 35)
            else:
                cond_p = self.get_conditional_prob(num, selected_nums)
                base = base_probs.get(num, 1 / 35)
                # 融合基础概率和条件概率
                result[num] = base * 0.6 + cond_p * 0.4

        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result
