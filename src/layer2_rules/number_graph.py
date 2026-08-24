"""
号码关系网络（彭湃核心规则之二）
建立35个前区号码节点，边包括：共现关系、邻号关系、尾数关系、纠缠关系
输出节点Embedding和关系评分
"""
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict


class NumberGraph:
    """号码关系网络图"""

    def __init__(self, config: dict):
        self.cfg = config
        self.front_nums = list(range(1, 36))
        self.graph = nx.Graph()
        self._init_nodes()

    def _init_nodes(self):
        """初始化35个号码节点"""
        for num in self.front_nums:
            self.graph.add_node(num,
                                tail=num % 10,
                                zone=self._get_zone(num),
                                is_odd=num % 2 == 1)

    @staticmethod
    def _get_zone(num: int) -> int:
        if num <= 9:
            return 1
        elif num <= 18:
            return 2
        elif num <= 27:
            return 3
        else:
            return 4

    def build_from_history(self, df: pd.DataFrame, window: int = None):
        """
        从历史数据构建关系网络
        window: 只使用最近N期，None表示全部
        """
        front_cols = ["front01", "front02", "front03", "front04", "front05"]
        if window and len(df) > window:
            df = df.tail(window).reset_index(drop=True)

        # 共现计数
        cooccur = defaultdict(int)
        appear_count = defaultdict(int)
        for _, row in df.iterrows():
            nums = sorted([int(row[c]) for c in front_cols])
            for n in nums:
                appear_count[n] += 1
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    cooccur[(nums[i], nums[j])] += 1

        total = len(df)
        # 清空旧边
        self.graph.remove_edges_from(list(self.graph.edges()))

        # 添加共现边
        for (a, b), cnt in cooccur.items():
            # 归一化共现强度（Jaccard-like）
            pa = appear_count[a] / total if total else 0
            pb = appear_count[b] / total if total else 0
            pco = cnt / total if total else 0
            # 提升度 lift = P(A,B) / (P(A)*P(B))
            lift = pco / (pa * pb) if (pa * pb) > 0 else 1.0
            self.graph.add_edge(a, b,
                                 weight=float(pco),
                                 lift=float(lift),
                                 count=cnt,
                                 edge_type="cooccur")

        # 添加邻号关系边（固定结构）
        for num in range(1, 35):
            if not self.graph.has_edge(num, num + 1):
                self.graph.add_edge(num, num + 1, weight=0.01,
                                     lift=1.0, count=0, edge_type="neighbor")

        # 添加尾数关系边
        for tail in range(10):
            tail_nums = [n for n in self.front_nums if n % 10 == tail]
            for i in range(len(tail_nums)):
                for j in range(i + 1, len(tail_nums)):
                    a, b = tail_nums[i], tail_nums[j]
                    if not self.graph.has_edge(a, b):
                        self.graph.add_edge(a, b, weight=0.005,
                                             lift=1.0, count=0, edge_type="tail")

        return self.graph

    def get_node_embedding(self, dimensions: int = 8) -> dict:
        """
        生成节点Embedding
        基于图结构的谱特征 + 节点属性
        """
        if len(self.graph.edges()) == 0:
            return {n: np.zeros(dimensions) for n in self.front_nums}

        try:
            # 使用谱聚类风格的嵌入
            adj = nx.to_numpy_array(self.graph, nodelist=self.front_nums, weight="weight")
            # 度矩阵
            degree = np.diag(adj.sum(axis=1) + 1e-8)
            # 归一化拉普拉斯
            laplacian = np.eye(len(self.front_nums)) - np.linalg.inv(degree) @ adj
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            # 取前 dimensions 个非平凡特征向量
            embedding = eigenvectors[:, 1:dimensions + 1]
        except Exception:
            embedding = np.random.randn(len(self.front_nums), dimensions) * 0.01

        result = {}
        for i, num in enumerate(self.front_nums):
            result[num] = embedding[i]
        return result

    def get_relation_score(self, num_a: int, num_b: int) -> float:
        """获取两个号码的关系评分（综合权重）"""
        if self.graph.has_edge(num_a, num_b):
            edge = self.graph.edges[num_a, num_b]
            return float(edge.get("weight", 0) * edge.get("lift", 1.0))
        return 0.0

    def get_entanglement_score(self, num: int) -> float:
        """
        纠缠评分：号码在网络中的中心性
        度数越高、加权度越高，纠缠越强
        """
        if num not in self.graph:
            return 0.0
        # 加权度
        weighted_degree = sum(
            d.get("weight", 0) * d.get("lift", 1.0)
            for _, _, d in self.graph.edges(num, data=True)
        )
        return float(weighted_degree)

    def get_top_related(self, num: int, top_k: int = 5) -> list:
        """获取与指定号码关系最强的top_k号码"""
        if num not in self.graph:
            return []
        neighbors = []
        for neighbor in self.graph.neighbors(num):
            score = self.get_relation_score(num, neighbor)
            neighbors.append((neighbor, score))
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_k]

    def analyze_communities(self) -> dict:
        """
        社区发现：识别号码抱团群组
        """
        if len(self.graph.edges()) == 0:
            return {}
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(self.graph, weight="weight"))
            result = {}
            for i, comm in enumerate(communities):
                for num in comm:
                    result[num] = i
            return result
        except Exception:
            return {n: 0 for n in self.front_nums}
