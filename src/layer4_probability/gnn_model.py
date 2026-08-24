"""
GNN 图神经网络模型
学习号码之间的关系，输出节点Embedding和关系评分
基于号码关系网络图，使用图卷积网络学习节点表示
"""
import numpy as np
import pandas as pd
import networkx as nx

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True

    class SimpleGCN(nn.Module):
        """简单图卷积网络"""

        def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        def forward(self, x, adj):
            for layer in self.layers[:-1]:
                x = torch.mm(adj, x)
                x = F.relu(layer(x))
            x = torch.mm(adj, x)
            x = self.layers[-1](x)
            return x

except ImportError:
    HAS_TORCH = False
    SimpleGCN = None


class GNNModel:
    """GNN号码关系模型"""

    def __init__(self, config: dict):
        self.cfg = config["probability_models"]["gnn"]
        self.hidden_dim = self.cfg.get("hidden_dim", 64)
        self.num_layers = self.cfg.get("num_layers", 3)
        self.model = None
        self.node_embeddings = None
        self.front_nums = list(range(1, 36))

    def fit(self, graph: nx.Graph, number_features: dict = None, epochs: int = 30):
        """
        训练GNN模型
        graph: 号码关系网络图
        number_features: {number: feature_vector} 可选的节点特征
        """
        if graph is None or len(graph.nodes()) == 0:
            print("[GNN] 图为空，使用谱嵌入")
            self._spectral_embedding(graph)
            return

        if not HAS_TORCH:
            print("[GNN] 未安装torch，使用谱嵌入")
            self._spectral_embedding(graph)
            return

        try:
            # 构建邻接矩阵和节点特征
            n_nodes = len(self.front_nums)
            adj = nx.to_numpy_array(graph, nodelist=self.front_nums, weight="weight")
            adj = adj + np.eye(n_nodes)  # 自环
            # 归一化
            degree = adj.sum(axis=1, keepdims=True)
            degree[degree == 0] = 1
            adj_norm = adj / degree

            # 节点特征
            if number_features:
                feat_dim = len(next(iter(number_features.values())))
                X = np.zeros((n_nodes, feat_dim))
                for i, num in enumerate(self.front_nums):
                    if num in number_features:
                        X[i] = number_features[num]
            else:
                # 用度和属性作为初始特征
                X = np.zeros((n_nodes, 5))
                for i, num in enumerate(self.front_nums):
                    X[i, 0] = adj[i].sum()  # 加权度
                    X[i, 1] = num % 10  # 尾数
                    X[i, 2] = 1 if num % 2 == 1 else 0  # 奇偶
                    X[i, 3] = 1 if num >= 18 else 0  # 大小
                    if num <= 9:
                        X[i, 4] = 1
                    elif num <= 18:
                        X[i, 4] = 2
                    elif num <= 27:
                        X[i, 4] = 3
                    else:
                        X[i, 4] = 4

            X_t = torch.FloatTensor(X)
            adj_t = torch.FloatTensor(adj_norm)

            self.model = SimpleGCN(X.shape[1], self.hidden_dim, self.num_layers)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

            # 自监督训练：重构邻接矩阵
            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                embeddings = self.model(X_t, adj_t)
                # 重构损失：内积近似邻接
                recon = torch.mm(embeddings, embeddings.t())
                loss = F.mse_loss(recon, adj_t)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                self.node_embeddings = self.model(X_t, adj_t).numpy()

            print(f"[GNN] 训练完成，嵌入维度: {self.hidden_dim}, 重构loss: {loss.item():.4f}")
        except Exception as e:
            print(f"[GNN] 训练失败: {e}，使用谱嵌入")
            self._spectral_embedding(graph)

    def _spectral_embedding(self, graph: nx.Graph):
        """谱嵌入兜底"""
        n = len(self.front_nums)
        if graph and len(graph.edges()) > 0:
            try:
                adj = nx.to_numpy_array(graph, nodelist=self.front_nums, weight="weight")
                degree = np.diag(adj.sum(axis=1) + 1e-8)
                laplacian = np.eye(n) - np.linalg.inv(degree) @ adj
                _, eigenvectors = np.linalg.eigh(laplacian)
                self.node_embeddings = eigenvectors[:, 1:self.hidden_dim + 1]
                return
            except Exception:
                pass
        self.node_embeddings = np.random.randn(n, self.hidden_dim) * 0.01

    def get_embedding(self, num: int) -> np.ndarray:
        """获取号码的节点嵌入"""
        if self.node_embeddings is None:
            return np.zeros(self.hidden_dim)
        idx = self.front_nums.index(num) if num in self.front_nums else 0
        return self.node_embeddings[idx]

    def predict_proba(self, number_features: dict = None) -> dict:
        """
        基于GNN嵌入预测号码概率
        使用嵌入的范数（中心性）作为概率基础
        """
        if self.node_embeddings is None:
            return {num: 1 / 35 for num in range(1, 36)}

        result = {}
        for i, num in enumerate(self.front_nums):
            emb = self.node_embeddings[i]
            # 嵌入范数作为活跃度评分
            score = float(np.linalg.norm(emb))
            result[num] = max(score, 1e-6)

        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result

    def get_relation_score(self, num_a: int, num_b: int) -> float:
        """基于嵌入相似度计算关系评分"""
        if self.node_embeddings is None:
            return 0.0
        emb_a = self.get_embedding(num_a)
        emb_b = self.get_embedding(num_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
