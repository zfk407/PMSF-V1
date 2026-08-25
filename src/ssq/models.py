"""
双色球高阶算法模型层 (SsqModels)
在彭湃规则作为先验/状态空间的基础上, 叠加高阶算法模型:
1. MarkovModel     - 马尔可夫链: 号码出现->下期转移概率(同线期序列)
2. GraphModel      - 关系网络: 共现图 + PageRank中心性 + 邻居效应 (GNN近似)
3. TemporalModel   - 时序模型: EWMA指数加权 + 趋势动量
4. MLModel         - XGBoost/CatBoost: 号码特征化二分类(统计+彭湃+时序特征)
5. Fusion          - 贝叶斯融合: 多模型分数加权(滚动回测调权)
彭湃思想始终作为先验约束(双线隔离/恒值配对/过渡号/纠缠/拓展/返点/蓝补红/尾数),
高阶模型在彭湃定义的状态空间内学习, 而非替代它。
"""
import numpy as np
import pandas as pd
from collections import defaultdict

from .rules import SsqPengpaiRules


class MarkovModel:
    """马尔可夫链: 号码状态转移概率
    P(下一期出现 n | 当期出现 i) 用同线期序列估计(拉普拉斯平滑)
    """

    def __init__(self):
        self.rules = SsqPengpaiRules()

    def fit_predict(self, df, target_issue, red_cols=None):
        red_cols = red_cols or ["red01", "red02", "red03", "red04", "red05", "red06"]
        line = self.rules.issue_parity(target_issue)
        line_df = self.rules.split_lines(df)[line]
        seq = []
        for _, row in line_df.iterrows():
            seq.append(set(int(row[c]) for c in red_cols))

        n = len(seq)
        if n < 3:
            return {}

        # 转移计数: 当前期出现 i -> 下一期出现 j
        trans = defaultdict(lambda: defaultdict(float))
        for t in range(n - 1):
            for i in seq[t]:
                for j in range(1, 34):
                    trans[i][j] += 1.0 if j in seq[t + 1] else 0.0

        # 当前期活跃号(最近一期同线号码)作为条件
        latest = seq[-1]
        alpha = 0.3  # 拉普拉斯平滑
        probs = {}
        for j in range(1, 34):
            s = 0.0
            for i in latest:
                total_j = trans[i][j] + alpha
                total_all = sum(trans[i].values()) + alpha * 33
                s += total_j / total_all
            probs[j] = s / len(latest) if latest else 0.0
        return {k: v for k, v in probs.items()}


class GraphModel:
    """关系网络模型 (GNN近似)
    33节点共现网络: 边权=条件共现强度(Lift), 节点分=PageRank中心性*邻居效应
    """

    def __init__(self):
        self.rules = SsqPengpaiRules()

    def fit_predict(self, df, target_issue, window=100, red_cols=None):
        red_cols = red_cols or ["red01", "red02", "red03", "red04", "red05", "red06"]
        line = self.rules.issue_parity(target_issue)
        line_df = self.rules.split_lines(df)[line].tail(window)

        n = len(line_df)
        if n < 5:
            return {}

        # 共现计数 + 单号频率
        co_occur = defaultdict(lambda: defaultdict(float))
        freq = defaultdict(float)
        for _, row in line_df.iterrows():
            nums = sorted(set(int(row[c]) for c in red_cols))
            for i in nums:
                freq[i] += 1
                for j in nums:
                    if j != i:
                        co_occur[i][j] += 1

        # 构建网络: 边权 = Lift (P(i,j)/(P(i)P(j)))
        edges = []
        for i in range(1, 34):
            for j in range(i + 1, 34):
                cij = co_occur[i].get(j, 0)
                if cij == 0:
                    continue
                p_i = freq[i] / n
                p_j = freq[j] / n
                p_ij = cij / n
                lift = p_ij / (p_i * p_j) if p_i * p_j > 0 else 0
                if lift >= 1.0:
                    edges.append((i, j, lift))

        # PageRank 中心性(用numpy幂迭代, 避免networkx依赖开销)
        pr = self._pagerank(edges, n_iter=50, d=0.85)
        if pr is None:
            return {}

        # 邻居效应: 节点分 = PageRank * (1 + 与高频号共现加成)
        top_centers = sorted(pr, key=pr.get, reverse=True)[:6]
        probs = {}
        for node in range(1, 34):
            neighbor_boost = 0.0
            for c in top_centers:
                if c != node:
                    lift = 0.0
                    if node < c:
                        lift = next((e[2] for e in edges if (e[0] == node and e[1] == c)), 0)
                    else:
                        lift = next((e[2] for e in edges if (e[0] == c and e[1] == node)), 0)
                    neighbor_boost += lift
            probs[node] = pr.get(node, 0) * (1 + 0.3 * neighbor_boost)

        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()} if total > 0 else {}

    @staticmethod
    def _pagerank(edges, n_iter=50, d=0.85):
        n_nodes = 33
        if not edges:
            return None
        adj = np.zeros((n_nodes, n_nodes))
        for i, j, w in edges:
            adj[i - 1][j - 1] += w
            adj[j - 1][i - 1] += w
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        M = adj / row_sum
        pr = np.ones(n_nodes) / n_nodes
        for _ in range(n_iter):
            pr = (1 - d) / n_nodes + d * M.T @ pr
        pr = pr / pr.sum()
        return {i + 1: float(pr[i]) for i in range(n_nodes)}


class TemporalModel:
    """时序模型: EWMA指数加权频率 + 趋势动量(近5期 vs 前5期)"""

    def __init__(self):
        self.rules = SsqPengpaiRules()

    def fit_predict(self, df, target_issue, window=60, red_cols=None):
        red_cols = red_cols or ["red01", "red02", "red03", "red04", "red05", "red06"]
        line = self.rules.issue_parity(target_issue)
        line_df = self.rules.split_lines(df)[line].tail(window)

        n = len(line_df)
        if n < 10:
            return {}

        # 每期出现集合
        seq = [set(int(row[c]) for c in red_cols) for _, row in line_df.iterrows()]

        probs = {}
        for num in range(1, 34):
            # EWMA: 越近期权重越大
            alpha = 0.35
            ewma = 0.0
            for t, s in enumerate(seq):
                hit = 1.0 if num in s else 0.0
                ewma = alpha * hit + (1 - alpha) * ewma
            # 趋势动量: 近5期 vs 前5期
            recent5 = sum(1 for s in seq[-5:] if num in s)
            prev5 = sum(1 for s in seq[-10:-5] if num in s)
            momentum = (recent5 - prev5) / 5.0
            probs[num] = max(0.0, ewma + 0.15 * momentum)
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()} if total > 0 else {}


class MLModel:
    """XGBoost / CatBoost 号码概率模型
    特征: 统计(频率/遗漏/EWMA/趋势) + 彭湃(组热度/纠缠/拓展/返点/蓝补红/过渡号/点位周期/尾数)
    标签: 该号本期是否出现
    样本: 同线期历史, 用上一期特征预测本期(避免泄漏), 时间切分验证
    """

    def __init__(self):
        self.rules = SsqPengpaiRules()
        self._xgb = None
        self._cat = None

    def _pengpai_features(self, analysis, num):
        """从一次full_analysis结果提取单号码彭湃特征(共享复用)"""
        g = self.rules.get_group(num)
        feats = []
        feats.append(analysis["group_hotness"].get(g, 1.0))
        feats.append(analysis["entanglement"].get(g, {}).get("lift", 0.0))
        feats.append(analysis["extension"].get(g, 0.0))
        feats.append(analysis["rebound"].get(num, 0.0))
        feats.append(analysis["blue_boost"].get(num, 0.0))
        feats.append(1.0 if num == analysis["main_transition"] else 0.0)
        feats.append(1.0 if num in analysis["sub_transitions"] else 0.0)
        cs = analysis["cycle_status"].get(g, "normal")
        feats.append(1.0 if cs == "hot" else (0.0 if cs == "rest" else 0.5))
        tail = num % 10
        feats.append(min(analysis["tail_law"].get(tail, 100) / 50.0, 2.0))
        return feats

    def _stat_features(self, appears, n):
        """统计特征: 各窗口频率/遗漏/EWMA/趋势"""
        feats = []
        for w in [5, 10, 20, 50]:
            f = sum(appears[-w:]) / min(w, n)
            feats.append(f)
        miss = 0
        for v in reversed(appears):
            if v == 1:
                break
            miss += 1
        feats.append(min(miss, 50) / 50.0)
        alpha = 0.35
        ewma = 0.0
        for v in appears:
            ewma = alpha * v + (1 - alpha) * ewma
        feats.append(ewma)
        r5 = sum(appears[-5:])
        p5 = sum(appears[-10:-5])
        feats.append(max(-1, min(1, (r5 - p5) / 3.0)))
        return feats

    def _build_dataset(self, df, red_cols, min_period=25):
        """构建同线期样本: (特征, 标签)
        每个period只算一次full_analysis(复用), 用上一期数据预测本期(避免泄漏)
        """
        line = self.rules.issue_parity(str(df['issue'].iloc[-1]))
        line_df = self.rules.split_lines(df)[line]
        n = len(line_df)
        if n < min_period + 3:
            return None, None

        rows, labels = [], []
        for period in range(min_period, n):
            prev = line_df.iloc[:period]
            issue = line_df.iloc[period]['issue']
            analysis = self.rules.full_analysis(prev, str(issue))
            # 当前期出现集合(标签)
            cur = set(int(line_df.iloc[period][c]) for c in red_cols)
            # 各号码出现历史(用于统计特征)
            hist = {num: [1 if num in set(int(line_df.iloc[t][c]) for c in red_cols) else 0
                          for t in range(period)] for num in range(1, 34)}
            for num in range(1, 34):
                stat = self._stat_features(hist[num], period)
                pp = self._pengpai_features(analysis, num)
                rows.append(stat + pp)
                labels.append(1 if num in cur else 0)
        return rows, labels

    def fit_predict(self, df, target_issue, red_cols=None):
        red_cols = red_cols or ["red01", "red02", "red03", "red04", "red05", "red06"]
        rows, labels = self._build_dataset(df, red_cols)
        if rows is None or len(rows) < 200 or len(set(labels)) < 2:
            return {}

        X = np.array(rows, dtype=float)
        y = np.array(labels, dtype=int)
        # 时间切分: 前70%训练, 后30%验证
        split = int(len(X) * 0.7)
        X_tr, y_tr = X[:split], y[:split]
        X_va, y_va = X[split:], y[split:]

        results = {}
        # XGBoost
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=80, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', tree_method='hist', verbosity=0)
            model.fit(X_tr, y_tr)
            acc = float(model.score(X_va, y_va))
            results["xgb"] = (model, acc)
        except Exception:
            results["xgb"] = (None, 0.0)

        # CatBoost
        try:
            from catboost import CatBoostClassifier
            model2 = CatBoostClassifier(
                iterations=80, depth=4, learning_rate=0.1,
                verbose=False, allow_writing_files=False)
            model2.fit(X_tr, y_tr)
            acc2 = float(model2.score(X_va, y_va))
            results["cat"] = (model2, acc2)
        except Exception:
            results["cat"] = (None, 0.0)

        # 用最新同线数据预测下一期
        line = self.rules.issue_parity(str(df['issue'].iloc[-1]))
        line_df = self.rules.split_lines(df)[line]
        n = len(line_df)
        analysis = self.rules.full_analysis(line_df, target_issue)
        hist = {num: [1 if num in set(int(line_df.iloc[t][c]) for c in red_cols) else 0
                      for t in range(n)] for num in range(1, 34)}
        probs = {}
        for num in range(1, 34):
            stat = self._stat_features(hist[num], n)
            pp = self._pengpai_features(analysis, num)
            f = stat + pp
            p = 0.0
            cnt = 0
            for key in ("xgb", "cat"):
                model, acc = results[key]
                if model is not None and acc > 0.5:
                    # 用验证准确率做可信度加权
                    pred = float(model.predict_proba(np.array([f]))[0][1])
                    p += pred * acc
                    cnt += acc
            if cnt > 0:
                probs[num] = p / cnt
        return probs


class SsqModelFusion:
    """贝叶斯融合: 彭湃先验概率 × 高阶模型融合修正
    各模型分数归一化后按可信度加权融合
    """

    def __init__(self):
        self.rules = SsqPengpaiRules()
        self.markov = MarkovModel()
        self.graph = GraphModel()
        self.temporal = TemporalModel()
        self.ml = MLModel()

    def run(self, df, target_issue):
        """返回融合后的红球概率及各模型分量"""
        red_cols = ["red01", "red02", "red03", "red04", "red05", "red06"]

        # 高阶模型分量
        markov = self.markov.fit_predict(df, target_issue, red_cols=red_cols)
        graph = self.graph.fit_predict(df, target_issue, red_cols=red_cols)
        temporal = self.temporal.fit_predict(df, target_issue, red_cols=red_cols)
        ml = self.ml.fit_predict(df, target_issue, red_cols=red_cols)

        # 彭湃先验(同线期贝叶斯 × 彭湃规则修正)
        analysis = self.rules.full_analysis(df, target_issue)
        bayes = analysis["bayes_probs"]
        pengpai_prior = self._pengpai_prior(bayes, analysis)

        # 模型分量权重(初始科学设置, 后续可滚动回测调权)
        weights = {
            "pengpai": 0.30,   # 彭湃先验(规则+贝叶斯)
            "markov": 0.20,    # 马尔可夫链
            "graph": 0.15,     # 关系网络
            "temporal": 0.15,  # 时序EWMA
            "ml": 0.20         # XGBoost/CatBoost
        }

        components = {
            "pengpai": self._normalize(pengpai_prior),
            "markov": self._normalize(markov),
            "graph": self._normalize(graph),
            "temporal": self._normalize(temporal),
            "ml": self._normalize(ml)
        }

        fused = {}
        for num in range(1, 34):
            s = 0.0
            for k, w in weights.items():
                s += w * components[k].get(num, 0)
            fused[num] = s

        total = sum(fused.values())
        fused = {k: v / total for k, v in fused.items()}
        return fused, components, weights

    def _pengpai_prior(self, bayes, analysis):
        """彭湃先验: 贝叶斯基础 × 规则修正因子"""
        prior = {}
        for num in range(1, 34):
            base = bayes.get(num, 1 / 33)
            factor = 1.0
            g = self.rules.get_group(num)
            factor *= (0.6 + 0.4 * analysis["group_hotness"].get(g, 1.0))
            if g in analysis["hot_groups"]:
                factor *= 1.25
            factor *= (1 + 1.5 * max(0, analysis["extension"].get(g, 0)))
            factor *= (1 + 0.6 * analysis["rebound"].get(num, 0))
            factor *= (1 + 0.4 * analysis["blue_boost"].get(num, 0))
            if num == analysis["main_transition"]:
                factor *= 1.5
            if num in analysis["sub_transitions"]:
                factor *= 1.3
            cs = analysis["cycle_status"].get(g, "normal")
            if cs == "hot":
                factor *= 1.2
            elif cs == "rest":
                factor *= 0.75
            tail = num % 10
            cold = analysis["tail_law"].get(tail, 100)
            if cold > 15:
                factor *= (1 + 0.2 * min(cold / 30.0, 0.5))
            pair = self.rules.get_pair(num)
            if pair:
                pair_bayes = bayes.get(pair, 1 / 33)
                factor *= (1 + 0.3 * (pair_bayes / (1 / 33) - 1))
            prior[num] = base * factor
        return prior

    @staticmethod
    def _normalize(d):
        if not d:
            return {}
        vmin = min(d.values())
        vmax = max(d.values())
        if vmax - vmin < 1e-9:
            return {k: 1.0 for k in d}
        return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}
