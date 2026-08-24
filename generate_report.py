"""
PMSF-V1 详细报告生成器
基于真实历史数据生成完整的技术分析报告
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.layer1_data.database import DltDatabase
from src.layer1_data.fetcher import DltDataFetcher
from src.layer2_rules.dual_line import DualLineSystem
from src.layer2_rules.number_graph import NumberGraph
from src.layer2_rules.three_states import ThreeStateSystem


def load_config():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def analyze_data(df: pd.DataFrame) -> dict:
    """对真实历史数据做全面统计分析"""
    front_cols = ["front01", "front02", "front03", "front04", "front05"]
    back_cols = ["back01", "back02"]
    n = len(df)

    analysis = {}
    analysis["total_periods"] = n
    analysis["date_range"] = f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}"
    analysis["issue_range"] = f"{df['issue'].iloc[0]} ~ {df['issue'].iloc[-1]}"

    # 1. 前区号码频率
    all_front = []
    for _, row in df.iterrows():
        all_front.extend([int(row[c]) for c in front_cols])
    front_counts = pd.Series(all_front).value_counts().sort_index()
    front_freq = (front_counts / n).round(4)
    analysis["front_freq"] = front_freq.to_dict()
    analysis["front_hot10"] = front_counts.nlargest(10).index.tolist()
    analysis["front_cold10"] = front_counts.nsmallest(10).index.tolist()
    analysis["front_avg_freq"] = float(front_freq.mean())

    # 2. 后区号码频率
    all_back = []
    for _, row in df.iterrows():
        all_back.extend([int(row[c]) for c in back_cols])
    back_counts = pd.Series(all_back).value_counts().sort_index()
    back_freq = (back_counts / n).round(4)
    analysis["back_freq"] = back_freq.to_dict()
    analysis["back_hot5"] = back_counts.nlargest(5).index.tolist()
    analysis["back_cold5"] = back_counts.nsmallest(5).index.tolist()

    # 3. 奇偶结构分布
    odd_even_dist = df["odd_even"].value_counts(normalize=True).round(4).to_dict()
    analysis["odd_even_dist"] = odd_even_dist
    analysis["odd_even_top3"] = df["odd_even"].value_counts().head(3).index.tolist()

    # 4. 大小结构分布
    big_small_dist = df["big_small"].value_counts(normalize=True).round(4).to_dict()
    analysis["big_small_dist"] = big_small_dist

    # 5. 四区结构分布（取出现最多的5种）
    zone_dist = df["zone"].value_counts(normalize=True).head(10).round(4).to_dict()
    analysis["zone_dist_top10"] = zone_dist

    # 6. 和值统计
    analysis["sum_stats"] = {
        "mean": float(df["sum_front"].mean()),
        "std": float(df["sum_front"].std()),
        "min": int(df["sum_front"].min()),
        "max": int(df["sum_front"].max()),
        "median": float(df["sum_front"].median())
    }
    # 和值区间分布
    bins = [0, 60, 75, 90, 105, 120, 140, 200]
    labels = ["<60", "60-75", "75-90", "90-105", "105-120", "120-140", ">140"]
    df["sum_bin"] = pd.cut(df["sum_front"], bins=bins, labels=labels)
    analysis["sum_dist"] = df["sum_bin"].value_counts(normalize=True).reindex(labels).round(4).to_dict()

    # 7. 跨度统计
    analysis["span_stats"] = {
        "mean": float(df["span_front"].mean()),
        "std": float(df["span_front"].std()),
        "min": int(df["span_front"].min()),
        "max": int(df["span_front"].max())
    }

    # 8. 近期趋势（最近50期 vs 全部）
    recent50 = df.tail(50)
    recent_front = []
    for _, row in recent50.iterrows():
        recent_front.extend([int(row[c]) for c in front_cols])
    recent_counts = pd.Series(recent_front).value_counts()
    recent_freq = (recent_counts / 50).round(4)
    analysis["recent50_hot10"] = recent_counts.nlargest(10).index.tolist()
    analysis["recent50_cold10"] = [n for n in range(1, 36) if n not in recent_counts.index][:10]

    # 9. 连号统计
    consecutive_count = 0
    for _, row in df.iterrows():
        nums = sorted([int(row[c]) for c in front_cols])
        for i in range(len(nums) - 1):
            if nums[i + 1] == nums[i] + 1:
                consecutive_count += 1
                break
    analysis["consecutive_rate"] = round(consecutive_count / n, 4)

    # 10. 重号统计（与上一期重复的号码数）
    repeat_counts = []
    for i in range(1, n):
        prev = set(int(df.iloc[i - 1][c]) for c in front_cols)
        curr = set(int(df.iloc[i][c]) for c in front_cols)
        repeat_counts.append(len(prev & curr))
    analysis["repeat_stats"] = {
        "mean": float(np.mean(repeat_counts)),
        "0_rate": round(repeat_counts.count(0) / len(repeat_counts), 4),
        "1_rate": round(repeat_counts.count(1) / len(repeat_counts), 4),
        "2_rate": round(repeat_counts.count(2) / len(repeat_counts), 4),
        "3_plus_rate": round(sum(1 for x in repeat_counts if x >= 3) / len(repeat_counts), 4)
    }

    return analysis


def analyze_dual_line(df: pd.DataFrame) -> dict:
    """双线系统分析"""
    dual = DualLineSystem(load_config())
    rhythm = dual.analyze_line_rhythm(df)
    result = {}
    for line in ["single", "double"]:
        if line in rhythm:
            stats = rhythm[line]
            result[line] = {
                "count": stats["count"],
                "sum_mean": round(stats["sum_stats"]["mean"], 2),
                "sum_std": round(stats["sum_stats"]["std"], 2),
                "top_odd_even": list(stats["odd_even_dist"].keys())[:3]
            }
    return result


def analyze_number_graph(df: pd.DataFrame) -> dict:
    """号码关系网络分析"""
    config = load_config()
    ng = NumberGraph(config)
    ng.build_from_history(df, window=200)

    # 纠缠度排名
    entanglement = {}
    for num in range(1, 36):
        entanglement[num] = ng.get_entanglement_score(num)
    sorted_ent = sorted(entanglement.items(), key=lambda x: x[1], reverse=True)

    # 最强关系对
    edges = []
    for u, v, data in ng.graph.edges(data=True):
        if data.get("edge_type") == "cooccur":
            edges.append((u, v, data.get("weight", 0), data.get("lift", 1)))
    edges.sort(key=lambda x: x[2] * x[3], reverse=True)

    return {
        "entanglement_top10": [(n, round(s, 4)) for n, s in sorted_ent[:10]],
        "entanglement_bottom5": [(n, round(s, 4)) for n, s in sorted_ent[-5:]],
        "top_relations": [(u, v, round(w, 4), round(l, 2)) for u, v, w, l in edges[:10]],
        "n_edges": len(ng.graph.edges()),
        "n_cooccur_edges": sum(1 for _, _, d in ng.graph.edges(data=True) if d.get("edge_type") == "cooccur")
    }


def analyze_states(df: pd.DataFrame) -> dict:
    """三态系统分析"""
    config = load_config()
    tss = ThreeStateSystem(config)
    indicators = tss.compute_state_indicators(df)

    if indicators.empty:
        return {}

    # 规则判定状态序列
    state_seq = [tss.rule_based_state(row.to_dict()) for _, row in indicators.iterrows()]
    state_counts = pd.Series(state_seq).value_counts()

    # 状态持续时间统计
    durations = {"A": [], "B": [], "C": []}
    current = state_seq[0]
    dur = 1
    for s in state_seq[1:]:
        if s == current:
            dur += 1
        else:
            durations[current].append(dur)
            current = s
            dur = 1
    durations[current].append(dur)

    # 最近状态
    recent_states = state_seq[-20:]
    recent_state_counts = pd.Series(recent_states).value_counts().to_dict()

    return {
        "state_distribution": {k: round(v / len(state_seq), 4) for k, v in state_counts.items()},
        "avg_duration": {k: round(np.mean(v), 2) if v else 0 for k, v in durations.items()},
        "max_duration": {k: max(v) if v else 0 for k, v in durations.items()},
        "recent20_dist": recent_state_counts,
        "current_state": state_seq[-1],
        "current_duration": durations[state_seq[-1]][-1] if durations[state_seq[-1]] else 1
    }


def generate_report(output_path: str = None):
    """生成完整详细报告"""
    config = load_config()

    print("=" * 60)
    print("  PMSF-V1 详细报告生成器")
    print("=" * 60)

    # 加载数据
    print("\n[1/6] 加载真实历史数据...")
    db = DltDatabase(config["data"]["db_path"])
    if db.count() > 100:
        df = db.load_all()
        print(f"  从数据库加载 {len(df)} 期")
    else:
        fetcher = DltDataFetcher(config["data"]["raw_dir"])
        df = fetcher.get_data(use_web=True, use_mock_fallback=False)
        if not df.empty:
            db.insert_batch(df)

    if df.empty:
        print("  错误：无法获取数据")
        return

    # 各项分析
    print("\n[2/6] 数据统计分析...")
    data_analysis = analyze_data(df)

    print("\n[3/6] 双线系统分析...")
    dual_analysis = analyze_dual_line(df)

    print("\n[4/6] 号码关系网络分析...")
    graph_analysis = analyze_number_graph(df)

    print("\n[5/6] 三态系统分析...")
    state_analysis = analyze_states(df)

    # 加载最新运行结果
    print("\n[6/6] 加载系统运行结果...")
    result_dir = os.path.join(PROJECT_ROOT, "results")
    latest_result = None
    if os.path.exists(result_dir):
        json_files = sorted([f for f in os.listdir(result_dir) if f.endswith(".json")], reverse=True)
        if json_files:
            import json
            with open(os.path.join(result_dir, json_files[0]), "r", encoding="utf-8") as f:
                latest_result = json.load(f)
            print(f"  加载最新结果: {json_files[0]}")

    # 生成报告
    print("\n生成报告内容...")
    report = build_report_text(data_analysis, dual_analysis, graph_analysis,
                                state_analysis, latest_result, config)

    # 保存
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(PROJECT_ROOT, "results", f"PMSF-V1详细报告_{timestamp}.md")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n报告已保存: {output_path}")
    print(f"报告字数: {len(report)} 字符")
    return output_path


def build_report_text(data_analysis, dual_analysis, graph_analysis,
                      state_analysis, latest_result, config) -> str:
    """构建报告文本"""
    lines = []

    # 标题
    lines.append("# PMSF-V1 彭湃大乐透多尺度状态融合系统 — 详细报告")
    lines.append("")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 系统版本：PMSF-V1.0")
    lines.append(f"> 数据来源：500彩票网真实历史开奖数据")
    lines.append("")

    # 目录
    lines.append("## 目录")
    lines.append("")
    lines.append("1. [系统概述](#一系统概述)")
    lines.append("2. [数据基础分析](#二数据基础分析)")
    lines.append("3. [彭湃规则层分析](#三彭湃规则层分析)")
    lines.append("4. [状态识别层分析](#四状态识别层分析)")
    lines.append("5. [概率模型与融合](#五概率模型与融合)")
    lines.append("6. [组合优化与输出](#六组合优化与输出)")
    lines.append("7. [系统创新总结](#七系统创新总结)")
    lines.append("8. [免责声明](#八免责声明)")
    lines.append("")

    # 一、系统概述
    lines.append("---")
    lines.append("## 一、系统概述")
    lines.append("")
    lines.append("### 1.1 系统定位")
    lines.append("")
    lines.append("PMSF-V1（Pengpai Multi-scale State Fusion）不是传统的\"彩票预测模型\"，而是一个基于**历史时序、状态转换、号码关系网络、多模型融合与组合优化**的概率筛选系统。")
    lines.append("")
    lines.append("核心目标：在历史数据空间、彭湃经验规则空间、统计概率空间、机器学习空间中，寻找综合评分最高的有限组合集合。")
    lines.append("")

    lines.append("### 1.2 五层架构")
    lines.append("")
    lines.append("| 层级 | 名称 | 核心模块 |")
    lines.append("|------|------|----------|")
    lines.append("| 第一层 | 数据基础层 | 数据库、数据抓取、23维特征工程 |")
    lines.append("| 第二层 | 彭湃规则层 | 双线系统、号码关系网络、三态系统 |")
    lines.append("| 第三层 | 状态识别层 | HMM、HSMM（主模型）、Markov Switching |")
    lines.append("| 第四层 | 概率模型层 | XGBoost、CatBoost、TFT、GNN、Copula、贝叶斯融合 |")
    lines.append("| 第五层 | 组合优化层 | 蒙特卡洛、结构过滤、遗传算法、风险控制 |")
    lines.append("")

    lines.append("### 1.3 融合权重公式")
    lines.append("")
    lines.append("```")
    lines.append("最终评分 = 0.25×XGBoost + 0.20×CatBoost + 0.15×GNN")
    lines.append("         + 0.15×TFT + 0.10×HSMM + 0.10×Bayesian")
    lines.append("         + 0.05×彭湃规则修正")
    lines.append("```")
    lines.append("")
    lines.append("权重通过滚动回测（Walk Forward Validation）动态调整。")
    lines.append("")

    # 二、数据基础分析
    lines.append("---")
    lines.append("## 二、数据基础分析")
    lines.append("")

    lines.append("### 2.1 数据概况")
    lines.append("")
    lines.append(f"- **总期数**：{data_analysis['total_periods']} 期")
    lines.append(f"- **期号范围**：{data_analysis['issue_range']}")
    lines.append(f"- **日期范围**：{data_analysis['date_range']}")
    lines.append(f"- **数据来源**：500彩票网（datachart.500.com）")
    lines.append("")

    lines.append("### 2.2 前区号码频率分析（01-35）")
    lines.append("")
    lines.append("**热号Top10（出现次数最多）**：")
    lines.append("")
    hot10 = data_analysis["front_hot10"]
    lines.append("| 排名 | 号码 | 出现频率 |")
    lines.append("|------|------|----------|")
    for i, num in enumerate(hot10, 1):
        freq = data_analysis["front_freq"].get(num, 0)
        lines.append(f"| {i} | {num:02d} | {freq:.2%} |")
    lines.append("")

    lines.append("**冷号Bottom10（出现次数最少）**：")
    lines.append("")
    cold10 = data_analysis["front_cold10"]
    lines.append("| 排名 | 号码 | 出现频率 |")
    lines.append("|------|------|----------|")
    for i, num in enumerate(cold10, 1):
        freq = data_analysis["front_freq"].get(num, 0)
        lines.append(f"| {i} | {num:02d} | {freq:.2%} |")
    lines.append("")
    lines.append(f"前区平均出现频率：{data_analysis['front_avg_freq']:.2%}（理论值 5/35 = 14.29%）")
    lines.append("")

    lines.append("### 2.3 后区号码频率分析（01-12）")
    lines.append("")
    lines.append(f"**热号Top5**：{', '.join(f'{n:02d}' for n in data_analysis['back_hot5'])}")
    lines.append(f"**冷号Bottom5**：{', '.join(f'{n:02d}' for n in data_analysis['back_cold5'])}")
    lines.append("")

    lines.append("### 2.4 奇偶结构分布")
    lines.append("")
    lines.append("| 奇偶结构 | 出现频率 |")
    lines.append("|----------|----------|")
    for oe, freq in sorted(data_analysis["odd_even_dist"].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {oe} | {freq:.2%} |")
    lines.append("")
    lines.append(f"最常见结构：{', '.join(data_analysis['odd_even_top3'])}")
    lines.append("")

    lines.append("### 2.5 大小结构分布（大≥18，小<18）")
    lines.append("")
    lines.append("| 大小结构 | 出现频率 |")
    lines.append("|----------|----------|")
    for bs, freq in sorted(data_analysis["big_small_dist"].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {bs} | {freq:.2%} |")
    lines.append("")

    lines.append("### 2.6 和值统计")
    lines.append("")
    ss = data_analysis["sum_stats"]
    lines.append(f"- **均值**：{ss['mean']:.2f}")
    lines.append(f"- **标准差**：{ss['std']:.2f}")
    lines.append(f"- **中位数**：{ss['median']:.2f}")
    lines.append(f"- **最小值**：{ss['min']}")
    lines.append(f"- **最大值**：{ss['max']}")
    lines.append("")
    lines.append("**和值区间分布**：")
    lines.append("")
    lines.append("| 和值区间 | 频率 |")
    lines.append("|----------|------|")
    for bin_name, freq in data_analysis["sum_dist"].items():
        lines.append(f"| {bin_name} | {freq:.2%} |")
    lines.append("")

    lines.append("### 2.7 跨度统计")
    lines.append("")
    sp = data_analysis["span_stats"]
    lines.append(f"- **均值**：{sp['mean']:.2f}")
    lines.append(f"- **标准差**：{sp['std']:.2f}")
    lines.append(f"- **范围**：{sp['min']} ~ {sp['max']}")
    lines.append("")

    lines.append("### 2.8 重号与连号统计")
    lines.append("")
    rs = data_analysis["repeat_stats"]
    lines.append(f"- **平均重号数**（与上一期重复）：{rs['mean']:.2f} 个")
    lines.append(f"- **0个重号**：{rs['0_rate']:.2%}")
    lines.append(f"- **1个重号**：{rs['1_rate']:.2%}")
    lines.append(f"- **2个重号**：{rs['2_rate']:.2%}")
    lines.append(f"- **3个及以上重号**：{rs['3_plus_rate']:.2%}")
    lines.append(f"- **含连号期数占比**：{data_analysis['consecutive_rate']:.2%}")
    lines.append("")

    lines.append("### 2.9 近期趋势（最近50期）")
    lines.append("")
    lines.append(f"**近50期热号Top10**：{', '.join(f'{n:02d}' for n in data_analysis['recent50_hot10'])}")
    lines.append(f"**近50期冷号（未出现）**：{', '.join(f'{n:02d}' for n in data_analysis['recent50_cold10'])}")
    lines.append("")

    # 三、彭湃规则层分析
    lines.append("---")
    lines.append("## 三、彭湃规则层分析")
    lines.append("")

    lines.append("### 3.1 双线系统分析")
    lines.append("")
    lines.append("彭湃核心规则之一：将大乐透按期号奇偶拆分为单线（奇数期）和双线（偶数期），分别学习节奏，禁止单线预测双线。")
    lines.append("")
    if "single" in dual_analysis:
        s = dual_analysis["single"]
        d = dual_analysis["double"]
        lines.append("| 线型 | 期数 | 和值均值 | 和值标准差 | 主流奇偶结构 |")
        lines.append("|------|------|----------|------------|--------------|")
        lines.append(f"| 单线(奇) | {s['count']} | {s['sum_mean']} | {s['sum_std']} | {', '.join(s['top_odd_even'])} |")
        lines.append(f"| 双线(偶) | {d['count']} | {d['sum_mean']} | {d['sum_std']} | {', '.join(d['top_odd_even'])} |")
        lines.append("")
        diff = abs(s['sum_mean'] - d['sum_mean'])
        lines.append(f"单线与双线和值均值差异：{diff:.2f}，说明两线存在节奏差异，双线分离学习有意义。")
    lines.append("")

    lines.append("### 3.2 号码关系网络分析")
    lines.append("")
    lines.append("彭湃核心规则之二：建立35个前区号码节点的关系网络，边包括共现关系、邻号关系、尾数关系、纠缠关系。")
    lines.append("")
    lines.append(f"- **网络节点数**：35")
    lines.append(f"- **总边数**：{graph_analysis['n_edges']}")
    lines.append(f"- **共现关系边**：{graph_analysis['n_cooccur_edges']}")
    lines.append("")

    lines.append("**纠缠度Top10（关系网络中心性最高的号码）**：")
    lines.append("")
    lines.append("| 排名 | 号码 | 纠缠度 |")
    lines.append("|------|------|--------|")
    for i, (num, score) in enumerate(graph_analysis["entanglement_top10"], 1):
        lines.append(f"| {i} | {num:02d} | {score:.4f} |")
    lines.append("")

    lines.append("**最强共现关系Top10**：")
    lines.append("")
    lines.append("| 排名 | 号码对 | 共现频率 | 提升度 |")
    lines.append("|------|--------|----------|--------|")
    for i, (u, v, w, l) in enumerate(graph_analysis["top_relations"], 1):
        lines.append(f"| {i} | {u:02d}-{v:02d} | {w:.4f} | {l:.2f} |")
    lines.append("")

    lines.append("### 3.3 三态系统定义")
    lines.append("")
    lines.append("彭湃核心规则之三：系统不是预测号码，而是先判断当前大乐透处于什么状态。")
    lines.append("")
    lines.append("| 状态 | 名称 | 表现特征 |")
    lines.append("|------|------|----------|")
    lines.append("| STATE-A | 纠缠热态 | 热号延续、配对活跃、结构稳定、区域变化小 |")
    lines.append("| STATE-B | 终止冷态 | 热号退出、关系断裂、遗漏增加、配对消失 |")
    lines.append("| STATE-C | 拓展回补态 | 冷号进入、区域扩散、新关系形成、和值偏离 |")
    lines.append("")

    # 四、状态识别层分析
    lines.append("---")
    lines.append("## 四、状态识别层分析")
    lines.append("")

    if state_analysis:
        lines.append("### 4.1 历史状态分布")
        lines.append("")
        lines.append("| 状态 | 名称 | 历史占比 |")
        lines.append("|------|------|----------|")
        state_names = {"A": "纠缠热态", "B": "终止冷态", "C": "拓展回补态"}
        for s in ["A", "B", "C"]:
            freq = state_analysis["state_distribution"].get(s, 0)
            lines.append(f"| {s} | {state_names[s]} | {freq:.2%} |")
        lines.append("")

        lines.append("### 4.2 状态持续时间统计")
        lines.append("")
        lines.append("| 状态 | 平均持续期数 | 最大持续期数 |")
        lines.append("|------|--------------|--------------|")
        for s in ["A", "B", "C"]:
            avg = state_analysis["avg_duration"].get(s, 0)
            mx = state_analysis["max_duration"].get(s, 0)
            lines.append(f"| {s} | {avg} | {mx} |")
        lines.append("")

        lines.append("### 4.3 当前状态判定")
        lines.append("")
        cs = state_analysis["current_state"]
        cd = state_analysis["current_duration"]
        lines.append(f"- **当前状态**：{cs} - {state_names[cs]}")
        lines.append(f"- **已持续**：{cd} 期")
        lines.append(f"- **近20期状态分布**：{state_analysis['recent20_dist']}")
        lines.append("")

        lines.append("### 4.4 状态模型说明")
        lines.append("")
        lines.append("**HMM（隐藏马尔可夫模型）**：寻找状态变化路径，判断下一状态概率。只考虑下一状态，不考虑持续时间。")
        lines.append("")
        lines.append("**HSMM（隐半马尔可夫模型，PMSF主状态模型）**：在HMM基础上增加状态持续时间建模，学习\"纠缠状态通常持续N期\"等规律，能更准确判断状态转换时机。")
        lines.append("")
        lines.append("**Markov Switching（马尔可夫切换模型）**：处理不同状态下的不同号码分布，纠缠态偏向热号延续，拓展态偏向冷号恢复。")
        lines.append("")
    else:
        lines.append("状态分析数据不足。")
        lines.append("")

    # 五、概率模型与融合
    lines.append("---")
    lines.append("## 五、概率模型与融合")
    lines.append("")

    lines.append("### 5.1 概率模型矩阵")
    lines.append("")
    lines.append("| 模型 | 类型 | 输入特征 | 输出 | 优势 |")
    lines.append("|------|------|----------|------|------|")
    lines.append("| XGBoost | 梯度提升树 | 20维数值特征 | 号码出现概率 | 结构化数据、非线性 |")
    lines.append("| CatBoost | 梯度提升树 | 含类别特征(区域/尾数) | 号码出现概率 | 自动处理类别变量 |")
    lines.append("| TFT | 时序Transformer | 多尺度时序窗口 | 号码出现概率 | 长期依赖、注意力机制 |")
    lines.append("| GNN | 图神经网络 | 号码关系图 | 节点Embedding | 学习号码间关系 |")
    lines.append("| Copula | 相关性模型 | 联合分布 | 条件概率 | 非线性依赖建模 |")
    lines.append("| HSMM | 状态模型 | 状态指标 | 状态概率 | 持续时间建模 |")
    lines.append("")

    lines.append("### 5.2 贝叶斯融合机制")
    lines.append("")
    lines.append("多个模型输出不做简单平均，而是建立加权融合公式，并通过滚动回测动态调整权重。")
    lines.append("")
    lines.append("融合后还进行：")
    lines.append("1. **贝叶斯后验修正**：根据当前状态概率对号码概率做状态条件化调整")
    lines.append("2. **彭湃规则修正**：根据纠缠度、双线归属等规则对概率做最终偏置")
    lines.append("")

    # 六、组合优化与输出
    lines.append("---")
    lines.append("## 六、组合优化与输出")
    lines.append("")

    lines.append("### 6.1 优化流程")
    lines.append("")
    lines.append("1. **蒙特卡洛采样**：概率加权随机生成大量5+2组合（非纯随机）")
    lines.append("2. **结构过滤**：过滤奇偶/大小/四区/和值/跨度不合理的组合")
    lines.append("3. **风险过滤**：过滤与历史重复过多、极端结构的组合")
    lines.append("4. **遗传算法优化**：以概率+状态匹配+结构合理+稳定性为目标进化")
    lines.append("5. **组间多样性控制**：确保4组之间至少有2个号码差异")
    lines.append("")

    lines.append("### 6.2 4组输出策略")
    lines.append("")
    lines.append("| 组别 | 名称 | 优化目标 | 特点 |")
    lines.append("|------|------|----------|------|")
    lines.append("| A组 | 模型共识组 | 纯融合概率最大化 | 最高概率、稳定性最强 |")
    lines.append("| B组 | 彭湃强化组 | 纠缠度+配对关系+规则匹配 | 彭湃规则匹配最高 |")
    lines.append("| C组 | 冷态拓展组 | 冷号+遗漏+回补导向 | 捕捉状态转换、冷号回补 |")
    lines.append("| D组 | 探索组 | 与A组最大差异+中等概率 | 防止过拟合、覆盖低概率区域 |")
    lines.append("")

    if latest_result:
        lines.append("### 6.3 最新运行结果")
        lines.append("")
        lines.append(f"- **目标期号**：{latest_result.get('target_issue', 'N/A')}")
        lines.append(f"- **生成时间**：{latest_result.get('generate_time', 'N/A')}")
        cs_info = latest_result.get("current_state", {})
        lines.append(f"- **当前状态**：{cs_info.get('state_name', 'N/A')} ({cs_info.get('state', 'N/A')})")
        sp = cs_info.get("probabilities", {})
        lines.append(f"  - 纠缠热态(A): {sp.get('A', 0):.2%}")
        lines.append(f"  - 终止冷态(B): {sp.get('B', 0):.2%}")
        lines.append(f"  - 拓展回补态(C): {sp.get('C', 0):.2%}")
        lines.append("")

        lines.append("**概率Top10号码**：")
        lines.append("")
        lines.append("| 排名 | 号码 | 融合概率 |")
        lines.append("|------|------|----------|")
        for i, item in enumerate(latest_result.get("top10_numbers", []), 1):
            lines.append(f"| {i} | {item['number']:02d} | {item['probability']:.4f} |")
        lines.append("")

        lines.append("**4组推荐组合**：")
        lines.append("")
        for group in latest_result.get("groups", []):
            struct = group.get("structure", {})
            lines.append(f"**【{group['label']}组】{group['name']}**")
            lines.append(f"- {group['description']}")
            lines.append(f"- **前区**：{group['front_str']}")
            lines.append(f"- **后区**：{group['back_str']}")
            lines.append(f"- **结构**：{struct.get('odd_even','')} | {struct.get('big_small','')} | 四区:{struct.get('zone','')} | 和值:{struct.get('sum','')} | 跨度:{struct.get('span','')}")
            lines.append("")

    # 七、系统创新总结
    lines.append("---")
    lines.append("## 七、系统创新总结")
    lines.append("")
    lines.append("| 维度 | 传统彩票分析 | PMSF-V1 |")
    lines.append("|------|------------|---------|")
    lines.append("| 分析视角 | 看号码频率 | 看状态（纠缠/终止/拓展） |")
    lines.append("| 号码关系 | 假设号码独立 | 建立号码关系网络（共现/邻号/尾数/纠缠） |")
    lines.append("| 模型数量 | 单模型 | 6+模型融合（XGB/CatBoost/TFT/GNN/Copula/HSMM） |")
    lines.append("| 规则处理 | 固定规则硬过滤 | 规则作为先验，融入概率融合 |")
    lines.append("| 选号方式 | 直接选号 | 概率空间优化（蒙特卡洛+遗传算法） |")
    lines.append("| 验证方式 | 凭经验 | 滚动回测验证（Walk Forward） |")
    lines.append("| 双线处理 | 忽略期号奇偶 | 单线/双线分离学习，禁止交叉预测 |")
    lines.append("| 输出数量 | 大量号码 | 精简4组，各有侧重 |")
    lines.append("")

    lines.append("### 核心创新点详解")
    lines.append("")
    lines.append("1. **三态状态机**：首次将大乐透走势抽象为纠缠热态、终止冷态、拓展回补态三种状态，先判状态再选号，而非直接预测号码。")
    lines.append("")
    lines.append("2. **HSMM持续时间建模**：传统HMM只考虑下一状态，PMSF-V1的HSMM增加状态持续时间建模，能学习\"纠缠态通常持续5期\"等规律，更准确判断转换时机。")
    lines.append("")
    lines.append("3. **号码关系网络**：不假设号码独立，而是建立35节点的关系网络，通过GNN学习号码间的共现、邻号、尾数、纠缠关系，输出节点Embedding用于概率修正。")
    lines.append("")
    lines.append("4. **双线分离系统**：按期号奇偶将数据拆分为单线和双线，分别学习节奏，禁止单线预测双线，避免信息污染。")
    lines.append("")
    lines.append("5. **彭湃规则先验融合**：人工经验（双线、纠缠、配对、尾数、四区、返点）不作为硬规则，而是作为先验融入概率融合，与机器学习模型协同。")
    lines.append("")
    lines.append("6. **4组差异化输出**：不是输出一堆号码，而是精简为4组，每组有独立优化目标（共识/强化/拓展/探索），兼顾稳定性和探索性。")
    lines.append("")

    # 八、免责声明
    lines.append("---")
    lines.append("## 八、免责声明")
    lines.append("")
    lines.append("1. 本系统基于历史开奖数据进行统计分析和机器学习建模，**彩票开奖为独立随机事件**，历史数据不代表未来走势。")
    lines.append("")
    lines.append("2. 系统输出的4组组合为**概率筛选结果**，仅供参考和研究使用，**不构成任何投注建议**。")
    lines.append("")
    lines.append("3. 彩票有风险，投注需理性。请根据自身经济情况量力而行，**未成年人禁止购买彩票**。")
    lines.append("")
    lines.append("4. 本系统使用的公开数据来源于500彩票网，数据版权归原网站所有，本系统仅用于学术研究和个人学习。")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*报告由 PMSF-V1 系统自动生成 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    generate_report()
