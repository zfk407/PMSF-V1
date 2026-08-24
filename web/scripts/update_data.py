"""
PMSF-V1 Web 数据更新脚本
功能：
1. 从500彩票网抓取最新开奖数据
2. 对比上期推荐的4组进行复盘分析（命中数、奖项等级）
3. 重新运行五层系统，预测下一期
4. 生成所有前端所需JSON数据文件
5. 更新运行日志

用法：python scripts/update_data.py
可配合 GitHub Actions 定时执行（大乐透开奖后：周一/三/六 21:00后）
"""
import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

# 项目根目录（web的上一级）
WEB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(WEB_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.layer1_data.database import DltDatabase
from src.layer1_data.fetcher import DltDataFetcher
from src.layer2_rules.dual_line import DualLineSystem
from src.layer2_rules.number_graph import NumberGraph
from src.layer2_rules.three_states import ThreeStateSystem


def load_config():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(data, filename):
    """保存JSON到web/data目录"""
    path = os.path.join(WEB_DIR, "data", filename)
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [保存] {filename} ({len(json.dumps(data, ensure_ascii=False))} 字符)")


def calc_prize(front_hit, back_hit):
    """计算大乐透奖项等级"""
    if front_hit == 5 and back_hit == 2: return {"level": 1, "name": "一等奖"}
    if front_hit == 5 and back_hit == 1: return {"level": 2, "name": "二等奖"}
    if front_hit == 5 and back_hit == 0: return {"level": 3, "name": "三等奖"}
    if front_hit == 4 and back_hit == 2: return {"level": 4, "name": "四等奖"}
    if front_hit == 4 and back_hit == 1: return {"level": 5, "name": "五等奖"}
    if front_hit == 3 and back_hit == 2: return {"level": 6, "name": "六等奖"}
    if front_hit == 4 and back_hit == 0: return {"level": 7, "name": "七等奖"}
    if front_hit == 3 and back_hit == 1: return {"level": 8, "name": "八等奖"}
    if front_hit == 2 and back_hit == 2: return {"level": 9, "name": "九等奖"}
    return {"level": 0, "name": "未中奖"}


def generate_history_json(df):
    """生成历史开奖数据JSON"""
    records = []
    for _, row in df.iterrows():
        records.append({
            "issue": str(row["issue"]),
            "date": str(row.get("date", "")),
            "front01": int(row["front01"]),
            "front02": int(row["front02"]),
            "front03": int(row["front03"]),
            "front04": int(row["front04"]),
            "front05": int(row["front05"]),
            "back01": int(row["back01"]),
            "back02": int(row["back02"]),
            "sum_front": int(row.get("sum_front", 0)),
            "span_front": int(row.get("span_front", 0)),
            "odd_even": str(row.get("odd_even", "")),
            "big_small": str(row.get("big_small", "")),
            "zone": str(row.get("zone", "")),
            "tail": str(row.get("tail", ""))
        })
    return records


def generate_stats_json(df, config):
    """生成统计分析JSON"""
    front_cols = ["front01", "front02", "front03", "front04", "front05"]
    back_cols = ["back01", "back02"]
    n = len(df)

    # 前区频率
    all_front = []
    for _, row in df.iterrows():
        all_front.extend([int(row[c]) for c in front_cols])
    front_counts = pd.Series(all_front).value_counts().sort_index()
    front_freq = (front_counts / n).round(4).to_dict()

    # 后区频率
    all_back = []
    for _, row in df.iterrows():
        all_back.extend([int(row[c]) for c in back_cols])
    back_counts = pd.Series(all_back).value_counts().sort_index()
    back_freq = (back_counts / n).round(4).to_dict()

    # 奇偶分布
    odd_even_dist = df["odd_even"].value_counts(normalize=True).round(4).to_dict()
    # 大小分布
    big_small_dist = df["big_small"].value_counts(normalize=True).round(4).to_dict()
    # 四区分布Top10
    zone_dist = df["zone"].value_counts(normalize=True).head(10).round(4).to_dict()

    # 和值统计
    sum_stats = {
        "mean": round(float(df["sum_front"].mean()), 2),
        "std": round(float(df["sum_front"].std()), 2),
        "min": int(df["sum_front"].min()),
        "max": int(df["sum_front"].max()),
        "median": round(float(df["sum_front"].median()), 2)
    }
    # 和值区间分布
    bins = [0, 60, 75, 90, 105, 120, 140, 200]
    labels = ["<60", "60-75", "75-90", "90-105", "105-120", "120-140", ">140"]
    df["sum_bin"] = pd.cut(df["sum_front"], bins=bins, labels=labels)
    sum_dist = df["sum_bin"].value_counts(normalize=True).reindex(labels).round(4).to_dict()
    sum_dist = {k: float(v) if pd.notna(v) else 0 for k, v in sum_dist.items()}

    # 号码关系网络
    ng = NumberGraph(config)
    ng.build_from_history(df, window=200)
    entanglement = {}
    for num in range(1, 36):
        entanglement[num] = round(ng.get_entanglement_score(num), 4)
    top_entanglement = sorted(entanglement.items(), key=lambda x: x[1], reverse=True)[:10]

    # 最强共现关系
    edges = []
    for u, v, data in ng.graph.edges(data=True):
        if data.get("edge_type") == "cooccur":
            edges.append([int(u), int(v), round(float(data.get("weight", 0)), 4), round(float(data.get("lift", 1)), 2)])
    edges.sort(key=lambda x: x[2] * x[3], reverse=True)
    top_relations = edges[:20]

    # 融合权重
    fusion_weights = config["probability_models"]["fusion_weights"]

    return {
        "total_periods": n,
        "date_range": f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}",
        "issue_range": f"{df['issue'].iloc[0]} ~ {df['issue'].iloc[-1]}",
        "front_freq": {int(k): float(v) for k, v in front_freq.items()},
        "back_freq": {int(k): float(v) for k, v in back_freq.items()},
        "front_hot10": [int(x) for x in front_counts.nlargest(10).index.tolist()],
        "front_cold10": [int(x) for x in front_counts.nsmallest(10).index.tolist()],
        "odd_even_dist": odd_even_dist,
        "big_small_dist": big_small_dist,
        "zone_dist_top10": zone_dist,
        "sum_stats": sum_stats,
        "sum_dist": sum_dist,
        "top_entanglement": top_entanglement,
        "top_relations": top_relations,
        "fusion_weights": fusion_weights
    }


def generate_predictions_json(db):
    """从数据库加载历史预测记录"""
    conn = db._get_conn()
    try:
        df = pd.read_sql("SELECT * FROM backtest_results ORDER BY issue DESC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()

    records = []
    for _, row in df.iterrows():
        pred_front = [int(x) for x in str(row["predicted_front"]).split(",")] if pd.notna(row["predicted_front"]) else []
        pred_back = [int(x) for x in str(row["predicted_back"]).split(",")] if pd.notna(row["predicted_back"]) else []
        actual_front = [int(x) for x in str(row["actual_front"]).split(",")] if pd.notna(row["actual_front"]) else []
        actual_back = [int(x) for x in str(row["actual_back"]).split(",")] if pd.notna(row["actual_back"]) else []
        prize = calc_prize(int(row.get("front_hit", 0)), int(row.get("back_hit", 0)))
        records.append({
            "issue": str(row["issue"]),
            "group": str(row.get("group_label", "")),
            "predicted_front": pred_front,
            "predicted_back": pred_back,
            "actual_front": actual_front,
            "actual_back": actual_back,
            "front_hit": int(row.get("front_hit", 0)),
            "back_hit": int(row.get("back_hit", 0)),
            "prize_level": prize["level"],
            "prize_name": prize["name"]
        })
    return records


def generate_runtime_json(logs, last_issue=None):
    """生成运行时信息"""
    return {
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_issue": last_issue,
        "logs": logs[-50:] if logs else []
    }


def run_update():
    """执行完整更新流程"""
    print("=" * 60)
    print("  PMSF-V1 Web 数据更新")
    print("=" * 60)

    config = load_config()
    logs = []

    def log(action, detail, status="success"):
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "detail": detail,
            "status": status
        }
        logs.append(entry)
        print(f"  [{status.upper()}] {action}: {detail}")

    # 1. 加载/抓取数据
    print("\n[1/6] 数据获取...")
    db = DltDatabase(config["data"]["db_path"])
    fetcher = DltDataFetcher(config["data"]["raw_dir"])

    old_count = db.count()
    df = fetcher.get_data(use_web=True, use_mock_fallback=False)

    if df.empty:
        log("数据获取", "无法获取真实数据，使用数据库现有数据", "warning")
        df = db.load_all()
    else:
        db.insert_batch(df)
        new_count = db.count()
        log("数据获取", f"获取 {len(df)} 期数据（新增 {new_count - old_count} 期）")

    if df.empty:
        print("错误：无数据可用")
        return

    latest_issue = str(df["issue"].iloc[-1])

    # 2. 生成历史数据JSON
    print("\n[2/6] 生成历史数据JSON...")
    history_json = generate_history_json(df)
    save_json(history_json, "history.json")
    log("历史数据", f"生成 {len(history_json)} 期历史记录")

    # 3. 生成统计分析JSON
    print("\n[3/6] 生成统计分析JSON...")
    stats_json = generate_stats_json(df, config)
    save_json(stats_json, "stats.json")
    log("统计分析", "生成号码频率、分布、关系网络等统计")

    # 4. 运行系统预测
    print("\n[4/6] 运行PMSF系统预测...")
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from main import PMSFSystem
        system = PMSFSystem(os.path.join(PROJECT_ROOT, "config.yaml"))
        system.data = df
        system.build_features()
        system.run_rules_layer()
        system.run_state_layer()
        system.run_probability_layer()
        system.run_optimization_layer()
        output = system.generate_output()
        save_json(output, "latest_prediction.json")
        log("系统预测", f"生成 {output.get('target_issue', 'N/A')} 期4组推荐")
    except Exception as e:
        log("系统预测", f"预测失败: {e}", "error")
        import traceback
        traceback.print_exc()

    # 5. 生成预测记录JSON
    print("\n[5/6] 生成预测记录JSON...")
    predictions_json = generate_predictions_json(db)
    save_json(predictions_json, "predictions.json")
    log("预测记录", f"生成 {len(predictions_json)} 条预测复盘记录")

    # 6. 生成运行时信息JSON
    print("\n[6/6] 生成运行时信息...")
    runtime_json = generate_runtime_json(logs, latest_issue)
    save_json(runtime_json, "runtime.json")
    log("运行时", "更新运行日志和状态")

    print("\n" + "=" * 60)
    print("  数据更新完成！")
    print(f"  最新期号: {latest_issue}")
    print(f"  JSON文件保存在: {os.path.join(WEB_DIR, 'data')}")
    print("=" * 60)


if __name__ == "__main__":
    run_update()
