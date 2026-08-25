"""
双色球预测主入口 (SsqPengpai)
运行流程：数据获取 -> 彭湃规则分析 -> 状态判定 -> 概率融合 -> 组合优化 -> 输出2组
"""
import os
import sys
import json
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.ssq.data import SsqDataFetcher
from src.ssq.database import SsqDatabase
from src.ssq.analyzer import SsqAnalyzer


def calc_ssq_prize(red_hit: int, blue_hit: int):
    """双色球奖项计算"""
    if red_hit == 6 and blue_hit == 1: return 1, "一等奖"
    if red_hit == 6 and blue_hit == 0: return 2, "二等奖"
    if red_hit == 5 and blue_hit == 1: return 3, "三等奖"
    if red_hit == 5 and blue_hit == 0: return 4, "四等奖"
    if red_hit == 4 and blue_hit == 1: return 4, "四等奖"
    if red_hit == 4 and blue_hit == 0: return 5, "五等奖"
    if red_hit == 3 and blue_hit == 1: return 5, "五等奖"
    if red_hit in (0, 1, 2) and blue_hit == 1: return 6, "六等奖"
    return 0, "未中奖"


def review_pending(db, latest_issue, latest_row):
    """复盘上一期双色球预测"""
    pending_path = os.path.join(PROJECT_ROOT, "docs", "data", "ssq_pending.json")
    if not os.path.exists(pending_path):
        return 0, "无待复盘预测"
    with open(pending_path, "r", encoding="utf-8") as f:
        pending = json.load(f)
    target = str(pending.get("target_issue", ""))
    if target != latest_issue:
        return 0, f"待复盘期号{target}与最新{latest_issue}不匹配"
    actual_red = [int(latest_row[f"red{i}"]) for i in range(1, 7)]
    actual_blue = int(latest_row["blue"])
    reviewed = 0
    for g in pending.get("groups", []):
        pred_red = g.get("front", [])
        pred_blue = g.get("blue", 0)
        rh = len(set(pred_red) & set(actual_red))
        bh = 1 if pred_blue == actual_blue else 0
        lvl, name = calc_ssq_prize(rh, bh)
        db.save_prediction(latest_issue, g.get("label", ""), pred_red, pred_blue,
                           actual_red, actual_blue, rh, bh)
        reviewed += 1
    os.remove(pending_path)
    return reviewed, f"复盘{reviewed}组完成"


def run_predict(db_path="data/processed/pmsf_dlt.db", raw_dir="data/raw",
                use_web=True, target_issue=None, save_docs=True):
    """执行双色球完整预测"""
    print("=" * 60)
    print("  彭湃双色球预测系统 (SsqPengpai)")
    print("=" * 60)

    db = SsqDatabase(db_path)
    fetcher = SsqDataFetcher(raw_dir)

    # 1. 数据获取
    print("\n[1/4] 数据获取...")
    if db.count() > 100:
        df = db.load_all()
        print(f"  从数据库加载 {len(df)} 期")
        # 尝试网络增量更新
        if use_web:
            new_df = fetcher.get_data(use_web=True, use_mock_fallback=False)
            if not new_df.empty and len(new_df) > len(df):
                db.insert_batch(new_df)
                df = db.load_all()
                print(f"  网络更新后共 {len(df)} 期")
    else:
        df = fetcher.get_data(use_web=use_web, use_mock_fallback=True)
        if not df.empty:
            db.insert_batch(df)
            print(f"  已入库 {len(df)} 期")

    if df.empty:
        print("错误：无双色球数据可用")
        return None

    latest_issue = str(df["issue"].iloc[-1])
    latest_row = df.iloc[-1]

    # 2. 复盘上一期
    print("\n[2/4] 复盘上一期...")
    reviewed, msg = review_pending(db, latest_issue, latest_row)
    print(f"  {msg}")

    # 3. 预测
    print("\n[3/4] 彭湃规则分析 + 概率融合 + 组合优化...")
    analyzer = SsqAnalyzer()
    output = analyzer.predict(df, target_issue)
    print(f"  目标期号: {output['target_issue']}")
    print(f"  状态: {output['current_state']['state_name']} ({output['current_state']['state']})")
    print(f"  主过渡号: {output['rules']['main_transition']}")
    print(f"  副过渡号: {output['rules']['sub_transitions']}")
    for g in output["groups"]:
        reds = " ".join(str(n).zfill(2) for n in g["front"])
        print(f"  [{g['label']}组] {g['name']}: 红 {reds} | 蓝 {str(g['blue']).zfill(2)}")

    # 4. 保存
    print("\n[4/4] 保存结果...")
    out_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"ssq_output_{output['target_issue']}_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_path}")

    # 保存待复盘
    if save_docs:
        docs_data = os.path.join(PROJECT_ROOT, "docs", "data")
        os.makedirs(docs_data, exist_ok=True)
        with open(os.path.join(docs_data, "ssq_latest.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        with open(os.path.join(docs_data, "ssq_pending.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print("  docs/data/ssq_latest.json + ssq_pending.json 已更新")

    return output


def main():
    parser = argparse.ArgumentParser(description="彭湃双色球预测系统")
    parser.add_argument("--issue", type=str, default=None, help="目标期号")
    parser.add_argument("--no-web", action="store_true", help="不使用网络")
    args = parser.parse_args()
    run_predict(use_web=not args.no_web, target_issue=args.issue)


if __name__ == "__main__":
    main()
