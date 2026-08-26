"""
26097组合生成逻辑的组合级回测验证
用与26097完全相同的规则(窗口80同线频率+结构约束→A组; +次级信号→B组)
回测 25001~26096 共246期, 统计每组前区/后区命中分布与奖级
对照随机期望: 前区命中5*5/35=0.714, 后区2*2/12=0.333
"""
import sys, sqlite3, json
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, r'E:\PMSF-V1')
from src.layer2_rules.dual_line import DualLineSystem

CFG = yaml.safe_load(open(r'E:\PMSF-V1\config.yaml', encoding='utf-8'))
conn = sqlite3.connect(r'E:\PMSF-V1\data\processed\pmsf_dlt.db')
df = pd.read_sql("SELECT issue, date, front01, front02, front03, front04, front05, back01, back02 FROM dlt_history ORDER BY issue", conn)
conn.close()
FRONT = ["front01","front02","front03","front04","front05"]
BACK = ["back01","back02"]
df = pd.concat([df, pd.DataFrame([{"issue":"26096","date":"2026-08-24","front01":8,"front02":9,
    "front03":10,"front04":11,"front05":25,"back01":4,"back02":12}])], ignore_index=True)
df = df.sort_values("issue").reset_index(drop=True)
dl = DualLineSystem(CFG)
issues = df["issue"].astype(str).tolist()
start_idx = issues.index("25001")
end_idx = len(df) - 1

def parity(issue):
    return "single" if int(str(issue)[-1]) % 2 == 1 else "double"

def freq_scores(line_df, w, cols=FRONT, rng_=35):
    f = {x: 0 for x in range(1, rng_+1)}
    for _, r in line_df.tail(w).iterrows():
        for c in cols: f[int(r[c])] += 1
    return f

def pick_balanced(nums, k=5):
    for target_odd in (2, 3):
        res = []
        for n in nums:
            is_odd = n % 2 == 1
            cnt_odd = sum(1 for x in res if x % 2 == 1)
            cnt_even = len(res) - cnt_odd
            if is_odd and cnt_odd >= target_odd: continue
            if (not is_odd) and cnt_even >= (k - target_odd): continue
            res.append(n)
            if len(res) == k: break
        if len(res) == k:
            zones = set(1 if x<=9 else 2 if x<=18 else 3 if x<=27 else 4 for x in res)
            if len(zones) >= 3:
                return sorted(res)
    return sorted(nums[:k])

def make_group(f_f, line_df, mode="A"):
    """与26097完全相同的生成逻辑"""
    if mode == "A":
        return pick_balanced(sorted(f_f, key=lambda x: -f_f[x]), 5)
    # B组: 频率+次级信号
    psets = [set(int(r[c]) for c in FRONT) for _, r in line_df.iterrows()]
    miss = {x: 0 for x in range(1, 36)}
    for x in range(1, 36):
        m = 0
        for s in reversed(psets):
            if x in s: break
            m += 1
        miss[x] = m
    stage3 = set(psets[-3]) if len(psets) >= 3 else set()
    last_period = set(psets[-1]) if psets else set()
    fmax = max(f_f.values()) if f_f else 1
    b_sec = {}
    for x in range(1, 36):
        s = f_f[x] / fmax
        if x in stage3: s += 0.12
        if miss[x] >= 11: s += 0.10 * min((miss[x]-11)/15, 1)
        if x in last_period: s += 0.08
        b_sec[x] = s
    return pick_balanced(sorted(b_sec, key=lambda x: -b_sec[x]), 5)

def pick_back(f_b):
    return sorted(f_b, key=lambda x: -f_b[x])[:2]

def prize(f, b):
    if f==5 and b==2: return 1
    if f==5 and b==1: return 2
    if f==5 and b==0: return 3
    if f==4 and b==2: return 4
    if f==4 and b==1: return 5
    if f==3 and b==2: return 6
    if f==4 and b==0: return 7
    if (f==3 and b==1) or (f==2 and b==2): return 8
    if (f==3 and b==0) or (f==1 and b==2) or (f==2 and b==1) or (f==0 and b==2): return 9
    return None

# 固定窗口80 (与26097一致) + 动态对比
results = {"A_fixed80": [], "B_fixed80": [], "A_dynamic": [], "B_dynamic": []}
WINS = [50,80,100,120,150,180,200]
dynamic_w = 150
switches = 0

for idx in range(start_idx, end_idx+1):
    train = df.iloc[:idx].reset_index(drop=True)
    row = df.iloc[idx]
    target = str(row['issue'])
    af = set(int(row[c]) for c in FRONT)
    ab = set(int(row[c]) for c in BACK)
    line = parity(target)
    line_df = dl.split(train)[line]

    # 固定窗口80
    f_f = freq_scores(line_df, 80, FRONT, 35)
    f_b = freq_scores(line_df, 80, BACK, 12)
    ga = make_group(f_f, line_df, "A")
    gb = make_group(f_f, line_df, "B")
    bg = pick_back(f_b)
    fa = len(set(ga) & af); ba = len(set(bg) & ab)
    fb = len(set(gb) & af); bb = len(set(bg) & ab)
    results["A_fixed80"].append({"issue":target,"f":fa,"b":ba,"grp":ga,"back":bg,"prize":prize(fa,ba)})
    results["B_fixed80"].append({"issue":target,"f":fb,"b":bb,"grp":gb,"back":bg,"prize":prize(fb,bb)})

    # 动态窗口 (每期用当前learned窗口)
    f_fd = freq_scores(line_df, dynamic_w, FRONT, 35)
    f_bd = freq_scores(line_df, dynamic_w, BACK, 12)
    ga_d = make_group(f_fd, line_df, "A")
    gb_d = make_group(f_fd, line_df, "B")
    bg_d = pick_back(f_bd)
    fad = len(set(ga_d) & af); bad = len(set(bg_d) & ab)
    fbd = len(set(gb_d) & af); bbd = len(set(bg_d) & ab)
    results["A_dynamic"].append({"issue":target,"f":fad,"b":bad,"prize":prize(fad,bad)})
    results["B_dynamic"].append({"issue":target,"f":fbd,"b":bbd,"prize":prize(fbd,bbd)})

    # 动态窗口学习(同模拟: 每20期复盘)
    if (idx-start_idx+1) % 20 == 0:
        recent = [r for r in results["A_dynamic"][-20:]]
        win_scores = {}
        for cand in WINS:
            s = 0
            for rec in recent:
                ti = issues.index(rec["issue"])
                tr = df.iloc[:ti]; rw = df.iloc[ti]
                ln = parity(rec["issue"])
                ld = dl.split(tr)[ln]
                ff = freq_scores(ld, cand, FRONT, 35)
                p10 = set(sorted(ff, key=lambda x: -ff[x])[:10])
                s += len(p10 & set(int(rw[c]) for c in FRONT))
            win_scores[cand] = s
        best = max(win_scores, key=win_scores.get)
        if best != dynamic_w:
            dynamic_w = best; switches += 1

def report(name, recs):
    fr = np.mean([r["f"] for r in recs]); br = np.mean([r["b"] for r in recs])
    fdist = [0]*6; bdist = [0]*3
    prize_dist = {}
    for r in recs:
        fdist[r["f"]] += 1; bdist[r["b"]] += 1
        p = r["prize"]
        prize_dist[p] = prize_dist.get(p, 0) + 1
    n = len(recs)
    win_rate = 1 - prize_dist.get(None, 0) / n
    print(f"\n{name} ({n}期)")
    print(f"  前区平均命中: {fr:.3f} (随机0.714)  后区平均: {br:.3f} (随机0.333)")
    print(f"  前区命中分布 0~5个: {fdist}")
    print(f"  后区命中分布 0~2个: {bdist}")
    print(f"  中奖率(任一奖级): {win_rate*100:.1f}%")
    if prize_dist:
        tiers = {k:v for k,v in sorted(prize_dist.items(), key=lambda x: (x[0] is None, x[0]))}
        print(f"  奖级分布: {tiers}")
    return fr, br

print("="*76)
print("26097 组合生成逻辑 回测验证 (25001~26096, 246期)")
print("="*76)
print("注: 前区5中5理论, 中3前区+1后区及以上即中奖")
for k in ["A_fixed80","B_fixed80","A_dynamic","B_dynamic"]:
    report(k, results[k])

print(f"\n动态窗口切换次数: {switches}, 最终窗口: {dynamic_w}")
# 26097实际组合在这套规则下的历史同类表现
print("\n" + "="*76)
print("26097 具体组合:")
print("  A组: 05 08 13 14 26 + 02 04  (固定窗口80 A组规则)")
print("  B组: 03 04 05 14 26 + 02 04  (固定窗口80 B组规则)")
print("  即上方 A_fixed80 / B_fixed80 的历史命中分布即该规则的预期表现")

json.dump(results, open(r'E:\PMSF-V1\docs\data\dlt_combo_backtest.json','w',encoding='utf-8'), ensure_ascii=False)
print("\n已保存 docs/data/dlt_combo_backtest.json")
