"""
大乐透优化v2: 加法信号融合回测 (07001~26096全量)
对比: 纯同线频率 vs 频率+遗漏回补(加法) vs 频率+3阶返点(加法) vs 全组合
同时优化后区(全量/同线/窗口/遗漏)
用最优配置 + 全量历史 输出 26097
"""
import sys, sqlite3, time, json
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
print(f"全量数据: {len(df)}期 (07001~26096), 回测区间: 25001~26096 ({end_idx-start_idx+1}期)")

def parity(issue):
    return "single" if int(str(issue)[-1]) % 2 == 1 else "double"

def freq_scores(line_df, w, cols, rng_):
    f = {x: 0 for x in range(1, rng_+1)}
    for _, r in line_df.tail(w).iterrows():
        for c in cols: f[int(r[c])] += 1
    return f

def hit10(scores, actual):
    return len(set(sorted(scores, key=lambda x: -scores[x])[:10]) & set(actual))

# 候选组合 (加法融合)
COMBOS = {
    "freq":            lambda fr, mb, s3: dict(fr),
    "freq+0.5miss":    lambda fr, mb, s3: {x: fr[x] + 0.5*mb[x] for x in fr},
    "freq+1.0miss":    lambda fr, mb, s3: {x: fr[x] + 1.0*mb[x] for x in fr},
    "freq+0.5s3":      lambda fr, mb, s3: {x: fr[x] + 0.5*(1 if x in s3 else 0) for x in fr},
    "freq+1.0s3":      lambda fr, mb, s3: {x: fr[x] + 1.0*(1 if x in s3 else 0) for x in fr},
    "freq+0.5m+0.5s3": lambda fr, mb, s3: {x: fr[x] + 0.5*mb[x] + 0.5*(1 if x in s3 else 0) for x in fr},
    "freq+1.0m+0.5s3": lambda fr, mb, s3: {x: fr[x] + 1.0*mb[x] + 0.5*(1 if x in s3 else 0) for x in fr},
    "freq+1.0m+1.0s3": lambda fr, mb, s3: {x: fr[x] + 1.0*mb[x] + 1.0*(1 if x in s3 else 0) for x in fr},
}
# 窗口候选(在线学习)
WINS = [50,80,100,120,150,180,200]

t0 = time.time()
# 每期特征缓存 (回测期)
feat_cache = {}
for idx in range(start_idx, end_idx+1):
    train = df.iloc[:idx].reset_index(drop=True)
    row = df.iloc[idx]
    target = str(row['issue'])
    actual = set(int(row[c]) for c in FRONT)
    line = parity(target)
    line_df = dl.split(train)[line]
    psets = [set(int(r[c]) for c in FRONT) for _, r in line_df.iterrows()]
    # 遗漏(全量线, 同线)
    miss = {x: 0 for x in range(1, 36)}
    for x in range(1, 36):
        m = 0
        for s in reversed(psets):
            if x in s: break
            m += 1
        miss[x] = m
    mb = {x: min(miss[x]/25.0, 1.0) for x in range(1,36)}  # 长遗漏连续加分
    s3 = set(psets[-3]) if len(psets) >= 3 else set()
    feat_cache[idx] = {"actual": actual, "line_df": line_df, "psets": psets,
                       "mb": mb, "s3": s3, "line": line, "target": target}

# 在线学习: 动态窗口(每20期复盘)
res = {k: [] for k in COMBOS}
dynamic_w = 150
# 记录每个窗口的命中历史用于复盘
for idx in range(start_idx, end_idx+1):
    fc = feat_cache[idx]
    fr = freq_scores(fc["line_df"], dynamic_w, FRONT, 35)
    for name, fn in COMBOS.items():
        sc = fn(fr, fc["mb"], fc["s3"])
        res[name].append(hit10(sc, fc["actual"]))
    # 每20期复盘
    if (idx-start_idx+1) % 20 == 0:
        recent_idx = list(range(idx-19, idx+1))
        win_scores = {}
        for cand in WINS:
            s = 0
            for ri in recent_idx:
                rfc = feat_cache[ri]
                frr = freq_scores(rfc["line_df"], cand, FRONT, 35)
                s += hit10(frr, rfc["actual"])
            win_scores[cand] = s
        best = max(win_scores, key=win_scores.get)
        dynamic_w = best

print(f"回测完成 {time.time()-t0:.0f}s, 动态窗口收敛={dynamic_w}\n")
print("="*82)
print(f"{'组合':<18}{'Top10':<8}{'Top15':<8}{'ΔTop10':<8} 判定")
print("="*82)
summary = {}
for name in COMBOS:
    arr = res[name]
    t10 = np.mean(arr)
    # Top15 需要重算? 简化用Top10为主
    base = np.mean(res["freq"])
    delta = t10 - base
    verdict = "★★★" if delta > 0.08 else ("✓" if delta > 0.03 else ("—" if delta > -0.03 else "✗"))
    print(f"{name:<18}{t10:<8.2f}{'':<8}{delta:<+8.2f} {verdict} (vs 纯频率{base:.2f})")
    summary[name] = float(t10)
# 随机基准
rand_est = 1.43
print(f"\n随机基准 ≈ {rand_est}, 纯频率Top10 = {summary['freq']:.2f}")

# 后区优化
print("\n" + "="*82)
print("后区策略优化 (12选2, 随机期望2*2/12=0.333)")
print("="*82)
back_combos = {
    "line_w80": [], "all_w80": [], "line_w150": [], "all_w150": [],
    "line_w30": [], "line_w30_miss": []
}
for idx in range(start_idx, end_idx+1):
    fc = feat_cache[idx]
    actual_b = set(int(df.iloc[idx][c]) for c in BACK)
    line_df = fc["line_df"]; train_all = df.iloc[:idx]
    lb80 = freq_scores(line_df, 80, BACK, 12)
    ab80 = freq_scores(train_all, 80, BACK, 12)
    lb150 = freq_scores(line_df, 150, BACK, 12)
    ab150 = freq_scores(train_all, 150, BACK, 12)
    lb30 = freq_scores(line_df, 30, BACK, 12)
    # 后区遗漏(同线30)
    bmiss = {x:0 for x in range(1,13)}
    bseq = [int(r[c]) for c in BACK for r in line_df.tail(30).iterrows()] if False else None
    bpsets = [set(int(r[c]) for c in BACK) for _, r in line_df.iterrows()]
    for x in range(1,13):
        m=0
        for s in reversed(bpsets):
            if x in s: break
            m+=1
        bmiss[x]=m
    lb30m = {x: lb30[x] + 0.5*min(bmiss[x]/20,1) for x in range(1,13)}
    def hb(sc):
        return len(set(sorted(sc, key=lambda x:-sc[x])[:2]) & actual_b)
    back_combos["line_w80"].append(hb(lb80)); back_combos["all_w80"].append(hb(ab80))
    back_combos["line_w150"].append(hb(lb150)); back_combos["all_w150"].append(hb(ab150))
    back_combos["line_w30"].append(hb(lb30)); back_combos["line_w30_miss"].append(hb(lb30m))
for name, arr in back_combos.items():
    print(f"  {name:<16} 平均命中 {np.mean(arr):.3f}")

json.dump({"combos": summary, "back": {k: float(np.mean(v)) for k,v in back_combos.items()}},
          open(r'E:\PMSF-V1\docs\data\dlt_opt2_backtest.json','w',encoding='utf-8'), ensure_ascii=False)
print("\n已保存 docs/data/dlt_opt2_backtest.json")
