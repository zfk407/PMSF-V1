"""
验证优化: 新 rule_bias(同线150频率x遗漏回补x3阶返点) vs 纯同线150频率
100期 walk-forward, Top10命中对比
"""
import sys, sqlite3, time
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, r'E:\PMSF-V1')
from src.layer2_rules.dual_line import DualLineSystem

CFG = yaml.safe_load(open(r'E:\PMSF-V1\config.yaml', encoding='utf-8'))
conn = sqlite3.connect(r'E:\PMSF-V1\data\processed\pmsf_dlt.db')
df = pd.read_sql("SELECT issue, date, front01, front02, front03, front04, front05 FROM dlt_history ORDER BY issue", conn)
conn.close()
FRONT = ["front01","front02","front03","front04","front05"]
dl = DualLineSystem(CFG)

N_TEST = 100
n = len(df); start = n - N_TEST

def rank_top(scores, k=10):
    return sorted(range(1,36), key=lambda x: -scores.get(x,0))[:k]
def hit(r10, actual):
    return len(set(r10) & set(actual))

bl = {"base":[], "opt":[]}
t0=time.time()
for idx in range(start, n):
    train = df.iloc[:idx].reset_index(drop=True)
    row = df.iloc[idx]
    target = str(row['issue'])
    actual = set(int(row[c]) for c in FRONT)
    line = dl.get_line_for_issue(target)
    line_df = dl.split(train)[line]

    # 同线150频率
    freq = {x:0 for x in range(1,36)}
    for _,r in line_df.tail(150).iterrows():
        for c in FRONT: freq[int(r[c])]+=1
    fmax = max(freq.values()) if freq else 1

    # 遗漏
    psets = [set(int(r[c]) for c in FRONT) for _,r in line_df.iterrows()]
    miss = {x:0 for x in range(1,36)}
    for x in range(1,36):
        m=0
        for s in reversed(psets):
            if x in s: break
            m+=1
        miss[x]=m
    # 3阶返点
    stage3 = set(psets[-3]) if len(psets)>=3 else set()

    base_sc = {x: freq[x] for x in range(1,36)}
    opt_sc = {}
    for x in range(1,36):
        b = 0.92 + 0.16*(freq[x]/fmax)
        if miss[x]>=11: b *= 1.0+0.22*min((miss[x]-11)/15.0,1.0)
        if x in stage3: b *= 1.12
        opt_sc[x] = freq[x]*min(max(b,0.85),1.22)

    bl["base"].append(hit(rank_top(base_sc), actual))
    bl["opt"].append(hit(rank_top(opt_sc), actual))
    if (idx-start)%25==0:
        print(f"进度{(idx-start)}/{N_TEST} ({time.time()-t0:.0f}s)")

print(f"\n完成 {time.time()-t0:.0f}s, n={N_TEST}\n")
print("="*60)
print("Top10命中均值 (随机≈1.43)")
print("="*60)
print(f"基线(同线150频率)      : {np.mean(bl['base']):.2f}")
print(f"优化(频率x遗漏x3阶返点): {np.mean(bl['opt']):.2f}")
print(f"增量                    : {np.mean(bl['opt'])-np.mean(bl['base']):+.2f}")
# 逐期胜率
wins = sum(1 for a,b in zip(bl['base'],bl['opt']) if b>a)
draws = sum(1 for a,b in zip(bl['base'],bl['opt']) if b==a)
print(f"逐期胜率: 优化胜{int(wins)}期 平{int(draws)}期 负{int(N_TEST-wins-draws)}期")
