"""
三态状态识别诊断 (修正版)
1. 正确调用 rule_based_state(单期指标) 看 A/B/C 分布
2. HSMM 预测状态分布
3. 各态下 热/冷/中号 实际命中分布 (验证各态偏置方向)
4. 各态下"状态偏置评分 vs 纯频率评分" Top10 命中对比
"""
import sys, sqlite3, time
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, r'E:\PMSF-V1')
from src.layer2_rules.dual_line import DualLineSystem
from src.layer2_rules.three_states import ThreeStateSystem
from src.layer3_state.hsmm_model import HSMMStateModel

CFG = yaml.safe_load(open(r'E:\PMSF-V1\config.yaml', encoding='utf-8'))
conn = sqlite3.connect(r'E:\PMSF-V1\data\processed\pmsf_dlt.db')
df = pd.read_sql("SELECT issue, date, front01, front02, front03, front04, front05 FROM dlt_history ORDER BY issue", conn)
conn.close()
FRONT = ["front01","front02","front03","front04","front05"]
df["sum_front"] = df[FRONT].sum(axis=1)

dl = DualLineSystem(CFG)
ts = ThreeStateSystem(CFG)
hsmm = HSMMStateModel(CFG)

N_TEST = 100
n = len(df)
start = n - N_TEST

def rank_top(scores, k=10):
    ordered = sorted(range(1,36), key=lambda x: -scores.get(x,0))
    return ordered[:k]
def hit(r10, actual):
    return len(set(r10) & set(actual))

# 状态分布
rule_dist = {"A":0,"B":0,"C":0}
hsmm_dist = {"A":0,"B":0,"C":0}
# 各态热/中/冷命中 (每期5个号在 hot_top10 / cold_bot10 / 中 的个数)
detail_rule = {"A":{"hot":[],"mid":[],"cold":[]},"B":{"hot":[],"mid":[],"cold":[]},"C":{"hot":[],"mid":[],"cold":[]}}
detail_hsmm = {"A":{"hot":[],"mid":[],"cold":[]},"B":{"hot":[],"mid":[],"cold":[]},"C":{"hot":[],"mid":[],"cold":[]}}
# 偏置评分 vs 纯频率
bias_hit = {"A":[],"B":[],"C":[]}
freq_hit = {"A":[],"B":[],"C":[]}

t0=time.time()
for idx in range(start, n):
    train = df.iloc[:idx].reset_index(drop=True)
    row = df.iloc[idx]
    target = str(row['issue'])
    actual = set(int(row[c]) for c in FRONT)
    line = dl.get_line_for_issue(target)
    line_df = dl.split(train)[line]

    # 频率(同线150) 与 热/冷集合
    f150 = {x:0 for x in range(1,36)}
    for _,r in line_df.tail(150).iterrows():
        for c in FRONT: f150[int(r[c])]+=1
    rank_f = sorted(range(1,36), key=lambda x:-f150[x])
    hot_set = set(rank_f[:10]); cold_set = set(rank_f[-10:])
    freq10 = hit(rank_top(f150), actual)

    # 指标
    ind = ts.compute_state_indicators(train)
    if ind.empty: continue
    last = ind.iloc[-1].to_dict()

    # rule_based_state
    st_r = ts.rule_based_state(last)
    rule_dist[st_r]+=1
    for a in actual:
        if a in hot_set: detail_rule[st_r]["hot"].append(1); detail_rule[st_r]["mid"].append(0); detail_rule[st_r]["cold"].append(0)
        elif a in cold_set: detail_rule[st_r]["cold"].append(1); detail_rule[st_r]["mid"].append(0); detail_rule[st_r]["hot"].append(0)
        else: detail_rule[st_r]["mid"].append(1); detail_rule[st_r]["hot"].append(0); detail_rule[st_r]["cold"].append(0)

    # HSMM 预测
    try:
        hsmm.fit(ind)
        sr = hsmm.predict_next(ind)
        st_h = max(["A","B","C"], key=lambda s: sr.get(s,0))
    except Exception:
        st_h = "B"
    hsmm_dist[st_h]+=1
    for a in actual:
        if a in hot_set: detail_hsmm[st_h]["hot"].append(1); detail_hsmm[st_h]["mid"].append(0); detail_hsmm[st_h]["cold"].append(0)
        elif a in cold_set: detail_hsmm[st_h]["cold"].append(1); detail_hsmm[st_h]["mid"].append(0); detail_hsmm[st_h]["hot"].append(0)
        else: detail_hsmm[st_h]["mid"].append(1); detail_hsmm[st_h]["hot"].append(0); detail_hsmm[st_h]["cold"].append(0)

    # 状态偏置评分 (应用 bias 到频率分)
    for st in ["A","B","C"]:
        bias = ts.get_state_bias(st)
        sc = {}
        for num in range(1,36):
            f = f150[num]
            is_hot = num in hot_set
            is_cold = num in cold_set
            w = 1.0
            if is_hot: w *= bias.get("hot_weight",1.0)
            if is_cold: w *= bias.get("cold_weight",1.0)
            sc[num] = f * w
        if st==st_h:
            bias_hit[st].append(hit(rank_top(sc), actual))
            freq_hit[st].append(freq10)

    if (idx-start)%20==0:
        print(f"进度{(idx-start)}/{N_TEST} ({time.time()-t0:.0f}s)")

print(f"\n完成 {time.time()-t0:.0f}s\n")
print("="*64)
print("① 状态分布 (100期)")
print(f"rule_based: A={rule_dist['A']} B={rule_dist['B']} C={rule_dist['C']}")
print(f"HSMM     : A={hsmm_dist['A']} B={hsmm_dist['B']} C={hsmm_dist['C']}")

print("\n" + "="*64)
print("② 各态下实际5号中 热/中/冷号 平均个数 (随机期望各1.43)")
print(f"{'状态':<5}{'来源':<10}{'热号':<8}{'中号':<8}{'冷号':<8}")
for st in ["A","B","C"]:
    d = detail_rule[st]
    if not d["hot"]: continue
    h=np.mean(d["hot"])*5; m=np.mean(d["mid"])*5; c=np.mean(d["cold"])*5
    print(f"{st:<5}{'rule':<10}{h:<8.2f}{m:<8.2f}{c:<8.2f}")
for st in ["A","B","C"]:
    d = detail_hsmm[st]
    if not d["hot"]: continue
    h=np.mean(d["hot"])*5; m=np.mean(d["mid"])*5; c=np.mean(d["cold"])*5
    print(f"{st:<5}{'hsmm':<10}{h:<8.2f}{m:<8.2f}{c:<8.2f}")

print("\n" + "="*64)
print("③ 状态偏置评分 vs 纯频率 Top10命中 (仅该态被判定时)")
for st in ["A","B","C"]:
    if not bias_hit[st]: continue
    b=np.mean(bias_hit[st]); f=np.mean(freq_hit[st])
    print(f"{st}: 偏置{b:.2f} vs 频率{f:.2f}  Δ={b-f:+.2f} (n={len(bias_hit[st])})")
