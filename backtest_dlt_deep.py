"""
大乐透专项深度回测 (100期 walk-forward, 2872期真实)
聚焦两个有效信号 + 深挖边界:
1. 频率窗口调优 (20/30/50/80/100/150/200 全量与同线 + 指数衰减)
2. 三态偏置细分 (A/B/C各态下 热/中/冷 号码实际命中分布)
3. 遗漏形态 (分桶命中率)
4. 返点分阶 (1-5阶各自贡献, λ衰减验证)
5. 双线隔离验证 (同线 vs 跨线)
"""
import sys, sqlite3, time, json
import numpy as np
import pandas as pd
sys.path.insert(0, r'E:\PMSF-V1')
from src.layer2_rules.dual_line import DualLineSystem
from src.layer2_rules.three_states import ThreeStateSystem
import yaml

CFG = yaml.safe_load(open(r'E:\PMSF-V1\config.yaml', encoding='utf-8'))
conn = sqlite3.connect(r'E:\PMSF-V1\data\processed\pmsf_dlt.db')
df = pd.read_sql("SELECT issue, date, front01, front02, front03, front04, front05 FROM dlt_history ORDER BY issue", conn)
conn.close()
FRONT = ["front01","front02","front03","front04","front05"]

dl = DualLineSystem(CFG)
ts = ThreeStateSystem(CFG)

N_TEST = 100
n = len(df)
start = n - N_TEST
rng = np.random.default_rng(99)

def rank_top(scores, k=10):
    ordered = sorted(range(1,36), key=lambda x: -scores.get(x,0))
    return ordered[:k]

def hit(ranked10, actual):
    return len(set(ranked10) & set(actual))

# 收集器
WIN = [20,30,50,80,100,150,200]
res = {f"freq{w}": [] for w in WIN}
res.update({f"freq_line{w}": [] for w in WIN})
res["freq_exp"] = []      # 指数衰减全量
res["freq_exp_line"] = [] # 指数衰减同线
res["random"] = []
# 三态细分: state -> hot/mid/cold 实际命中数 (每期5个号, 记录各属于热/中/冷几个)
state_detail = {"A": {"hot":[], "mid":[], "cold":[], "n":0},
                "B": {"hot":[], "mid":[], "cold":[], "n":0},
                "C": {"hot":[], "mid":[], "cold":[], "n":0}}
# 遗漏分桶: 每期35号码按遗漏分桶, 记录每桶号码数与该桶命中数
miss_buckets = {"0-2": {"total":0,"hit":0}, "3-5": {"total":0,"hit":0},
                "6-10": {"total":0,"hit":0}, "11-20": {"total":0,"hit":0},
                "21+": {"total":0,"hit":0}}
# 返点分阶
reb_stage = {f"stage{k}": {"n":0,"hit":0} for k in range(1,6)}
# 双线隔离
line_same = []; line_cross = []

def exp_weights(n, lam=0.05):
    return np.exp(-lam*np.arange(n)[::-1])

t0 = time.time()
for idx in range(start, n):
    train = df.iloc[:idx].reset_index(drop=True)
    row = df.iloc[idx]
    target = str(row['issue'])
    actual = set(int(row[c]) for c in FRONT)
    line = dl.get_line_for_issue(target)
    line_df = dl.split(train)[line]

    # ---- 频率窗口 ----
    for w in WIN:
        fw = {x:0 for x in range(1,36)}
        for _,r in train.tail(w).iterrows():
            for c in FRONT: fw[int(r[c])]+=1
        res[f"freq{w}"].append(hit(rank_top(fw), actual))
        fl = {x:0 for x in range(1,36)}
        for _,r in line_df.tail(w).iterrows():
            for c in FRONT: fl[int(r[c])]+=1
        res[f"freq_line{w}"].append(hit(rank_top(fl), actual))
    # 指数衰减
    nwin = 100
    fexp = {x:0.0 for x in range(1,36)}
    wts = exp_weights(min(nwin,len(train)))
    for _,r in train.tail(nwin).iterrows():
        pass
    for i,(_,r) in enumerate(train.tail(nwin).iterrows()):
        w = wts[i]
        for c in FRONT: fexp[int(r[c])]+=w
    res["freq_exp"].append(hit(rank_top(fexp), actual))
    fexp_l = {x:0.0 for x in range(1,36)}
    for i,(_,r) in enumerate(line_df.tail(nwin).iterrows()):
        w = exp_weights(min(nwin,len(line_df)))[i]
        for c in FRONT: fexp_l[int(r[c])]+=w
    res["freq_exp_line"].append(hit(rank_top(fexp_l), actual))
    res["random"].append(hit(rank_top({x:rng.random() for x in range(1,36)}), actual))

    # ---- 三态细分 ----
    try:
        indicators = ts.compute_state_indicators(train)
        state = ts.rule_based_state(indicators)
    except Exception:
        state = "B"
    f100 = {x:0 for x in range(1,36)}
    for _,r in train.tail(100).iterrows():
        for c in FRONT: f100[int(r[c])]+=1
    rank_f = sorted(range(1,36), key=lambda x:-f100[x])
    hot_set = set(rank_f[:10]); cold_set = set(rank_f[-10:])
    for a in actual:
        if a in hot_set: state_detail[state]["hot"].append(1); state_detail[state]["mid"].append(0); state_detail[state]["cold"].append(0)
        elif a in cold_set: state_detail[state]["cold"].append(1); state_detail[state]["hot"].append(0); state_detail[state]["mid"].append(0)
        else: state_detail[state]["mid"].append(1); state_detail[state]["hot"].append(0); state_detail[state]["cold"].append(0)
    state_detail[state]["n"] += 1

    # ---- 遗漏分桶 ----
    miss = {}
    for num in range(1,36):
        m=0
        for _,r in reversed(list(line_df.iterrows())):
            if num in [int(r[c]) for c in FRONT]: break
            m+=1
        miss[num]=m
    for num in range(1,36):
        m = miss[num]
        key = "0-2" if m<=2 else ("3-5" if m<=5 else ("6-10" if m<=10 else ("11-20" if m<=20 else "21+")))
        miss_buckets[key]["total"] += 1
        if num in actual: miss_buckets[key]["hit"] += 1

    # ---- 返点分阶 ----
    period_nums = [set(int(r[c]) for c in FRONT) for _,r in line_df.iterrows()]
    for k in range(1,6):
        if len(period_nums) >= k:
            stage_nums = period_nums[-k]
            reb_stage[f"stage{k}"]["n"] += len(stage_nums)
            reb_stage[f"stage{k}"]["hit"] += len(stage_nums & actual)

    # ---- 双线隔离 ----
    cross = "single" if line=="double" else "double"
    cross_df = dl.split(train)[cross]
    f_same = {x:0 for x in range(1,36)}
    for _,r in line_df.tail(100).iterrows():
        for c in FRONT: f_same[int(r[c])]+=1
    f_cross = {x:0 for x in range(1,36)}
    for _,r in cross_df.tail(100).iterrows():
        for c in FRONT: f_cross[int(r[c])]+=1
    line_same.append(hit(rank_top(f_same), actual))
    line_cross.append(hit(rank_top(f_cross), actual))

    if (idx-start)%20==0:
        print(f"进度 {(idx-start)}/{N_TEST} ({time.time()-t0:.0f}s)")

print(f"\n完成 耗时{time.time()-t0:.0f}s\n")

out = {}
print("="*70)
print("① 频率窗口调优 (Top10命中均值, 随机基准≈1.43)")
print("="*70)
print(f"{'窗口':<8}{'全量':<10}{'同线':<10}  最优")
for w in WIN:
    a = np.mean(res[f"freq{w}"]); b = np.mean(res[f"freq_line{w}"])
    mark = "←" if a==max(np.mean(res[f"freq{w}"]) for w in WIN) else ""
    print(f"{w:<8}{a:<10.2f}{b:<10.2f}  {mark}")
print(f"{'exp100':<8}{np.mean(res['freq_exp']):<10.2f}{np.mean(res['freq_exp_line']):<10.2f}")
print(f"{'random':<8}{np.mean(res['random']):<10.2f}")
out["freq_window"] = {f"freq{w}": float(np.mean(res[f"freq{w}"])) for w in WIN}
out["freq_window"].update({f"freq_line{w}": float(np.mean(res[f"freq_line{w}"])) for w in WIN})
out["freq_window"]["freq_exp"] = float(np.mean(res["freq_exp"]))
out["freq_window"]["freq_exp_line"] = float(np.mean(res["freq_exp_line"]))
out["freq_window"]["random"] = float(np.mean(res["random"]))

print("\n" + "="*70)
print("② 三态偏置细分 (各状态下, 实际5号中热/中/冷号平均个数)")
print("="*70)
print(f"{'状态':<6}{'期数':<6}{'热号':<8}{'中号':<8}{'冷号':<8}  随机期望: 热1.43 冷1.43")
for s in ["A","B","C"]:
    d = state_detail[s]
    if d["n"]==0: continue
    hv = np.mean(d["hot"]); mv=np.mean(d["mid"]); cv=np.mean(d["cold"])
    print(f"{s:<6}{d['n']:<6}{hv:<8.2f}{mv:<8.2f}{cv:<8.2f}")
    out[f"state_{s}"] = {"n":d["n"], "hot":float(hv), "mid":float(mv), "cold":float(cv)}

print("\n" + "="*70)
print("③ 遗漏分桶 (号码命中率/随机期望, 35桶均匀≈0.143)")
print("="*70)
for k,v in miss_buckets.items():
    rate = v["hit"]/v["total"] if v["total"] else 0
    print(f"{k:<8} 总{v['total']:<6} 命中{v['hit']:<4} 命中率{rate:.3f}")
    out[f"miss_bucket_{k}"] = {"total":v["total"],"hit":v["hit"],"rate":float(rate)}

print("\n" + "="*70)
print("④ 返点分阶 (1-5阶, 每期命中率)")
print("="*70)
for k in range(1,6):
    v = reb_stage[f"stage{k}"]
    rate = v["hit"]/v["n"] if v["n"] else 0
    print(f"第{k}阶(往前{k}同线期): 号码数{v['n']:<5} 命中率{rate:.3f}")
    out[f"reb_stage{k}"] = {"n":v["n"],"hit":v["hit"],"rate":float(rate)}

print("\n" + "="*70)
print("⑤ 双线隔离验证 (Top10命中)")
print("="*70)
print(f"同线频率: {np.mean(line_same):.2f}  (正确用法)")
print(f"跨线频率: {np.mean(line_cross):.2f}  (错误用法)")
out["line_same"] = float(np.mean(line_same)); out["line_cross"] = float(np.mean(line_cross))

json.dump(out, open(r'E:\PMSF-V1\docs\data\dlt_deep_backtest.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print("\n已保存 docs/data/dlt_deep_backtest.json")
