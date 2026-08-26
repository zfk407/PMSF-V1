"""
大乐透逐思路回测 (Walk-Forward, 真实2872期)
验证: 频率/同线频率/遗漏/隔期返点/图纠缠/谱嵌入/三态偏置/组合
指标: Top10命中 / Top15命中 / 平均排名 (基准: Top10=1.43, Top15=2.14, 排名18)
"""
import sys, sqlite3, time, yaml
import numpy as np
import pandas as pd
sys.path.insert(0, r'E:\PMSF-V1')
from src.layer2_rules.dual_line import DualLineSystem
from src.layer2_rules.number_graph import NumberGraph
from src.layer2_rules.three_states import ThreeStateSystem

CFG = yaml.safe_load(open(r'E:\PMSF-V1\config.yaml', encoding='utf-8'))
conn = sqlite3.connect(r'E:\PMSF-V1\data\processed\pmsf_dlt.db')
df = pd.read_sql("SELECT issue, date, front01, front02, front03, front04, front05, back01, back02 FROM dlt_history ORDER BY issue", conn)
conn.close()
FRONT = ["front01","front02","front03","front04","front05"]

dl = DualLineSystem(CFG)
ng = NumberGraph(CFG)
ts = ThreeStateSystem(CFG)

N_TEST = 80
n = len(df)
start = n - N_TEST
rng = np.random.default_rng(42)

def rank_scores(scores):
    ordered = sorted(range(1,36), key=lambda x: -scores.get(x,0))
    rk = {num:i+1 for i,num in enumerate(ordered)}
    return rk, ordered

metrics = {}
def add(name, t10, t15, ar):
    metrics.setdefault(name, {"top10":[],"top15":[],"avg_rank":[]})
    metrics[name]["top10"].append(t10)
    metrics[name]["top15"].append(t15)
    metrics[name]["avg_rank"].append(ar)

def order_weight(order, lam=0.35):
    return np.exp(-lam*(order-1))

t0 = time.time()
for idx in range(start, n):
    train = df.iloc[:idx].reset_index(drop=True)
    row = df.iloc[idx]
    target = str(row['issue'])
    actual = [int(row[c]) for c in FRONT]

    if len(train) < 200:
        continue
    line = dl.get_line_for_issue(target)
    line_df = dl.split(train)[line]

    # 频率
    freq100 = {num: 0 for num in range(1,36)}
    for _, r in train.tail(100).iterrows():
        for c in FRONT: freq100[int(r[c])] += 1
    freq_line = {num: 0 for num in range(1,36)}
    for _, r in line_df.tail(60).iterrows():
        for c in FRONT: freq_line[int(r[c])] += 1
    # 遗漏(同线)
    miss = {}
    for num in range(1,36):
        m=0
        for _, r in reversed(list(line_df.iterrows())):
            if num in [int(r[c]) for c in FRONT]: break
            m+=1
        miss[num]=m
    # 隔期返点(同线, 1-5阶 λ=0.35)
    period_nums = [set(int(r[c]) for c in FRONT) for _, r in line_df.iterrows()]
    rebound = {}
    if len(period_nums) >= 5:
        rb = {}
        for order in range(1,6):
            w = order_weight(order)
            for num in period_nums[-order]:
                rb[num] = rb.get(num,0)+w
        mx = max(rb.values()) if rb else 1
        rebound = {k:v/mx for k,v in rb.items()}
    # 图纠缠(同线建图)
    ng.build_from_history(line_df, window=200)
    graph_score = {num: ng.get_entanglement_score(num) for num in range(1,36)}
    # 谱嵌入第一维
    emb = ng.get_node_embedding(2)
    emb_score = {num: emb[num][0] for num in range(1,36)}
    # 三态偏置
    try:
        indicators = ts.compute_state_indicators(train)
        state = ts.rule_based_state(indicators)
    except Exception:
        state = "B"
    bias = ts.get_state_bias(state)
    # 号码热/冷属性(近30期频率)
    freq30 = {num: 0 for num in range(1,36)}
    for _, r in train.tail(30).iterrows():
        for c in FRONT: freq30[int(r[c])] += 1
    fmax = max(freq30.values()) if freq30 else 1
    state_score = {}
    for num in range(1,36):
        f = freq30[num]
        is_hot = f >= fmax*0.7
        is_cold = f <= 2
        w = 1.0
        if is_hot: w *= bias.get("hot_weight",1.0)
        if is_cold: w *= bias.get("cold_weight",1.0)
        state_score[num] = w * (1 + f/10.0)
    # 组合: 频率+遗漏+返点
    combo = {num: freq_line.get(num,0)/6.0 + miss.get(num,0)/40.0 + rebound.get(num,0)*2.0 for num in range(1,36)}

    scores = {
        "random": {num: rng.random() for num in range(1,36)},
        "freq100": freq100,
        "freq_line": freq_line,
        "miss": miss,
        "rebound": rebound,
        "graph": graph_score,
        "graph_emb": emb_score,
        "state_bias": state_score,
        "combo": combo,
    }
    for name, sc in scores.items():
        if not sc or max(sc.values())==min(sc.values()):
            continue
        rk, ordered = rank_scores(sc)
        t10 = len(set(ordered[:10]) & set(actual))
        t15 = len(set(ordered[:15]) & set(actual))
        ar = np.mean([rk[a] for a in actual])
        add(name, t10, t15, ar)
    if (idx-start) % 10 == 0:
        print(f"进度 {idx-start}/{N_TEST} ({time.time()-t0:.0f}s)")

print(f"\n回测完成 耗时{time.time()-t0:.0f}s\n")
print("="*88)
print(f"{'思路':<14}{'Top10':<9}{'Top15':<9}{'平均排名':<9}  ΔTop10 vs随机(1.43)")
print("="*88)
for name in ["random","freq100","freq_line","miss","rebound","graph","graph_emb","state_bias","combo"]:
    if name not in metrics or not metrics[name]["top10"]:
        continue
    m = metrics[name]
    t10=np.mean(m["top10"]); t15=np.mean(m["top15"]); ar=np.mean(m["avg_rank"])
    delta = t10-1.43
    verdict = "★★★" if delta>0.25 else ("✓" if delta>0.08 else ("—" if delta>-0.08 else "✗"))
    print(f"{name:<14}{t10:<9.2f}{t15:<9.2f}{ar:<9.1f}  {delta:+.2f} {verdict}")
