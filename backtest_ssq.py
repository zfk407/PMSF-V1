"""
双色球逐思路回测 (Walk-Forward)
验证: 频率/遗漏/贝叶斯/彭湃各规则因子/算法模型(Markov/Graph/Temporal/ML)/融合
指标: Top10命中 / Top15命中 / 平均排名 / 蓝球命中
基准: 随机期望 Top10命中=1.82, 平均排名=17
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, r'E:\PMSF-V1')
from src.ssq.data import SsqDataFetcher
from src.ssq.rules import SsqPengpaiRules
from src.ssq.models import SsqModelFusion

# 加载真实数据
raw = json.load(open(r'E:\PMSF-V1\data\raw\ssq_cwl_real.json', encoding='utf-8'))
recs = []
for it in raw:
    r = [int(x) for x in it['red'].split(',')]
    recs.append({'issue': it['code'], 'red01':r[0],'red02':r[1],'red03':r[2],
                 'red04':r[3],'red05':r[4],'red06':r[5],'blue':int(it['blue'])})
df = pd.DataFrame(recs).sort_values('issue').reset_index(drop=True)
df = SsqDataFetcher.enrich_derived_features(df)
RED = ["red01","red02","red03","red04","red05","red06"]

rules = SsqPengpaiRules()
fusion = SsqModelFusion()

# 回测区间: 最近25期
N_TEST = 25
n = len(df)
start = n - N_TEST

# 思路字典
strategies = ["random","freq100","freq_line","miss","bayes","group_hot","entangle",
              "extension","rebound","blue_boost","tail_law","markov","graph",
              "temporal","ml","pengpai_prior","fused"]
metrics = {s: {"top10":[], "top15":[], "avg_rank":[], "blue_hit":[]} for s in strategies}

rng = np.random.default_rng(2026)

def rank_scores(scores):
    """分数->排名(1..33)"""
    ordered = sorted(range(1,34), key=lambda x: -scores.get(x, 0))
    rk = {num: i+1 for i, num in enumerate(ordered)}
    return rk, ordered

t0 = time.time()
for idx in range(start, n):
    train = df.iloc[:idx].reset_index(drop=True)
    row = df.iloc[idx]
    target = str(row['issue'])
    actual_reds = [int(row[c]) for c in RED]
    actual_blue = int(row['blue'])

    line = rules.issue_parity(target)
    line_df = rules.split_lines(train)[line]
    if len(line_df) < 20:
        continue

    # ---- 基础统计 ----
    freq100 = {}
    for num in range(1,34):
        freq100[num] = sum(1 for _,r in train.tail(100).iterrows() if num in [int(r[c]) for c in RED])
    freq_line = {}
    for num in range(1,34):
        freq_line[num] = sum(1 for _,r in line_df.tail(50).iterrows() if num in [int(r[c]) for c in RED])
    miss = {}
    for num in range(1,34):
        m = 0
        for _, r in reversed(list(line_df.iterrows())):
            if num in [int(r[c]) for c in RED]: break
            m += 1
        miss[num] = m

    # ---- 彭湃信号 ----
    an = rules.full_analysis(train, target)
    bayes = an['bayes_probs']
    gh = an['group_hotness']
    ent = {g: v['lift'] for g, v in an['entanglement'].items()}
    ext = an['extension']
    reb = an['rebound']
    bb = an['blue_boost']
    tl = an['tail_law']
    main_t = an['main_transition']
    sub_t = set(an['sub_transitions'])
    # 彭湃先验(同线贝叶斯x规则修正)
    pp = {}
    for num in range(1,34):
        base = bayes.get(num, 1/33); f = 1.0
        g = rules.get_group(num)
        f *= (0.6+0.4*gh.get(g,1.0))
        if g in an['hot_groups']: f *= 1.25
        f *= (1+1.5*max(0,ext.get(g,0)))
        f *= (1+0.6*reb.get(num,0))
        f *= (1+0.4*bb.get(num,0))
        if num == main_t: f *= 1.5
        if num in sub_t: f *= 1.3
        cs = an['cycle_status'].get(g,'normal')
        if cs=='hot': f *= 1.2
        elif cs=='rest': f *= 0.75
        cold = tl.get(num%10,100)
        if cold>15: f *= (1+0.2*min(cold/30,0.5))
        pair = rules.get_pair(num)
        if pair:
            pb = bayes.get(pair,1/33)
            f *= (1+0.3*(pb/(1/33)-1))
        pp[num] = base*f

    # ---- 算法模型分量 ----
    fused, comps, wts = fusion.run(train, target)

    # ---- 组合各思路评分 ----
    scores = {
        "random": {x: rng.random() for x in range(1,34)},
        "freq100": freq100,
        "freq_line": freq_line,
        "miss": miss,
        "bayes": bayes,
        "group_hot": {num: gh.get(rules.get_group(num),0) for num in range(1,34)},
        "entangle": {num: ent.get(rules.get_group(num),0) for num in range(1,34)},
        "extension": {num: ext.get(rules.get_group(num),0) for num in range(1,34)},
        "rebound": reb,
        "blue_boost": bb,
        "tail_law": {num: -tl.get(num%10,100) for num in range(1,34)},  # 尾数冷度高=回补潜力, 取负让越冷越靠前? 改为: 冷尾加分
        "markov": comps.get("markov", {}),
        "graph": comps.get("graph", {}),
        "temporal": comps.get("temporal", {}),
        "ml": comps.get("ml", {}),
        "pengpai_prior": pp,
        "fused": fused,
    }
    # tail_law 改为: 尾数距今越久越应该回补(冷尾加分)
    scores["tail_law"] = {num: tl.get(num%10,100) for num in range(1,34)}

    # 主/副过渡号思路
    trans = {num: (2.0 if num==main_t else (1.0 if num in sub_t else 0.0)) for num in range(1,34)}
    scores["transition"] = trans
    strategies = list(scores.keys())

    # ---- 评估 ----
    for s, sc in scores.items():
        if not sc or max(sc.values())==min(sc.values()):
            continue
        rk, ordered = rank_scores(sc)
        m = metrics.setdefault(s, {"top10":[],"top15":[],"avg_rank":[],"blue_hit":[]})
        top10 = len(set(ordered[:10]) & set(actual_reds))
        top15 = len(set(ordered[:15]) & set(actual_reds))
        avg_rank = np.mean([rk[a] for a in actual_reds])
        m["top10"].append(top10); m["top15"].append(top15); m["avg_rank"].append(avg_rank)

    if idx % 5 == 0:
        print(f"进度 {idx-start}/{N_TEST} 期 ({time.time()-t0:.0f}s)")

print(f"\n回测完成 耗时 {time.time()-t0:.0f}s, 共评估 {len(metrics.get('fused',{}).get('top10',[]))} 期\n")

# 随机基准期望
print("="*92)
print(f"{'思路':<16}{'Top10命中':<12}{'Top15命中':<12}{'平均排名':<10}  评价")
print("="*92)
# 计算基准
import math
# 随机 Top10 期望: 6*10/33=1.818; Top15: 6*15/33=2.727; 平均排名 17
for s in ["random","freq100","freq_line","miss","bayes","group_hot","entangle",
          "extension","rebound","blue_boost","tail_law","transition",
          "markov","graph","temporal","ml","pengpai_prior","fused"]:
    if s not in metrics or not metrics[s]["top10"]:
        continue
    m = metrics[s]
    t10 = np.mean(m["top10"]); t15 = np.mean(m["top15"]); ar = np.mean(m["avg_rank"])
    delta = t10 - 1.818  # vs 随机
    verdict = "★★★ 有效" if delta > 0.25 else ("✓ 微效" if delta > 0.08 else ("— 无效" if delta > -0.08 else "✗ 负效"))
    print(f"{s:<16}{t10:<12.2f}{t15:<12.2f}{ar:<10.1f}  {verdict} (ΔTop10={delta:+.2f})")

# 蓝球评估(用频率+遗漏的简单模型)
print("\n" + "="*60)
print("蓝球命中率评估 (预测=同线期频率+遗漏加权Top)")
print("="*60)
blue_hits = {"freq_only":[], "freq_miss":[]}
for idx in range(start, n):
    train = df.iloc[:idx]
    row = df.iloc[idx]
    target = str(row['issue']); actual_blue = int(row['blue'])
    line_df = rules.split_lines(train)[rules.issue_parity(target)]
    blues = [int(r['blue']) for _, r in line_df.tail(50).iterrows()]
    bc = {}
    for b in blues: bc[b] = bc.get(b,0)+1
    top_freq = max(bc, key=bc.get)
    blue_hits["freq_only"].append(1 if top_freq==actual_blue else 0)
    # freq+miss
    missb = {}
    for b in range(1,17):
        m2=0
        for past in reversed(blues):
            if past==b: break
            m2+=1
        missb[b]=m2
    alpha=2.0; N=len(blues)
    score = {b: (bc.get(b,0)+alpha)/(N+alpha*16)*(1+0.35*min(missb[b]/25,0.8)) for b in range(1,17)}
    top = max(score, key=score.get)
    blue_hits["freq_miss"].append(1 if top==actual_blue else 0)
for k, v in blue_hits.items():
    print(f"  {k}: 命中率 {np.mean(v)*100:.1f}%  (随机基准 6.25%)")
