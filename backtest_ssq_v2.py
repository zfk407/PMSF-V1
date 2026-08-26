"""
双色球逐思路回测 v2 (40期, 更大样本验证稳健性)
专注: 彭湃各规则因子 + 简单统计 + 组合
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, r'E:\PMSF-V1')
from src.ssq.data import SsqDataFetcher
from src.ssq.rules import SsqPengpaiRules

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

N_TEST = 40
n = len(df)
start = n - N_TEST
rng = np.random.default_rng(7)

def rank_scores(scores):
    ordered = sorted(range(1,34), key=lambda x: -scores.get(x,0))
    rk = {num:i+1 for i,num in enumerate(ordered)}
    return rk, ordered

metrics = {}
def add(name, t10, t15, ar):
    metrics.setdefault(name, {"top10":[],"top15":[],"avg_rank":[]})
    metrics[name]["top10"].append(t10); metrics[name]["top15"].append(t15)
    metrics[name]["avg_rank"].append(ar)

t0 = time.time()
for idx in range(start, n):
    train = df.iloc[:idx].reset_index(drop=True)
    row = df.iloc[idx]
    target = str(row['issue'])
    actual = [int(row[c]) for c in RED]

    line = rules.issue_parity(target)
    line_df = rules.split_lines(train)[line]
    if len(line_df) < 25: continue

    # 统计
    freq100 = {num:0 for num in range(1,34)}
    for _,r in train.tail(100).iterrows():
        for c in RED: freq100[int(r[c])]+=1
    freq_line = {num:0 for num in range(1,34)}
    for _,r in line_df.tail(50).iterrows():
        for c in RED: freq_line[int(r[c])]+=1
    miss = {}
    for num in range(1,34):
        m=0
        for _,r in reversed(list(line_df.iterrows())):
            if num in [int(r[c]) for c in RED]: break
            m+=1
        miss[num]=m

    an = rules.full_analysis(train, target)
    gh=an['group_hotness']; ent={g:v['lift'] for g,v in an['entanglement'].items()}
    ext=an['extension']; reb=an['rebound']; bb=an['blue_boost']; tl=an['tail_law']
    main_t=an['main_transition']; sub_t=set(an['sub_transitions'])
    bayes=an['bayes_probs']

    # 彭湃先验
    pp={}
    for num in range(1,34):
        base=bayes.get(num,1/33); f=1.0; g=rules.get_group(num)
        f*=(0.6+0.4*gh.get(g,1.0))
        if g in an['hot_groups']: f*=1.25
        f*=(1+1.5*max(0,ext.get(g,0)))
        f*=(1+0.6*reb.get(num,0))
        f*=(1+0.4*bb.get(num,0))
        if num==main_t: f*=1.5
        if num in sub_t: f*=1.3
        cs=an['cycle_status'].get(g,'normal')
        if cs=='hot': f*=1.2
        elif cs=='rest': f*=0.75
        cold=tl.get(num%10,100)
        if cold>15: f*=(1+0.2*min(cold/30,0.5))
        pair=rules.get_pair(num)
        if pair:
            pb=bayes.get(pair,1/33); f*=(1+0.3*(pb/(1/33)-1))
        pp[num]=base*f

    # 简单组合: 频率+遗漏+返点+蓝补红+过渡号
    combo = {num: freq_line[num]/6.0 + miss[num]/40.0 + reb.get(num,0)*1.5 + bb.get(num,0)*0.8
             + (1.5 if num==main_t else (0.8 if num in sub_t else 0.0)) for num in range(1,34)}

    scores = {
        "random": {num: rng.random() for num in range(1,34)},
        "freq100": freq100, "freq_line": freq_line, "miss": miss, "bayes": bayes,
        "group_hot": {num: gh.get(rules.get_group(num),0) for num in range(1,34)},
        "entangle": {num: ent.get(rules.get_group(num),0) for num in range(1,34)},
        "extension": {num: ext.get(rules.get_group(num),0) for num in range(1,34)},
        "rebound": reb, "blue_boost": bb,
        "tail_law": {num: tl.get(num%10,100) for num in range(1,34)},
        "transition": {num: (2.0 if num==main_t else (1.0 if num in sub_t else 0.0)) for num in range(1,34)},
        "pengpai_prior": pp, "combo": combo,
    }
    for name, sc in scores.items():
        if not sc or max(sc.values())==min(sc.values()): continue
        rk, ordered = rank_scores(sc)
        add(name, len(set(ordered[:10])&set(actual)), len(set(ordered[:15])&set(actual)), np.mean([rk[a] for a in actual]))
    if (idx-start)%10==0:
        print(f"进度 {(idx-start)}/{N_TEST} ({time.time()-t0:.0f}s)")

print(f"\n完成 耗时{time.time()-t0:.0f}s, 共{len(metrics.get('fused',metrics.get('combo',{}).get('top10',[])))}期\n")
print("="*90)
print(f"{'思路':<14}{'Top10':<9}{'Top15':<9}{'平均排名':<9}  ΔTop10 vs理论(1.82)")
print("="*90)
for name in ["random","freq100","freq_line","miss","bayes","group_hot","entangle",
             "extension","rebound","blue_boost","tail_law","transition","pengpai_prior","combo"]:
    if name not in metrics or not metrics[name]["top10"]: continue
    m=metrics[name]
    t10=np.mean(m["top10"]); t15=np.mean(m["top15"]); ar=np.mean(m["avg_rank"])
    delta=t10-1.82
    verdict="★★★" if delta>0.25 else ("✓" if delta>0.08 else ("—" if delta>-0.08 else "✗"))
    print(f"{name:<14}{t10:<9.2f}{t15:<9.2f}{ar:<9.1f}  {delta:+.2f} {verdict}")
