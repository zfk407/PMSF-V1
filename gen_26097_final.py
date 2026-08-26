"""
26097 最终生成 (全量数据07001~26096 + 优化v2配置)
前区: 同线动态窗口(80)频率 Top + 结构约束 -> A/B组
后区: 同线150期频率 Top2 (优化后最优0.362)
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
print(f"全量数据: {len(df)}期 (07001~26096)")

def parity(issue):
    return "single" if int(str(issue)[-1]) % 2 == 1 else "double"

def freq_scores(line_df, w, cols, rng_):
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

# 26097 预测 (双线: 26097奇=单线)
target = "26097"
line = parity(target)
line_df = dl.split(df)[line]
print(f"目标 {target}, 线型: {line}, 同线历史: {len(line_df)}期")

# 前区: 动态窗口80 (在线学习收敛值)
f_f = freq_scores(line_df, 80, FRONT, 35)
# 后区: 同线150 (优化后最优)
f_b = freq_scores(line_df, 150, BACK, 12)

top10 = sorted(f_f, key=lambda x: -f_f[x])[:10]
back_top = sorted(f_b, key=lambda x: -f_b[x])[:6]

# A组: 纯频率+结构约束
group_A = pick_balanced(sorted(f_f, key=lambda x: -f_f[x]), 5)
# B组: 频率+次级信号差异化
psets = [set(int(r[c]) for c in FRONT) for _, r in line_df.iterrows()]
miss = {x: 0 for x in range(1, 36)}
for x in range(1, 36):
    m = 0
    for s in reversed(psets):
        if x in s: break
        m += 1
    miss[x] = m
stage3 = set(psets[-3]) if len(psets) >= 3 else set()
last_p = set(psets[-1]) if psets else set()
fmax = max(f_f.values()) if f_f else 1
b_sec = {}
for x in range(1, 36):
    s = f_f[x] / fmax
    if x in stage3: s += 0.12
    if miss[x] >= 11: s += 0.10 * min((miss[x]-11)/15, 1)
    if x in last_p: s += 0.08
    b_sec[x] = s
group_B = pick_balanced(sorted(b_sec, key=lambda x: -b_sec[x]), 5)
if len(set(group_A) & set(group_B)) >= 4:
    alt = [x for x in sorted(b_sec, key=lambda x: -b_sec[x]) if x not in group_A]
    if alt:
        common = [x for x in group_B if x in group_A]
        group_B = sorted([x for x in group_B if x not in common[:1]] + [alt[0]])

back_grp = sorted(f_b, key=lambda x: -f_b[x])[:2]

def zone_info(g):
    return [sum(1 for x in g if 1<=x<=9), sum(1 for x in g if 10<=x<=18),
            sum(1 for x in g if 19<=x<=27), sum(1 for x in g if 28<=x<=35)]

print("\n" + "="*72)
print("★ 26097期 最终模拟组合 (全量数据+优化v2)")
print("="*72)
print(f"前区Top10: {' '.join(f'{x:02d}' for x in top10)}")
print(f"后区Top6 : {' '.join(f'{x:02d}' for x in back_top)}")
print()
print(f"【A组】模型共识: 前区 {' '.join(f'{x:02d}' for x in group_A)}  后区 {' '.join(f'{x:02d}' for x in back_grp)}")
print(f"      结构: 奇偶 {sum(1 for x in group_A if x%2==1)}:{sum(1 for x in group_A if x%2==0)}"
      f" | 四区{zone_info(group_A)} | 和值{sum(group_A)} | 跨度{max(group_A)-min(group_A)}")
print(f"【B组】彭湃强化: 前区 {' '.join(f'{x:02d}' for x in group_B)}  后区 {' '.join(f'{x:02d}' for x in back_grp)}")
print(f"      结构: 奇偶 {sum(1 for x in group_B if x%2==1)}:{sum(1 for x in group_B if x%2==0)}"
      f" | 四区{zone_info(group_B)} | 和值{sum(group_B)} | 跨度{max(group_B)-min(group_B)}")
print(f"\n重叠度 A∩B: {sorted(set(group_A)&set(group_B))}")

out = {"issue":"26097","line":line,"front_window":80,"back_window":150,
       "A": {"front":group_A,"back":back_grp},"B":{"front":group_B,"back":back_grp},
       "top10":top10,"back_top":back_top}
json.dump(out, open(r'E:\PMSF-V1\docs\data\dlt_26097_final.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print("\n已保存 docs/data/dlt_26097_final.json")
