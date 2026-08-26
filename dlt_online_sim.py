"""
大乐透在线学习模拟引擎 (25001 -> 26097)
完善版规则: 同线频率(动态窗口) + 双线隔离 + 结构约束
在线学习: 每20期复盘, 用最近20期实测命中切换最优频率窗口
每期: 预测 -> 与实际对比 -> 记录 -> 学习
最终: 预测 26097 输出2组5+2
"""
import sys, sqlite3, json, time
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

# 补入 26096 真实开奖 (2026-08-24)
df = pd.concat([df, pd.DataFrame([{"issue":"26096","date":"2026-08-24","front01":8,"front02":9,
    "front03":10,"front04":11,"front05":25,"back01":4,"back02":12}])], ignore_index=True)
df = df.sort_values("issue").reset_index(drop=True)

dl = DualLineSystem(CFG)
WINS = [50, 80, 100, 120, 150, 180, 200]

# 定位 25001
issues = df["issue"].astype(str).tolist()
start_idx = issues.index("25001")
end_idx = len(df) - 1  # 最后一行=26096 (已开)
print(f"模拟区间: {issues[start_idx]} ~ {issues[end_idx]} ({end_idx-start_idx+1}期)")

def freq_scores(line_df, w, cols=FRONT, rng_=35):
    f = {x: 0 for x in range(1, rng_+1)}
    for _, r in line_df.tail(w).iterrows():
        for c in cols:
            f[int(r[c])] += 1
    return f

def parity(issue):
    return "single" if int(str(issue)[-1]) % 2 == 1 else "double"

def pick_group(freq, k=5, back=False):
    """按频率Top + 结构约束选k个 (防病态: 奇偶非极端, 区覆盖)"""
    nums = sorted(freq, key=lambda x: -freq[x])
    if back:
        return sorted(nums[:k])
    # 前区结构约束: 奇偶比 2:3/3:2 优先, 四区覆盖>=3
    best = None
    for combo_candidates in [nums, nums]:
        pass
    # 贪心: 依次选, 保证最终不极端
    selected = []
    for n in nums:
        if n in selected: continue
        selected.append(n)
        if len(selected) == k: break
    # 调整奇偶
    odds = [n for n in selected if n % 2 == 1]
    evens = [n for n in selected if n % 2 == 0]
    if len(odds) >= 4 or len(evens) >= 4:
        # 极端, 用剩余号码替换
        target_odd = 2 if len(odds) > len(evens) else 3
        # 简化: 从候选中重选满足 2奇3偶 或 3奇2偶
        res = pick_balanced(nums, k)
        return res
    return sorted(selected)

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
            # 检查四区覆盖
            zones = set(1 if x<=9 else 2 if x<=18 else 3 if x<=27 else 4 for x in res)
            if len(zones) >= 3:
                return sorted(res)
    return sorted(nums[:k])

# 在线学习
records = []
current_window = 150
window_history = []
best_window_log = []

t0 = time.time()
for idx in range(start_idx, end_idx + 1):
    train = df.iloc[:idx].reset_index(drop=True)
    row = df.iloc[idx]
    target = str(row['issue'])
    actual_f = set(int(row[c]) for c in FRONT)
    actual_b = set(int(row[c]) for c in BACK)
    line = parity(target)
    line_df = dl.split(train)[line]

    # 当前窗口预测
    w = current_window
    f_f = freq_scores(line_df, w, FRONT, 35)
    f_b = freq_scores(line_df, min(w, 80), BACK, 12)
    pred10 = sorted(f_f, key=lambda x: -f_f[x])[:10]
    pred15 = sorted(f_f, key=lambda x: -f_f[x])[:15]
    group = pick_group(f_f, 5)
    back_grp = pick_group(f_b, 2, back=True)

    hit10 = len(set(pred10) & actual_f)
    hit15 = len(set(pred15) & actual_f)
    hit5 = len(set(group) & actual_f)
    hit_b = len(set(back_grp) & actual_b)
    records.append({"issue": target, "win": w, "hit10": hit10, "hit15": hit15,
                    "hit5": hit5, "hit_b": hit_b, "group": group, "back": back_grp,
                    "actual_f": sorted(actual_f), "actual_b": sorted(actual_b)})

    # 每20期复盘: 用最近20期各窗口重算命中, 切最优窗口
    if (idx - start_idx + 1) % 20 == 0:
        recent = records[-20:]
        win_scores = {}
        for cand in WINS:
            s = 0
            for rec in recent:
                ti = issues.index(rec["issue"])
                tr = df.iloc[:ti]
                rw = df.iloc[ti]
                ln = parity(rec["issue"])
                ld = dl.split(tr)[ln]
                ff = freq_scores(ld, cand, FRONT, 35)
                p10 = set(sorted(ff, key=lambda x: -ff[x])[:10])
                s += len(p10 & set(int(rw[c]) for c in FRONT))
            win_scores[cand] = s
        best_w = max(win_scores, key=win_scores.get)
        if best_w != current_window:
            window_history.append({"at": target, "from": current_window, "to": best_w,
                                   "scores": win_scores})
            current_window = best_w
        best_window_log.append({"issue": target, "best": best_w, "cur": current_window})

    if (idx - start_idx + 1) % 100 == 0:
        print(f"进度 {idx-start_idx+1}/{end_idx-start_idx+1} ({time.time()-t0:.0f}s) 窗口={current_window}")

# ============ 26097 预测 ============
train_final = df.iloc[:len(df)].reset_index(drop=True)
target_26097 = "26097"
line = parity(target_26097)
line_df = dl.split(train_final)[line]
f_f = freq_scores(line_df, current_window, FRONT, 35)
f_b = freq_scores(line_df, min(current_window, 80), BACK, 12)
top10 = sorted(f_f, key=lambda x: -f_f[x])[:10]
group_A = pick_group(f_f, 5)
# B组: 频率 + 次级彭湃信号(3阶返点/遗漏回补/上期重号) 综合, 与A组差异化
psets = [set(int(r[c]) for c in FRONT) for _, r in line_df.iterrows()]
miss = {x: 0 for x in range(1, 36)}
for x in range(1, 36):
    m = 0
    for s in reversed(psets):
        if x in s:
            break
        m += 1
    miss[x] = m
stage3 = set(psets[-3]) if len(psets) >= 3 else set()
last_period = set(psets[-1]) if psets else set()
# 次级评分 (仅作为B组补充信号, 不扰动A组)
b_sec = {}
fmax = max(f_f.values()) if f_f else 1
for x in range(1, 36):
    s = f_f[x] / fmax
    if x in stage3: s += 0.12
    if miss[x] >= 11: s += 0.10 * min((miss[x] - 11) / 15, 1)
    if x in last_period: s += 0.08
    b_sec[x] = s
group_B = pick_balanced(sorted(b_sec, key=lambda x: -b_sec[x]), 5)
# 确保B组与A组至少差2个号
if len(set(group_A) & set(group_B)) >= 4:
    alt = [x for x in sorted(b_sec, key=lambda x: -b_sec[x]) if x not in group_A]
    extra = alt[0] if alt else None
    if extra:
        common = [x for x in group_B if x in group_A]
        group_B = [x for x in group_B if x not in common[:1]] + [extra]
        group_B = sorted(group_B)
back_grp = pick_group(f_b, 2, back=True)

# 汇总统计
dfr = pd.DataFrame(records)
print("\n" + "="*70)
print(f"在线学习模拟完成 {len(records)}期 (25001~26096), 耗时{time.time()-t0:.0f}s")
print("="*70)
print(f"Top10平均命中 : {dfr['hit10'].mean():.2f}  (随机1.43)")
print(f"Top15平均命中 : {dfr['hit15'].mean():.2f}  (随机2.14)")
print(f"5+2组合前区命中: {dfr['hit5'].mean():.2f}")
print(f"后区命中      : {dfr['hit_b'].mean():.2f}  (随机0.38)")
print(f"窗口切换次数  : {len(window_history)}")
print(f"最终学习窗口  : {current_window}")
# 阶段命中趋势
dfr['seg'] = (np.arange(len(dfr)) // 200) + 1
print("\n分段Top10命中 (每200期):")
for seg, g in dfr.groupby('seg'):
    print(f"  段{int(seg)}: Top10={g['hit10'].mean():.2f} Top15={g['hit15'].mean():.2f}")

# 26097 输出
print("\n" + "="*70)
print("★ 26097期模拟预测组合 (完善版在线学习引擎)")
print("="*70)
print(f"当前最优窗口: {current_window}期 (同线)")
print(f"前区Top10: {' '.join(f'{x:02d}' for x in top10)}")
print(f"后区Top:   {' '.join(f'{x:02d}' for x in sorted(f_b, key=lambda x:-f_b[x])[:6])}")
print(f"\n【A组】模型共识组: 前区 {' '.join(f'{x:02d}' for x in group_A)}  后区 {' '.join(f'{x:02d}' for x in back_grp)}")
print(f"      结构: 奇偶 {sum(1 for x in group_A if x%2==1)}:{sum(1 for x in group_A if x%2==0)}"
      f"  四区 {[sum(1 for x in group_A if 1<=x<=9),sum(1 for x in group_A if 10<=x<=18),sum(1 for x in group_A if 19<=x<=27),sum(1 for x in group_A if 28<=x<=35)]}"
      f"  和值 {sum(group_A)}")
print(f"【B组】彭湃强化组: 前区 {' '.join(f'{x:02d}' for x in group_B)}  后区 {' '.join(f'{x:02d}' for x in back_grp)}")
print(f"      结构: 奇偶 {sum(1 for x in group_B if x%2==1)}:{sum(1 for x in group_B if x%2==0)}"
      f"  四区 {[sum(1 for x in group_B if 1<=x<=9),sum(1 for x in group_B if 10<=x<=18),sum(1 for x in group_B if 19<=x<=27),sum(1 for x in group_B if 28<=x<=35)]}"
      f"  和值 {sum(group_B)}")

# 保存
out = {
    "records": records,
    "window_switches": window_history,
    "final_window": current_window,
    "predict_26097": {"A": group_A, "B": group_B, "back": back_grp, "top10": top10}
}
json.dump(out, open(r'E:\PMSF-V1\docs\data\dlt_online_sim.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print("\n已保存 docs/data/dlt_online_sim.json")
