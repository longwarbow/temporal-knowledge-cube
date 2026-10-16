# -*- coding: utf-8 -*-
"""
E1: 增量维护 vs 全量重建  （RQ1 问题 1 / P1，在 CED 传播树上）

验证目标
  A. 正确性：增量结果 ≡ 全量重建（骨架层精确等价，逐 cell 比对）
  B. 代价：增量 O(新增) << 全量 O(N)
  C. 鲁棒性：事件时间序 ≡ 打乱到达序（乱序容忍）
  D. 撤回：RETRACT(根帖) 的失效影响集 = 其传播级联（P2 最小重算原型）

输出：E1_报告.md + e1_metrics.json
"""
import os, io, json, time as _time, random, collections, datetime

BASE = r"D:\WSPforWB\10.keyan\场景1-海量短文本\数据\CED\Chinese_Rumor_Dataset-master\CED_Dataset"
RUM  = os.path.join(BASE, "rumor-repost")
NON  = os.path.join(BASE, "non-rumor-repost")
ORIG = os.path.join(BASE, "original-microblog")
OUT  = r"D:\WSPforWB\10.keyan\场景1-海量短文本\实验\E1_增量vs全量重建"
os.makedirs(OUT, exist_ok=True)

LOG = []
def p(x=""):
    LOG.append(str(x))

def parse_dt(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

# ---------------- 1. 解析 CED -> 统一事件流 ----------------
T0 = _time.time()
rum_files = set(os.listdir(RUM))
non_files = set(os.listdir(NON))
orig_files = sorted(os.listdir(ORIG))

events = []                       # (t, uid, lvl, cascade)
level_hist = collections.Counter()
n_seed = n_repost = 0
seed_missing_t = 0
bad = 0
for idx, fn in enumerate(orig_files):
    op = os.path.join(ORIG, fn)
    try:
        o = json.load(io.open(op, encoding="utf-8"))
    except Exception:
        bad += 1; continue
    if isinstance(o, list):
        o = o[0] if o else {}
    if not isinstance(o, dict):
        bad += 1; continue
    parts = fn[:-5].split("_")
    seed_mid = parts[1] if len(parts) >= 2 else fn
    label_dir = RUM if fn in rum_files else (NON if fn in non_files else None)
    if label_dir is None:
        bad += 1; continue
    # 转发
    arr = []
    rp = os.path.join(label_dir, fn)
    if os.path.exists(rp):
        try:
            arr = json.load(io.open(rp, encoding="utf-8"))
        except Exception:
            arr = []
    if not isinstance(arr, list):
        arr = []
    # 层级（parent=='' 或指向集合外 = 根 = 一级转发）
    nodes = {}
    for r in arr:
        if isinstance(r, dict) and r.get("mid"):
            nodes[r["mid"]] = r
    children = collections.defaultdict(list)
    roots = []
    for mid, r in nodes.items():
        par = r.get("parent")
        if not par or par not in nodes:
            roots.append(mid)
        else:
            children[par].append(mid)
    lvl = {}
    dq = collections.deque()
    for r in roots:
        lvl[r] = 1; dq.append(r)
    while dq:
        x = dq.popleft()
        for c in children.get(x, []):
            if c not in lvl:
                lvl[c] = lvl[x] + 1; dq.append(c)
    rep_times = []
    rep_events = []
    for r in arr:
        if not isinstance(r, dict):
            continue
        d = parse_dt(r.get("date") or "")
        if not d:
            continue
        l = lvl.get(r.get("mid"), 1)
        rep_events.append((d, r.get("uid"), l, seed_mid))
        rep_times.append(d)
        level_hist[min(l, 3)] += 1
    # 种子事件（A1：时间缺失时取最早转发时间作下界）
    tseed = o.get("time")
    if isinstance(tseed, (int, float)) and tseed > 0:
        st = datetime.datetime.fromtimestamp(tseed)
    elif rep_times:
        st = min(rep_times); seed_missing_t += 1
    else:
        st = None
    if st is not None:
        seed_uid = parts[2] if len(parts) >= 3 else None
        events.append((st, seed_uid, 0, seed_mid))
        n_seed += 1
    events.extend(rep_events)
    n_repost += len(rep_events)
    if (idx + 1) % 800 == 0:
        p("  解析进度 %d/%d ｜ 事件 %d" % (idx + 1, len(orig_files), len(events)))

events.sort(key=lambda e: e[0])
N = len(events)
p("")
p("## 1. 数据与流")
p("")
p("- CED 原帖 %d ｜ 转发 %d ｜ **统一事件流 %d** ｜ 解析失败 %d" % (len(orig_files), n_repost, N, bad))
p("- 种子事件 %d（其中事件时间缺失、走 A1 回退 %d）" % (n_seed, seed_missing_t))
p("- 传播层级分布（1 / 2 / 3+）：%s" % ", ".join("%s:%d" % (k, v) for k, v in sorted(level_hist.items())))
p("- 事件时间跨度：%s ~ %s" % (events[0][0].strftime("%Y-%m-%d"), events[-1][0].strftime("%Y-%m-%d")))
p("- 解析+排序用时：%.1fs" % (_time.time() - T0))

# 预展开为并行数组（提速）
D   = [e[0].strftime("%Y-%m-%d") for e in events]
M   = [e[0].strftime("%Y-%m")   for e in events]
LVL = [e[2] for e in events]
UID = [e[1] for e in events]
CAS = [e[3] for e in events]
BK  = ["seed" if l <= 0 else ("L1" if l == 1 else ("L2" if l == 2 else "L3+")) for l in LVL]

# ---------------- 2. 立方体定义 ----------------
# 4 个 cuboid：时间维（日 ⊂ 月 ⊂ 全）× 传播层级（级 ⊂ 全）
CUBOIDS = ["day_level", "month_level", "month_all", "all_all"]
def cells_of(i):
    b = BK[i]
    return (
        ("day_level",   (D[i], b)),
        ("month_level", (M[i], b)),
        ("month_all",   (M[i], "ALL")),
        ("all_all",     ("ALL", "ALL")),
    )

def new_cube():
    return {name: {} for name in CUBOIDS}

def upd(cube, i, sign=1):
    lv = LVL[i]; u = UID[i]
    for name, key in cells_of(i):
        d = cube[name]
        c = d.get(key)
        if c is None:
            c = [0, 0, {}]          # count, sum_level, uid->freq
            d[key] = c
        c[0] += sign
        c[1] += sign * lv
        if u is not None:
            f = c[2].get(u, 0) + sign
            if f > 0:
                c[2][u] = f
            elif u in c[2]:
                del c[2][u]

def rebuild(lo, hi):
    cube = new_cube()
    for i in range(lo, hi):
        upd(cube, i, 1)
    return cube

def cube_ops(cube):
    """(cells, relations) —— 关系数 = Σ(1 + uid数)"""
    cells = sum(len(cube[n]) for n in CUBOIDS)
    rels = 0
    for n in CUBOIDS:
        for v in cube[n].values():
            rels += 1 + len(v[2])
    return cells, rels

def compare(a, b):
    """返回不一致 cell 数与明细（上限 5 条）"""
    diffs = 0; detail = []
    for name in CUBOIDS:
        keys = set(a[name]) | set(b[name])
        for k in keys:
            ca = a[name].get(k); cb = b[name].get(k)
            va = (ca[0], ca[1], len(ca[2])) if ca else (0, 0, 0)
            vb = (cb[0], cb[1], len(cb[2])) if cb else (0, 0, 0)
            if va != vb:
                diffs += 1
                if len(detail) < 5:
                    detail.append((name, k, va, vb))
    return diffs, detail

# ---------------- 3. 增量 vs 全量：逐检查点比对 ----------------
p("")
p("## 2. 增量维护 ≡ 全量重建（逐检查点精确比对）")
p("")
p("| # | 前缀事件数 | 本次新增 | 增量耗时(s) | 全量重建(s) | 基线等价? | 增量 cells | 全量 cells |")
p("|---|---|---|---|---|---|---|---|")

fracs = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
ckpts = sorted(set(int(N * f) for f in fracs))
ckpts = [c for c in ckpts if c > 0]

inc = new_cube()
i = 0
prev = 0
rows = []
inc_ops = 0
reb_ops = 0
t_inc_total = 0.0
t_reb_total = 0.0
for k, c in enumerate(ckpts, 1):
    ti = _time.time()
    for j in range(i, c):
        upd(inc, j, 1)
    dt_inc = _time.time() - ti
    i = c
    tr = _time.time()
    rb = rebuild(0, c)
    dt_reb = _time.time() - tr
    diffs, _ = compare(inc, rb)
    icells, _ = cube_ops(inc)
    rcells, _ = cube_ops(rb)
    inc_ops += (c - prev) * len(CUBOIDS)
    reb_ops += c * len(CUBOIDS)
    t_inc_total += dt_inc
    t_reb_total += dt_reb
    prev = c
    rows.append((k, c, dt_inc, dt_reb, diffs, icells))
    p("| %d | %d | %d | %.3f | %.3f | %s | %d | %d |" % (
        k, c, c - (ckpts[k - 2] if k > 1 else 0), dt_inc, dt_reb,
        ("✅ 0 差异" if diffs == 0 else "❌ %d" % diffs), icells, rcells))

all_zero = all(r[4] == 0 for r in rows)
p("")
p("**结论 A**：%s（%d 个检查点全部逐 cell 精确相等）" % (
    "增量结果 ≡ 全量重建" if all_zero else "存在差异，需排查", len(rows)))

# ---------------- 4. 代价对比 ----------------
p("")
p("## 3. 维护代价（操作数计）")
p("")
p("- 增量：全流一次通过 = %d 事件 × %d cuboid = **%d 次单元更新**" % (N, len(CUBOIDS), inc_ops))
p("- 全量：%d 个检查点各重建一次 = Σ 前缀 = **%d 次单元更新**" % (len(ckpts), reb_ops))
p("- **累积代价比（重建/增量）= %.1f×**" % (reb_ops / max(1, inc_ops)))
last = ckpts[-1]
seg = last - ckpts[-2]
p("- 单看「增量维护最新 %d 条(%.0f%%)」：增量 %d 次 vs 重建 %d 次 = **%.1f×**" % (
    seg, 100.0 * seg / N, seg * len(CUBOIDS), last * len(CUBOIDS),
    (last * len(CUBOIDS)) / max(1, seg * len(CUBOIDS))))
p("- 实测耗时：增量累计 %.2fs ｜ 重建累计 %.2fs（比值 %.1f×）" % (
    t_inc_total, t_reb_total, t_reb_total / max(1e-9, t_inc_total)))

# ---------------- 5. 乱序鲁棒性 ----------------
p("")
p("## 4. 乱序到达鲁棒性")
p("")
order = list(range(N))
random.Random(42).shuffle(order)
sh = new_cube()
for j in order:
    upd(sh, j, 1)
diffs_sh, _ = compare(sh, inc)
p("- 将 %d 条事件**完全打乱**后重放：与事件时间序结果比对 → **%s**" % (
    N, "✅ 0 差异" if diffs_sh == 0 else "❌ %d 差异" % diffs_sh))
p("- 意义：聚合量（count / Σlevel / distinct-uid）在**单元层可交换**，故增量维护对到达顺序不敏感；")
p("  但**时间窗内的归属**仍依赖事件时间（双时态的必要性见 RQ1 §3）。")

# ---------------- 6. 撤回（P2 原型） ----------------
p("")
p("## 5. RETRACT 失效影响集 = 其传播级联（P2 最小重算原型）")
p("")
by_cascade = collections.Counter(CAS)
top_seed, top_n = by_cascade.most_common(1)[0]
p("- 选中最大级联的种子帖 `%s`：含转发 **%d** 条（占全流 %.2f%%）" % (
    top_seed, top_n, 100.0 * top_n / N))

# 受影响 cell 集合（按 cuboid 分组；增量只动这些）
affected_by_cub = {n: set() for n in CUBOIDS}
affected_events = []
for j in range(N):
    if CAS[j] == top_seed:
        affected_events.append(j)
        for name, key in cells_of(j):
            affected_by_cub[name].add(key)

# 写时复制：先把受影响 cell 复制出来，再逐事件 -1（未受影响 cell 与 inc 共享，未改动）
inc_after = {n: dict(inc[n]) for n in CUBOIDS}
for name in CUBOIDS:
    for key in affected_by_cub[name]:
        c = inc[name].get(key)
        inc_after[name][key] = [c[0], c[1], dict(c[2])] if c else [0, 0, {}]
for j in affected_events:
    for name, key in cells_of(j):
        c = inc_after[name][key]
        c[0] -= 1; c[1] -= LVL[j]
        u = UID[j]
        if u is not None:
            f = c[2].get(u, 0) - 1
            if f > 0:
                c[2][u] = f
            elif u in c[2]:
                del c[2][u]

# 全量对照：从所有「非该级联」事件重建
rb_after = new_cube()
for j in range(N):
    if CAS[j] != top_seed:
        upd(rb_after, j, 1)
diffs_r, det_r = compare(inc_after, rb_after)
affected_keys = set().union(*affected_by_cub.values())

total_cells, _ = cube_ops(inc)
p("- 全立方体 cells = **%d** ｜ 受影响 cells = **%d**（%.2f%%）" % (
    total_cells, len(affected_keys), 100.0 * len(affected_keys) / max(1, total_cells)))
p("- 撤回后「增量（只动受影响 cell）」vs「全量重建」：**%s**" % (
    "✅ 0 差异" if diffs_r == 0 else "❌ %d 差异 %s" % (diffs_r, det_r)))
p("- 代价：增量 = O(受影响) = %d 次更新 ｜ 朴素全量 = O(N) = %d 次更新 = **%.0f×**" % (
    top_n * len(CUBOIDS), N * len(CUBOIDS), N / max(1, top_n)))
p("")
p("> 说明：`distinct-uid` 采用**精确频次表**（uid→freq）实现，故撤回可**精确删减**（退化为 0 即移除），")
p("> 对应 RQ1 §4.3 问题 2 中「表示粒度决定最小重算单元」——精确表 → 单条可撤。")

# ---------------- 7. 汇总 ----------------
p("")
p("## 6. 汇总")
p("")
p("| 目标 | 结果 |")
p("|---|---|")
p("| A 增量 ≡ 全量（骨架层） | %s |" % ("✅ 全部 %d 个检查点 0 差异" % len(rows) if all_zero else "❌ 有差异"))
p("| B 代价 | 累积 **%.1f×**（重建/增量）；维护新增段 **%.1f×** |" % (
    reb_ops / max(1, inc_ops), (last * len(CUBOIDS)) / max(1, seg * len(CUBOIDS))))
p("| C 乱序鲁棒 | %s |" % ("✅ 0 差异" if diffs_sh == 0 else "❌ 有差异"))
p("| D 撤回失效集 | %s ；影响 **%.2f%%** cells |" % (
    "✅ 0 差异" if diffs_r == 0 else "❌ 有差异", 100.0 * len(affected_keys) / max(1, total_cells)))
p("")
p("**下一步**：① 把 `day_level` 换成**滑动窗**后重测（增量遇到迟到的历史事件时的重算代价）；")
p("② 注入**维度演化**（SPLIT/MERGE/REPARENT）—— 即 RQ1 §6.3.1 的「维度驱动的回溯重分区」。")

report = "\n".join(LOG)
io.open(os.path.join(OUT, "E1_报告.md"), "w", encoding="utf-8").write(
    "# E1 报告：增量维护 vs 全量重建（CED 传播树）\n\n"
    "> 生成：WorkBuddy ｜ 2026-10-09 ｜ RQ1 问题 1（P1）首个真实数据实验\n\n" + report + "\n")

metrics = {
    "events_total": N,
    "seeds": n_seed, "reposts": n_repost,
    "cuboids": CUBOIDS,
    "checkpoints": [{"k": r[0], "prefix": r[1], "t_inc": r[2], "t_rebuild": r[3],
                     "diffs": r[4], "inc_cells": r[5]} for r in rows],
    "inc_ops": inc_ops, "rebuild_ops": reb_ops,
    "cost_ratio_cumulative": reb_ops / max(1, inc_ops),
    "out_of_order_diffs": diffs_sh,
    "retract": {"seed": top_seed, "cascade_size": top_n,
                "total_cells": total_cells, "affected_cells": len(affected_keys),
                "diffs": diffs_r},
    "all_equivalent": all_zero,
}
io.open(os.path.join(OUT, "e1_metrics.json"), "w", encoding="utf-8").write(
    json.dumps(metrics, ensure_ascii=False, indent=2))

print("DONE ->", OUT)
