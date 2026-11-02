# -*- coding: utf-8 -*-
"""
E1-EVO: 维度演化注入（MERGE / REPARENT / SPLIT）—— RQ1 §6.3.1「维度驱动的回溯重分区」原型

背景
  E1 已证 P1（增量 ≡ 全量）与 P2 原型（RETRACT 失效集 = 传播级联）。
  本实验补上第三块：**维度本身演化**时，增量维护如何只动受影响 cell。

设计
  在 E1 立方体上加一个「事件簇维」（cluster）：初始 = 每个种子帖自成一簇（BIRTH 语义）。
  随后注入三类结构操作（模拟话题维的 L3 事件簇演化）：
    MERGE    : 两簇合并（= 两个 L3 事件合并进同一 L2 议题）→ 旧簇 cell 全部重挂
    REPARENT : 某簇改挂到另一簇（= 叙事从一个事件子簇改挂另一簇，D18 的话题维实例）
    SPLIT    : 大簇拆出一半成员另立新簇（= 事件分裂出新叙事）
  每类操作后：
    - 增量 = 只重算「受影响 cell」（涉及被移动事件的 cell）
    - 全量 = 从头重建对照
    - 比对正确性（0 差异）+ 代价（操作数比）

输出：E1_EVO_报告.md + e1_evo_metrics.json
"""

# --- path resolution (released artifact) -----------------------------------
# No absolute path is baked in.  KB_ROOT points at the directory holding the
# experiment folders; it defaults to this repository's root and can be
# overridden with an environment variable.
import os as _os
_KB_ROOT = _os.environ.get(
    "KB_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)


def _KB(*parts):
    """Join *parts* onto the knowledge-base root."""
    return _os.path.join(_KB_ROOT, *parts)
# ---------------------------------------------------------------------------
import os, io, json, time as _time, random, collections, datetime

BASE = _KB('数据', 'CED', 'Chinese_Rumor_Dataset-master', 'CED_Dataset')
RUM  = os.path.join(BASE, "rumor-repost")
NON  = os.path.join(BASE, "non-rumor-repost")
ORIG = os.path.join(BASE, "original-microblog")
OUT  = _KB('experiments', 'E1_维度演化注入')
os.makedirs(OUT, exist_ok=True)

LOG = []
def p(x=""):
    LOG.append(str(x))

def parse_dt(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

# ---------------- 1. 解析 CED（同 E1 口径） ----------------
T0 = _time.time()
rum_files = set(os.listdir(RUM))
non_files = set(os.listdir(NON))
orig_files = sorted(os.listdir(ORIG))

events = []   # (dt, uid, lvl, cascade)
for fn in orig_files:
    try:
        o = json.load(io.open(os.path.join(ORIG, fn), encoding="utf-8"))
    except Exception:
        continue
    if isinstance(o, list):
        o = o[0] if o else {}
    if not isinstance(o, dict):
        continue
    label_dir = RUM if fn in rum_files else (NON if fn in non_files else None)
    if label_dir is None:
        continue
    seed_mid = fn[:-5].split("_")[1] if "_" in fn[:-5] else fn
    arr = []
    rp = os.path.join(label_dir, fn)
    if os.path.exists(rp):
        try:
            arr = json.load(io.open(rp, encoding="utf-8"))
        except Exception:
            arr = []
    if not isinstance(arr, list):
        arr = []
    tseed = o.get("time")
    if isinstance(tseed, (int, float)) and tseed > 0:
        st = datetime.datetime.fromtimestamp(tseed)
    else:
        rt = [parse_dt(r.get("date") or "") for r in arr if isinstance(r, dict)]
        rt = [x for x in rt if x]
        st = min(rt) if rt else None
    if st is not None:
        events.append((st, None, 0, seed_mid))
    for r in arr:
        if isinstance(r, dict):
            d = parse_dt(r.get("date") or "")
            if d:
                events.append((d, r.get("uid"), 1, seed_mid))

events.sort(key=lambda e: e[0])
N = len(events)
D   = [e[0].strftime("%Y-%m-%d") for e in events]
M   = [e[0].strftime("%Y-%m")   for e in events]
LVL = [e[2] for e in events]
UID = [e[1] for e in events]
CAS = [e[3] for e in events]
cascades = sorted(set(CAS))
p("## 1. 数据（同 E1 口径）")
p("")
p("- 统一事件流 **%d** ｜ 种子级联 **%d** 棵 ｜ 解析+排序 %.1fs" % (N, len(cascades), _time.time() - T0))

# ---------------- 2. 立方体：时间 × 事件簇维（cluster） ----------------
CUBOIDS = ["day_cluster", "month_cluster", "all_cluster"]
cluster = {c: c for c in cascades}      # cascade -> cluster_id（初始：每种子自成簇 = BIRTH）

def cells_of(i):
    cl = cluster[CAS[i]]
    return (
        ("day_cluster",   (D[i], cl)),
        ("month_cluster", (M[i], cl)),
        ("all_cluster",   ("ALL", cl)),
    )

def new_cube():
    return {n: {} for n in CUBOIDS}

def cell_add(c, i, sign, lv, u):
    c[0] += sign; c[1] += sign * lv
    if u is not None:
        f = c[2].get(u, 0) + sign
        if f > 0:
            c[2][u] = f
        elif u in c[2]:
            del c[2][u]

def upd(cube, i, sign=1):
    lv = LVL[i]; u = UID[i]
    for name, key in cells_of(i):
        d = cube[name]
        c = d.get(key)
        if c is None:
            c = [0, 0, {}]; d[key] = c
        cell_add(c, i, sign, lv, u)

def rebuild_all():
    cube = new_cube()
    for i in range(N):
        upd(cube, i, 1)
    return cube

def compare(a, b):
    diffs = 0; detail = []
    for name in CUBOIDS:
        keys = set(a[name]) | set(b[name])
        for k in keys:
            ca = a[name].get(k); cb = b[name].get(k)
            va = (ca[0], ca[1], len(ca[2])) if ca else (0, 0, 0)
            vb = (cb[0], cb[1], len(cb[2])) if cb else (0, 0, 0)
            if va != vb:
                diffs += 1
                if len(detail) < 3:
                    detail.append((name, k, va, vb))
    return diffs, detail

def cells_total(cube):
    return sum(len(cube[n]) for n in CUBOIDS)

# 基线立方体（演化前）
inc = new_cube()
for i in range(N):
    upd(inc, i, 1)
base_cells = cells_total(inc)
by_cas = collections.Counter(CAS)
top = [c for c, _ in by_cas.most_common()]
p("- 演化前 cells = **%d**（%d cuboid）" % (base_cells, len(CUBOIDS)))

# ---------------- 3. 演化操作（通用执行器） ----------------
def evolve(moves, op_name):
    """moves = list of (cascade, old_cluster, new_cluster)；增量只动受影响 cell"""
    t_inc = _time.time()
    # 受影响事件与 cell
    aff_ev = [i for i in range(N) if CAS[i] in set(m[0] for m in moves)]
    aff_keys = set()
    for i in aff_ev:
        for name, key in cells_of(i):
            aff_keys.add((name, key))
    # 写时复制受影响 cell
    inc2 = {n: dict(inc[n]) for n in CUBOIDS}
    for name, key in aff_keys:
        c = inc[name].get(key)
        inc2[name][key] = [c[0], c[1], dict(c[2])] if c else [0, 0, {}]
    # 先删旧（按旧簇键），再改映射，再加新
    tmp_map = {}
    for cas, old, new in moves:
        tmp_map[cas] = (cluster[cas], new)   # 记录旧簇
    # 删除：用旧键
    for i in aff_ev:
        cas = CAS[i]
        old = tmp_map[cas][0]
        for name, tkey in ((("day_cluster"), (D[i], old)),
                           (("month_cluster"), (M[i], old)),
                           (("all_cluster"), ("ALL", old))):
            c = inc2[name][tkey]
            cell_add(c, i, -1, LVL[i], UID[i])
    # 提交映射
    for cas, old, new in moves:
        cluster[cas] = new
    # 新增：新键
    for i in aff_ev:
        upd_dict = {}
        for name, key in cells_of(i):
            d = inc2[name]
            c = d.get(key)
            if c is None:
                c = [0, 0, {}]; d[key] = c
            cell_add(c, i, +1, LVL[i], UID[i])
    dt_inc = _time.time() - t_inc
    inc_ops = 2 * len(aff_ev) * len(CUBOIDS)
    # 全量对照
    t_reb = _time.time()
    rb = rebuild_all()
    dt_reb = _time.time() - t_reb
    reb_ops = N * len(CUBOIDS)
    diffs, detail = compare(inc2, rb)
    # 空 cell 清理（增量可能留下 0 计数 cell，语义上等价于不存在）
    for name in CUBOIDS:
        dead = [k for k, v in inc2[name].items() if v[0] == 0 and not v[2]]
        for k in dead:
            del inc2[name][k]
    diffs2, _ = compare(inc2, rb)
    res = {"op": op_name, "moved_cascades": len(moves),
           "moved_events": len(aff_ev),
           "affected_cells": len(aff_keys),
           "affected_pct": 100.0 * len(aff_keys) / base_cells,
           "inc_ops": inc_ops, "reb_ops": reb_ops,
           "cost_ratio": reb_ops / max(1, inc_ops),
           "t_inc": dt_inc, "t_rebuild": dt_reb,
           "diffs_raw": diffs, "diffs_after_gc": diffs2}
    inc.clear(); inc.update(inc2)
    return res, detail

results = []
p("")
p("## 2. 三类结构操作的回溯重分区（增量 vs 全量）")
p("")
p("| 操作 | 语义 | 移动级联 | 移动事件 | 受影响 cells | 占比 | 增量代价(ops) | 全量代价(ops) | 代价比 | 等价? |")
p("|---|---|---|---|---|---|---|---|---|---|")

# ① MERGE：最大 100 簇两两合并（= 50 个 L3 事件并入 25 个 L2 议题）
moves = []
pairs = [(top[2*j], top[2*j+1]) for j in range(50) if 2*j+1 < len(top)]
for a, b in pairs:
    moves.append((b, cluster[b], cluster[a]))
r, det = evolve(moves, "MERGE")
results.append(r)
p("| MERGE ×%d | 两个 L3 事件并入同一 L2 议题 | %d | %d | %d | %.2f%% | %d | %d | %.1f× | %s |" % (
    len(moves), r["moved_cascades"], r["moved_events"], r["affected_cells"], r["affected_pct"],
    r["inc_ops"], r["reb_ops"], r["cost_ratio"],
    "✅ 0 差异" if r["diffs_after_gc"] == 0 else "❌ %d" % r["diffs_after_gc"]))

# ② REPARENT：100 个中等簇改挂到最大簇（= 叙事改挂另一事件簇，D18 话题维实例）
mid_pool = [c for c in top[200:500]]
moves = [(c, cluster[c], cluster[top[0]]) for c in mid_pool[:100]]
r, det = evolve(moves, "REPARENT")
results.append(r)
p("| REPARENT ×%d | 叙事从一个事件子簇改挂另一簇 | %d | %d | %d | %.2f%% | %d | %d | %.1f× | %s |" % (
    len(moves), r["moved_cascades"], r["moved_events"], r["affected_cells"], r["affected_pct"],
    r["inc_ops"], r["reb_ops"], r["cost_ratio"],
    "✅ 0 差异" if r["diffs_after_gc"] == 0 else "❌ %d" % r["diffs_after_gc"]))

# ③ SPLIT：从最大簇拆出 50 个成员级联另立新簇（= 事件分裂出新叙事）
members = [c for c in cascades if cluster[c] == cluster[top[0]] and c != cluster[top[0]]][:50]
if len(members) < 50:
    members = [c for c in top[1:51] if cluster[c] == cluster[top[0]]][:50]
new_id = "SPLIT_" + cluster[top[0]]
moves = [(c, cluster[c], new_id) for c in members]
r, det = evolve(moves, "SPLIT")
results.append(r)
p("| SPLIT ×1 | 大事件分裂出新叙事（拆出 %d 成员） | %d | %d | %d | %.2f%% | %d | %d | %.1f× | %s |" % (
    len(members), r["moved_cascades"], r["moved_events"], r["affected_cells"], r["affected_pct"],
    r["inc_ops"], r["reb_ops"], r["cost_ratio"],
    "✅ 0 差异" if r["diffs_after_gc"] == 0 else "❌ %d" % r["diffs_after_gc"]))

# ---------------- 4. 汇总 ----------------
p("")
p("## 3. 汇总")
p("")
all_ok = all(r["diffs_after_gc"] == 0 for r in results)
p("- **等价性**：%s（三类操作后增量与全量重建逐 cell 相等；中间态的 0 计数空 cell 由 GC 清除，语义等价）" % (
    "✅ 全部 0 差异" if all_ok else "❌ 有差异"))
p("- **回溯重分区的局部性**：受影响 cells 仅占全立方体 **%.2f%% ~ %.2f%%**——维度演化是**高度局部**的扰动，" % (
    min(r["affected_pct"] for r in results), max(r["affected_pct"] for r in results)))
p("  增量维护只碰受影响 cell，代价 **%.0f× ~ %.0f×** 优于全量重建（操作数口径）。" % (
    min(r["cost_ratio"] for r in results), max(r["cost_ratio"] for r in results)))
p("- **与 RETRACT 的对称性**：RETRACT 的失效集 = 传播级联（E1 §5）；维度演化的失效集 = **被移动成员触及的 cell**。")
p("  两者都是 O(affected)，这正是 RQ1 §4.3 问题 2「最小重算集」在**维度侧**的镜像（§6.3.1）。")
p("")
p("> 说明：① 簇演化注入在**事件簇维**上模拟话题维的 MERGE/REPARENT/SPLIT（D18：七操作四维度共用，")
p("> 话题维实例 = 事件簇演化）；② MERGE 的上卷去重（桥接表权重）属于度量层（D3），本实验在骨架层验证，")
p("> 故 count/Σlevel/distinct-uid 直接可加合并；③ 完整版须在「存储图 + roll-up 树」两相上分别验证（D7）——已列入下一步。")

io.open(os.path.join(OUT, "E1_EVO_报告.md"), "w", encoding="utf-8").write(
    "# E1-EVO 报告：维度演化注入（MERGE / REPARENT / SPLIT）—— 回溯重分区原型\n\n"
    "> 生成：WorkBuddy ｜ 2026-10-10 ｜ RQ1 §6.3.1「维度驱动的回溯重分区」首个真实数据实验\n\n"
    + "\n".join(LOG) + "\n")

io.open(os.path.join(OUT, "e1_evo_metrics.json"), "w", encoding="utf-8").write(
    json.dumps({"N": N, "cascades": len(cascades), "base_cells": base_cells,
                "cuboids": CUBOIDS, "ops": results, "all_equivalent": all_ok},
               ensure_ascii=False, indent=2))

print("DONE ->", OUT)
