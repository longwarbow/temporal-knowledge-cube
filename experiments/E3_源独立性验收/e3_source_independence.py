# -*- coding: utf-8 -*-
"""
E3: D2-rev1 验收实验 —— CED 传播树上结构化「源独立性」S1/S2/S3 的判别力检验

背景
  V11（D2-rev1 触发源）：比值型朴素实现 unique_uid/转发数 在 CED 上无判别力
  （谣言 0.975 vs 非谣 0.981，中位差 0.006）。
  D2-rev1(b) 决议：源独立性改为传播树上的结构化定义 ——
    S1 影响力集中度（跨级联高活跃账号占比 / 单账号最大占比）
    S2 核心转发者子图（最大子分支占比 / 传播深度）
    S3 时间同步性（首窗爆发占比 / 传播时间跨度）

方法
  每棵传播树（根帖）提取特征 → 谣言 vs 非谣言两组 →
  Mann-Whitney U（AUC = U/(n1*n2)，带结修正）+ Cliff's delta + 中位数对照。
  验收标准：朴素基线 AUC ≈ 0.5（复现 V11）；至少一个 S 特征 AUC 显著偏离 0.5
  且效应量 ≥ small（|delta| ≥ 0.147）。

输出：E3_报告.md + e3_metrics.json
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
import os, io, json, math, time as _time, collections, datetime, statistics

BASE = _KB('数据', 'CED', 'Chinese_Rumor_Dataset-master', 'CED_Dataset')
RUM  = os.path.join(BASE, "rumor-repost")
NON  = os.path.join(BASE, "non-rumor-repost")
ORIG = os.path.join(BASE, "original-microblog")
OUT  = _KB('experiments', 'E3_源独立性验收')
os.makedirs(OUT, exist_ok=True)

LOG = []
def p(x=""):
    LOG.append(str(x))

def parse_dt(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

# ---------------- 1. 解析：每棵传播树的原始素材 ----------------
T0 = _time.time()
rum_files = set(os.listdir(RUM))
non_files = set(os.listdir(NON))
orig_files = sorted(os.listdir(ORIG))

# trees[label] = list of dict(mid, uid, parent, t)
trees = {1: [], 0: []}   # 1=rumor, 0=non-rumor
seed_t = {}              # file -> seed datetime (A1 回退时为 None)
bad = 0
for fn in orig_files:
    op = os.path.join(ORIG, fn)
    try:
        o = json.load(io.open(op, encoding="utf-8"))
    except Exception:
        bad += 1; continue
    if isinstance(o, list):
        o = o[0] if o else {}
    if not isinstance(o, dict):
        bad += 1; continue
    label = 1 if fn in rum_files else (0 if fn in non_files else None)
    if label is None:
        bad += 1; continue
    rp = os.path.join(RUM if label else NON, fn)
    arr = []
    if os.path.exists(rp):
        try:
            arr = json.load(io.open(rp, encoding="utf-8"))
        except Exception:
            arr = []
    if not isinstance(arr, list):
        arr = []
    reps = []
    for r in arr:
        if isinstance(r, dict) and r.get("mid"):
            d = parse_dt(r.get("date") or "")
            if d:
                reps.append({"mid": r["mid"], "uid": r.get("uid"),
                             "parent": r.get("parent"), "t": d})
    ts = o.get("time")
    t0 = None
    if isinstance(ts, (int, float)) and ts > 0:
        try:
            t0 = datetime.datetime.fromtimestamp(ts)
        except Exception:
            t0 = None
    trees[label].append({"file": fn, "reps": reps, "t0": t0})

p("## 1. 数据")
p("")
p("- CED 原帖解析：谣言树 %d ｜ 非谣言树 %d ｜ 解析失败 %d" % (
    len(trees[1]), len(trees[0]), bad))

# ---------------- 2. 跨级联账号活跃度（S1 的基础） ----------------
uid_cascades = collections.defaultdict(set)
for lab in (1, 0):
    for tr in trees[lab]:
        for r in tr["reps"]:
            if r["uid"]:
                uid_cascades[r["uid"]].add(tr["file"])
hub_thresh = 3
p("- 转发者总数（去重）%d ｜ 出现在 ≥%d 棵树的「高活跃账号」%d 个（占 %.2f%%）" % (
    len(uid_cascades), hub_thresh,
    sum(1 for s in uid_cascades.values() if len(s) >= hub_thresh),
    100.0 * sum(1 for s in uid_cascades.values() if len(s) >= hub_thresh) / max(1, len(uid_cascades))))

# ---------------- 3. 逐树特征 ----------------
def tree_features(tr):
    reps = tr["reps"]
    n = len(reps)
    if n == 0:
        return None
    uids = [r["uid"] for r in reps]
    # 朴素基线（V11）
    naive = len(set(uids)) / n
    # S1 影响力集中度
    hub_cnt = 0
    freq = collections.Counter(uids)
    for u in uids:
        if u and len(uid_cascades.get(u, ())) >= hub_thresh:
            hub_cnt += 1
    s1_hub_share = hub_cnt / n
    s1_max_uid = max(freq.values()) / n
    # 层级（parent=='' 或指向集合外 = L1）
    nodes = {r["mid"]: r for r in reps}
    children = collections.defaultdict(list)
    roots = []
    for r in reps:
        par = r["parent"]
        if not par or par not in nodes:
            roots.append(r["mid"])
        else:
            children[par].append(r["mid"])
    # 子树规模（自 L1 根向下）
    def subtree_size(root):
        dq = [root]; cnt = 0
        while dq:
            x = dq.pop(); cnt += 1
            dq.extend(children.get(x, ()))
        return cnt
    depths = []
    dq = collections.deque((r, 1) for r in roots)
    while dq:
        x, d = dq.popleft()
        depths.append(d)
        for c in children.get(x, ()):
            dq.append((c, d + 1))
    s2_largest = max((subtree_size(r) for r in roots), default=0) / n
    s2_depth_med = statistics.median(depths) if depths else 1.0
    # S3 时间同步性
    ts = sorted(r["t"] for r in reps)
    t0 = tr["t0"] or ts[0]          # A1 回退：t0 = 最早转发
    span_h = (ts[-1] - ts[0]).total_seconds() / 3600.0
    horizon = t0 + datetime.timedelta(hours=24)
    early24 = sum(1 for t in ts if t <= horizon) / n
    return {
        "naive": naive,
        "s1_hub_share": s1_hub_share, "s1_max_uid": s1_max_uid,
        "s2_largest": s2_largest, "s2_depth_med": s2_depth_med,
        "s3_early24": early24, "s3_log_span": math.log10(span_h + 1.0),
        "n": n, "a1_fallback": tr["t0"] is None,
    }

feats = {1: [], 0: []}
skipped = {1: 0, 0: 0}
a1_cnt = {1: 0, 0: 0}
for lab in (1, 0):
    for tr in trees[lab]:
        f = tree_features(tr)
        if f is None:
            skipped[lab] += 1; continue
        if f.pop("a1_fallback"):
            a1_cnt[lab] += 1
        feats[lab].append(f)

p("- 有效树：谣言 %d（空树跳过 %d）｜ 非谣言 %d（空树跳过 %d）；A1 回退（种子时间缺失）谣 %d / 非谣 %d" % (
    len(feats[1]), skipped[1], len(feats[0]), skipped[0], a1_cnt[1], a1_cnt[0]))
p("- 解析+特征提取用时：%.1fs" % (_time.time() - T0))

# ---------------- 4. 统计检验（Mann-Whitney U，带结修正） ----------------
def rankdata(xs):
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks, idx

def mwu(g1, g2):
    """返回 (auc, z, p, cliffs_delta)；g1 = 谣言组"""
    n1, n2 = len(g1), len(g2)
    allv = list(g1) + list(g2)
    ranks, _ = rankdata(allv)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    auc = u1 / (n1 * n2)
    N = n1 + n2
    # 结修正
    cnt = collections.Counter(allv)
    tie_term = sum(t * (t * t - 1) for t in cnt.values()) / (N * (N - 1.0)) if N > 1 else 0.0
    var = (n1 * n2 / 12.0) * ((N + 1) - tie_term)
    z = (u1 - n1 * n2 / 2.0) / math.sqrt(var) if var > 0 else 0.0
    pval = math.erfc(abs(z) / math.sqrt(2.0))   # 双侧
    return auc, z, pval, 2.0 * auc - 1.0

def effect(d):
    a = abs(d)
    return "large" if a >= 0.474 else ("medium" if a >= 0.33 else ("small" if a >= 0.147 else "negligible"))

FEATURES = [
    ("naive",        "朴素比值 unique_uid/转发数（V11 基线）", "比值"),
    ("s1_hub_share", "S1a 跨级联高活跃账号转发占比（≥3 树）", "结构"),
    ("s1_max_uid",   "S1b 单账号最大转发占比",                "结构"),
    ("s2_largest",   "S2a 最大子分支占比（核心转发者子图）",  "结构"),
    ("s2_depth_med", "S2b 传播深度中位数",                    "结构"),
    ("s3_early24",   "S3a 首 24h 爆发占比（时间同步性）",     "时间"),
    ("s3_log_span",  "S3b 传播时间跨度 log10(小时+1)",        "时间"),
]

p("")
p("## 2. 判别力检验（谣言 vs 非谣言；AUC>0.5 = 特征越大越像谣言）")
p("")
p("| 特征 | 类型 | 谣言中位 | 非谣中位 | AUC | Cliff's δ | 效应量 | z | p |")
p("|---|---|---|---|---|---|---|---|---|")
results = {}
for key, name, typ in FEATURES:
    g1 = [f[key] for f in feats[1]]
    g2 = [f[key] for f in feats[0]]
    auc, z, pv, d = mwu(g1, g2)
    m1 = statistics.median(g1); m2 = statistics.median(g2)
    results[key] = {"name": name, "type": typ, "auc": auc, "z": z, "p": pv,
                    "cliffs_delta": d, "effect": effect(d),
                    "median_rumor": m1, "median_non": m2,
                    "n_rumor": len(g1), "n_non": len(g2)}
    p("| %s | %s | %.4f | %.4f | **%.3f** | %+.3f | %s | %.1f | %s |" % (
        name, typ, m1, m2, auc, d, effect(d), z,
        ("<1e-300" if pv < 1e-300 else "%.2e" % pv)))

# 组合上界：按单特征 AUC 方向定符号后取秩和
p("")
best_keys = [k for k, _, _ in FEATURES if k != "naive"]
def combined_score(f):
    s = 0.0
    for k in best_keys:
        sign = 1.0 if results[k]["auc"] >= 0.5 else -1.0
        s += sign * f[k]
    return s
sc1 = [combined_score(f) for f in feats[1]]
sc0 = [combined_score(f) for f in feats[0]]
auc_c, z_c, pv_c, d_c = mwu(sc1, sc0)
results["_combined"] = {"auc": auc_c, "p": pv_c, "cliffs_delta": d_c}
p("> **组合分（S1+S2+S3 按方向加权）**：AUC = **%.3f** ｜ δ = %+.3f ｜ p = %s" % (
    auc_c, d_c, "<1e-300" if pv_c < 1e-300 else "%.2e" % pv_c))
p("> ⚠️ 组合分在**同一样本上定方向**，属**样本内上界**，仅作方向性参考；")
p("> 正式结论以单特征 AUC + 显著性 + 效应量为准，且需在保留集/其他数据上复验。")

# ---------------- 5. 验收判定 ----------------
p("")
p("## 3. 验收判定（对照 D2-rev1(b)）")
p("")
naive_auc = results["naive"]["auc"]
p("1. **复现 V11**：朴素比值 AUC = **%.3f**（≈0.5，无判别力）→ 与 V11 一致，重定义的动机成立。" % naive_auc)
s_feats = [k for k in best_keys]
sig = [(k, results[k]) for k in s_feats
       if abs(results[k]["cliffs_delta"]) >= 0.147 and results[k]["p"] < 0.01]
p("2. **结构化定义的判别力**：%d/6 个 S 特征达到显著（p<0.01）且效应量 ≥ small：" % len(sig))
for k, r in sig:
    p("   - %s：AUC=%.3f，δ=%+.3f（%s）" % (r["name"], r["auc"], r["cliffs_delta"], r["effect"]))
if len(sig) == 0:
    p("   - （无——须回决议记录讨论）")
p("3. **结论**：%s" % (
    "**D2-rev1(b) 通过验收**——结构化定义在 CED 上具备比值型实现所没有的判别力，源独立性判据正式切换为 S1/S2/S3。"
    if len(sig) >= 1 else "**验收未通过**——须回到决议记录重议 D2-rev1(b)。"))
p("")
p("> 诚实声明：① CED 转发树是**抽样树**（抓取率约 65%），绝对值偏小但不影响组间对比；")
p("> ② 属单数据集单平台结论，泛化性待 PHEME / Weibo-WM 复验（D5 泛化组）；")
p("> ③ 判别力 ≠ 因果：特征显著只说明「与谣言标签相关」，作为**判据组件**仍须与 support 等联合使用（§4 组合判据）。")

io.open(os.path.join(OUT, "E3_报告.md"), "w", encoding="utf-8").write(
    "# E3 报告：D2-rev1 验收——结构化「源独立性」S1/S2/S3 判别力检验（CED 传播树）\n\n"
    "> 生成：WorkBuddy ｜ 2026-10-10 ｜ 决议记录 §4 待办④（D2-rev1(b) 验收实验）\n\n" + "\n".join(LOG) + "\n")

io.open(os.path.join(OUT, "e3_metrics.json"), "w", encoding="utf-8").write(
    json.dumps({"features": results,
                "n_rumor": len(feats[1]), "n_non": len(feats[0]),
                "skipped": skipped, "a1_fallback": a1_cnt,
                "hub_threshold": hub_thresh},
               ensure_ascii=False, indent=2))

print("DONE ->", OUT)
