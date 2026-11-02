# -*- coding: utf-8 -*-
"""
E1-SW: 滑动窗 + 迟到事件 —— 双时态必要性的量化（E1 报告「下一步①」）

背景
  E1 已证：**滚动型时间 cell（日/月，不关窗）** 对到达顺序不敏感 → 增量 = 全量。
  真实流式系统会在窗口关闭后**物化发布**；此时**迟到事件**会击穿已关闭窗口 → 已发布结果为错。

设计
  - 滑动窗：窗长 W=24h，步长 S=6h（每事件属于 M=4 个重叠窗）。
  - 真实延迟：δ ~ U(0, Δmax)，主设定 Δmax=24h（≈一天抓取延迟）。
  - 水位容差 Δw：物化时刻 = k*S+W+Δw；Δw 越接近 Δmax，出错越少，Δw≥Δmax 时归零。
  - 对每窗：真值 = 事件时间落窗者；已发布 = 到达早于物化时刻者。
    读数：出错窗占比 / 已发布计数相对误差（若不修正）/ 需重开的窗工作量占比（若修正）。
  - 对照锚：滚动型（不关窗）日 cell 恒 0 误差。

输出：E1_SW_报告.md + e1_sw_metrics.json
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
OUT  = _KB('experiments', 'E1_滑动窗迟到事件')
os.makedirs(OUT, exist_ok=True)

LOG = []
def p(x=""):
    LOG.append(str(x))

def parse_dt(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

EPOCH = datetime.datetime(1970, 1, 1)
def ts(dt):
    return (dt - EPOCH).total_seconds()

# ---------------- 1. 解析事件（同 E1） ----------------
T0 = _time.time()
rum_files = set(os.listdir(RUM)); non_files = set(os.listdir(NON))
orig_files = sorted(os.listdir(ORIG))
T = []; LVL = []
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
    arr = []
    rp = os.path.join(label_dir, fn)
    if os.path.exists(rp):
        try:
            arr = json.load(io.open(rp, encoding="utf-8"))
        except Exception:
            arr = []
    if not isinstance(arr, list):
        arr = []
    rt = []
    for r in arr:
        if isinstance(r, dict):
            d = parse_dt(r.get("date") or "")
            if d:
                rt.append(ts(d))
    tsd = o.get("time")
    st = float(tsd) if isinstance(tsd, (int, float)) and tsd > 0 else (min(rt) if rt else None)
    if st is not None:
        T.append(st); LVL.append(0)
    for x in rt:
        T.append(x); LVL.append(1)
N = len(T)
p("## 1. 数据")
p("")
p("- 事件总数 **%d** ｜ 时间跨度 %s ~ %s ｜ 解析 %.1fs" % (
    N, datetime.datetime.fromtimestamp(min(T)).strftime("%Y-%m-%d"),
    datetime.datetime.fromtimestamp(max(T)).strftime("%Y-%m-%d"), _time.time() - T0))

# ---------------- 2. 锚对照：滚动型（不关窗） ----------------
rng = random.Random(2026)
order = list(range(N)); rng.shuffle(order)
def tumbling(seq):
    d = {}
    for i in seq:
        k = datetime.datetime.fromtimestamp(T[i]).strftime("%Y-%m-%d")
        c = d.get(k)
        if c is None:
            d[k] = [1, LVL[i]]
        else:
            c[0] += 1; c[1] += LVL[i]
    return d
a = {k: tuple(v) for k, v in tumbling(range(N)).items()}
b = {k: tuple(v) for k, v in tumbling(order).items()}
td = sum(1 for k in set(a) | set(b) if a.get(k) != b.get(k))
p("")
p("## 2. 锚对照：滚动型（不关窗）日 cell")
p("")
p("- 事件时间序 vs 完全打乱序：**%s**" % ("✅ 0 差异" if td == 0 else "❌ %d 差异" % td))
p("- ⇒ 脆弱性**不是**来自聚合，而是**来自「窗口关闭 + 迟到」**（下面量化）。")

# ---------------- 3. 滑动窗预计算 ----------------
S = 6 * 3600; W = 24 * 3600; M = W // S
def windows_of(t):
    K = int(t // S)
    return [K - j for j in range(M)]
true_cnt = collections.Counter()
SLACK = []                     # 每事件的 (窗id, 该窗剩余时间 slack = 窗尾 - 事件时间)
for i in range(N):
    t = T[i]
    row = []
    for k in windows_of(t):
        true_cnt[k] += 1
        row.append((k, k * S + W - t))
    SLACK.append(row)
n_windows = len(true_cnt)
total_window_work = sum(true_cnt.values())     # 全窗重算工作量（= 总成员数）
p("")
p("## 3. 滑动窗设置")
p("")
p("- 窗长 W=24h ｜ 步长 S=6h ｜ 重叠 M=%d ｜ **窗数 %d** ｜ 全窗重算工作量 = **%d**（总成员数）" % (
    M, n_windows, total_window_work))

# ---------------- 4. 扫描：真实延迟 & 水位容差 ----------------
DMAX = 24 * 3600
delay = [rng.uniform(0, DMAX) for _ in range(N)]
ARR = [T[i] + delay[i] for i in range(N)]
d_sorted = sorted(delay)

def scan(dw):
    """事件对窗 k 为准时 ⟺ delay ≤ slack + dw"""
    pub = collections.Counter()
    for i in range(N):
        di = delay[i]
        for k, sl in SLACK[i]:
            if di <= sl + dw:
                pub[k] += 1
    bad = err = reopen = 0
    for k in true_cnt:
        g = pub.get(k, 0)
        if g != true_cnt[k]:
            bad += 1
            err += abs(g - true_cnt[k])
            reopen += true_cnt[k]
    return bad, err, reopen

p("")
p("## 4. 主扫描：真实延迟 δ~U(0, 24h)，扫水位容差 Δw")
p("")
p("| 水位容差 Δw | 出错窗数 | 占比 | 计数相对误差(不修正) | 需重开窗工作量 | 占全窗工作量 | 修正代价比(vs全窗重算) |")
p("|---|---|---|---|---|---|---|")
rows = []
for name, dw in [("0h（假定准时）", 0), ("1h", 3600), ("6h", 6 * 3600),
                 ("12h", 12 * 3600), ("24h（=Δmax，正确配置）", DMAX)]:
    bad, err, reopen = scan(dw)
    rel = 100.0 * err / total_window_work
    rows.append({"tol": name, "dw_h": dw / 3600.0, "bad_windows": bad,
                 "bad_pct": 100.0 * bad / n_windows, "rel_err_pct": rel,
                 "reopen_work": reopen,
                 "reopen_pct": 100.0 * reopen / total_window_work,
                 "cost_ratio": reopen / max(1, total_window_work)})
    p("| %s | %d | %.2f%% | %.3f%% | %d | %.2f%% | %.3f× |" % (
        name, bad, rows[-1]["bad_pct"], rel, reopen, rows[-1]["reopen_pct"], rows[-1]["cost_ratio"]))

p("")
p("**读数**：")
r0 = rows[0]
p("- 假定准时（Δw=0）：**%.1f%%** 的窗被迟到事件击穿，若不修正则已发布计数**误差 %.1f%%**；" % (
    r0["bad_pct"], r0["rel_err_pct"]))
p("  若修正（重开这些窗）则**需重做 %.1f%% 的窗口聚合工作量**（代价比 %.2f× vs 全窗重算）。" % (
    r0["reopen_pct"], r0["cost_ratio"]))
r4 = rows[-1]
p("- 正确配置（Δw≥Δmax）：**0 出错、0 误差、0 重算**（%.2f%%）——标准流式语义的收敛点。" % r4["bad_pct"])
p("- 误差随 Δw **单调下降**：%s。" % " → ".join("%.1f%%" % x["rel_err_pct"] for x in rows))

def inc_correction(dw):
    """合并式度量（D3）下的最小修正代价：每个迟到事件只对其受影响窗做一次 O(1) 合并"""
    late_ev = 0; ops = 0
    for i in range(N):
        di = delay[i]; hit = 0
        for k, sl in SLACK[i]:
            if di > sl + dw:
                hit += 1
        if hit:
            late_ev += 1; ops += hit
    return late_ev, ops

p("")
p("### 增量修正代价（合并式度量口径，D3/T1–T6）")
p("")
p("上表的「需重开窗工作量」按**整窗重算**计（保守）；若度量是**可合并的**（E1 的 count/Σlevel/distinct-uid 皆可），")
p("则修正一个窗只需把**迟到事件**做一次 O(1) 合并即可，代价 = O(迟到事件数)。")
p("")
p("| 水位容差 Δw | 迟到事件数 | 占比 | 增量修正 ops | 全窗重算 ops | 节省倍数 |")
p("|---|---|---|---|---|---|")
for name, dw in [("0h", 0), ("6h", 6 * 3600), ("12h", 12 * 3600)]:
    le, ops = inc_correction(dw)
    p("| %s | %d | %.2f%% | %d | %d | **%.0f×** |" % (
        name, le, 100.0 * le / N, ops, total_window_work, total_window_work / max(1, ops)))
p("")
p("> ⇒ **双时态的价值双向**：不修正 → 已发布结果**错**（上表误差）；修正 → 用 O(迟到事件) 的合并代价即可**回填**，")
p("> **无需全窗重算**。这正是「可合并度量 + 双时态」的组合意义：**既能发现错（event-time 归属），又能廉价改对（mergeable 度量）**。")

# ---------------- 5. 次扫描：真实延迟 vs 固定容差 ----------------
p("")
p("## 5. 次扫描：固定水位容差 Δw=6h，放大真实最大延迟 Δmax")
p("")
p("| 真实 Δmax | 出错窗占比 | 计数相对误差 | 需重开窗工作量占比 |")
p("|---|---|---|---|")
sub = []
for dmax_h in [1, 6, 12, 24, 48, 168]:
    dl = [rng.uniform(0, dmax_h * 3600) for _ in range(N)]
    pub = collections.Counter()
    for i in range(N):
        di = dl[i]
        for k, sl in SLACK[i]:
            if di <= sl + 6 * 3600:
                pub[k] += 1
    bad = 0; err = 0; reopen = 0
    for k in true_cnt:
        g = pub.get(k, 0)
        if g != true_cnt[k]:
            bad += 1; err += abs(g - true_cnt[k]); reopen += true_cnt[k]
    sub.append({"dmax_h": dmax_h, "bad_pct": 100.0 * bad / n_windows,
                "rel_err_pct": 100.0 * err / total_window_work,
                "reopen_pct": 100.0 * reopen / total_window_work})
    p("| %dh | %.2f%% | %.3f%% | %.2f%% |" % (dmax_h, sub[-1]["bad_pct"], sub[-1]["rel_err_pct"], sub[-1]["reopen_pct"]))
p("")
p("- ⇒ 相同的容差下，**真实延迟越大、误差越大**；系统要么**放大容差**（牺牲时效），要么**承担重算**（牺牲代价）。")
p("  这正是 P1 的「一致性 ∧ 时效」约束（RQ1 §4.3 问题 1）在**时间窗这一侧的体现**。")

# ---------------- 6. 汇总 ----------------
p("")
p("## 6. 汇总（E1 线闭环）")
p("")
p("| 维度轴 | 是否关窗 | 迟到敏感性 | 证据 |")
p("|---|---|---|---|")
p("| 滚动型（日/月，不关窗） | 否 | **不敏感（0 差异）** | 本节 §2；E1 §4 |")
p("| 滑动型（24h/6h，关闭物化） | 是 | **敏感**（Δw=0 时出错窗 %.1f%%） | 本节 §4 |" % r0["bad_pct"])
p("")
p("- **结论**：E1 的「增量 ≡ 全量」成立于**不关窗**的滚动 cell；一旦引入**关闭的滑动窗**，")
p("  迟到事件必然使已发布结果失效 → 必须**事件时间 t + 系统时间 τ 双轴**（双时态）方能自洽。")
p("  这是 RQ1 §1 定义 2（两条独立时间轴）的**首个真实数据量化证据**。")
p("")
p("> 诚实声明：CED 无真实到达时间，延迟为**注入模型**（U(0,Δmax)）；结论是「机制上的必然」，")
p("> 具体误差幅度取决于真实延迟分布，待 Weibo-WM / 自采流上复验。")

io.open(os.path.join(OUT, "E1_SW_报告.md"), "w", encoding="utf-8").write(
    "# E1-SW 报告：滑动窗 + 迟到事件 —— 双时态必要性的量化\n\n"
    "> 生成：WorkBuddy ｜ 2026-10-10 ｜ E1 报告「下一步①」；RQ1 §1 定义 2 / §3 的量化证据\n\n"
    + "\n".join(LOG) + "\n")

io.open(os.path.join(OUT, "e1_sw_metrics.json"), "w", encoding="utf-8").write(
    json.dumps({"N": N, "windows": n_windows, "W_h": 24, "S_h": 6,
                "delay_median_h": sorted(delay)[N // 2] / 3600.0,
                "tumbling_diffs": td,
                "main_scan_dmax24h": rows, "sub_scan_dw6h": sub},
               ensure_ascii=False, indent=2))

print("DONE ->", OUT)
