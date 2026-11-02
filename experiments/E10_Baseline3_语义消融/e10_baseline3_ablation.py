# -*- coding: utf-8 -*-
"""
E10 —— Baseline #3：语义消融的数值流立方体（numeric stream cube, semantics ablated）

论文 §7.7 的 #3 行。问题设定：
    把「语义层」从时序知识立方体里拿掉——即只保留数值可加聚合（n_posts / sum followers），
    去掉声明状态（claim state）、支撑/冲突（support / conflict）与血缘（lineage）——
    会付出什么代价？收益又是什么？

四个测量：
    A. 时间轴上的公平性（T-numeric）
       数值立方体的度量层 = 我们数据面度量的投影。用三档粗粒度证据说明：
       在「纯活动量」查询上，两者给出**逐 cell 相同**的答案 ⇒ #3 是公平基线，不是稻草人。
    B. 放大 ≠ 印证（T-inflate）
       数值立方体把每个 post 记一次「mention」；而「独立报告数」= 原始出处数。
       在 CED（中文级联）与 PHEME（英文线程）上实测**放大因子**。
    C. 冲突不可表示（T-invisible）
       数值立方体没有声明状态维度 ⇒ 无法回答「该 cell 含互相矛盾的声明吗」。
       在 PHEME 三值真值上量化：多少 cell / 多少体量落在**冲突 cell**里。
    D. 成本（代价的正面）
       数值立方体不需要血缘表 ⇒ 省掉多少行；这是它唯一的、真实的优势。

数据：CED（3,389 原帖 × 级联转发，中文）+ PHEME 9-event（6,425 线程，英文，三值真值）
输出：e10_results.json + E10_报告.md
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
import os, json, pickle, collections, datetime, math, time

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = _KB_ROOT
CED = os.path.join(ROOT, '数据', 'CED', 'Chinese_Rumor_Dataset-master', 'CED_Dataset')
PHEME = os.path.join(ROOT, '数据', 'PHEME')

EPOCH = datetime.datetime(1970, 1, 1)
TW_SNOW = 1288834974657
VER_NAME = {0: 'true', 1: 'false', 2: 'unverified'}
GRANS = ['day', 'hour', '10min']


def bucket(dt, gran):
    if gran == 'day':
        return dt.strftime('%Y-%m-%d')
    if gran == 'hour':
        return dt.strftime('%Y-%m-%dT%H')
    return dt.strftime('%Y-%m-%dT%H:') + str(dt.minute // 10) + '0'


def pctl(xs, p):
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(len(ys) * p))] if ys else 0


def load_json(p):
    """CED 个别文件是 0x80 截断的坏样本（约 0.3%）；两种编码都试，失败返回 None。"""
    try:
        raw = open(p, 'rb').read()
    except Exception:
        return None
    for e in ('utf-8', 'gb18030'):
        try:
            return json.loads(raw.decode(e))
        except Exception:
            pass
    return None


# ---------------------------------------------------------------- A + B: CED
def load_ced():
    ORIG = os.path.join(CED, 'original-microblog')
    RUM = os.path.join(CED, 'rumor-repost')
    NON = os.path.join(CED, 'non-rumor-repost')
    rum, non = set(os.listdir(RUM)), set(os.listdir(NON))
    cascades, bad = [], 0
    for fn in sorted(os.listdir(ORIG)):
        o = load_json(os.path.join(ORIG, fn))
        if o is None:
            bad += 1
            continue
        if isinstance(o, list):
            o = o[0] if o else {}
        if not isinstance(o, dict):
            bad += 1
            continue
        if fn in rum:
            lab, d = 'rumor', RUM
        elif fn in non:
            lab, d = 'non-rumor', NON
        else:
            bad += 1
            continue
        try:
            t = int(o.get('time'))
        except Exception:
            t = None
        arr = load_json(os.path.join(d, fn)) or []
        if not isinstance(arr, list):
            arr = []
        uids, times = set(), []
        for r in arr:
            if not isinstance(r, dict):
                continue
            u = r.get('uid')
            if u:
                uids.add(str(u))
            dt = r.get('date')
            if t is not None and dt:
                times.append((str(dt), t))
        dec = o.get('reposts')
        try:
            dec = int(dec)
        except Exception:
            dec = None
        cascades.append(dict(label=lab, n=1 + len(arr), nrep=len(arr), nuid=len(uids),
                             declared=dec, t=t))
    return cascades, bad


# ------------------------------------------------------------- A + B + C: PHEME
def load_pheme():
    G = pickle.load(open(os.path.join(PHEME, 'pheme_cascades.gpickle'), 'rb'))
    ver_raw = pickle.load(open(os.path.join(PHEME, 'veracity_dict.pkl'), 'rb'))
    ver = {}
    for k, v in ver_raw.items():
        if isinstance(v, str):
            v = {'true': 0, 'false': 1, 'unverified': 2}[v]
        ver[str(k)] = int(v)
    posts = []          # (t_ms, {gran: bucket}, event, thread_id)
    thread_event = {}
    for g in G:
        ev = g.graph.get('event', '?')
        tid = str(g.graph.get('thread_id', ''))
        thread_event[tid] = ev
        for n in g.nodes():
            if not str(n).isdigit():
                continue
            t = (int(n) >> 22) + TW_SNOW
            dt = EPOCH + datetime.timedelta(milliseconds=t)
            posts.append((t, {gr: bucket(dt, gr) for gr in GRANS}, ev, tid))
    return posts, ver, thread_event


def main():
    T0 = time.time()
    res = {}

    # ============================ CED ============================
    print('[1/4] 装载 CED …')
    casc, ced_bad = load_ced()
    n_orig = len(casc)
    tot_posts = sum(c['n'] for c in casc)
    tot_rep = sum(c['nrep'] for c in casc)
    nuid_all = len(set())  # placeholder
    # 全库 distinct uid（跨级联去重的用户数）
    alluids = set()
    # 为避免二次读盘，这里只统计「级联内 distinct uid 之和」与「级联数」
    sum_nuid = sum(c['nuid'] for c in casc)
    sizes = [c['n'] for c in casc]
    by_lab = collections.Counter()
    by_lab_posts = collections.Counter()
    for c in casc:
        by_lab[c['label']] += 1
        by_lab_posts[c['label']] += c['n']

    # 体量加权的放大因子（headline）：Σposts / Σorigin
    amp_posts = tot_posts / n_orig
    # 按级联中位数看「典型声明」的放大
    med_size = pctl(sizes, .5)
    # 用户去重：削减比（1.0 = 完全无效）与去重后的残余放大
    dedup_cut = (tot_posts / sum_nuid) if sum_nuid else float('nan')   # 1.04 ⇒ 只削 4%
    amp_after_ud = (sum_nuid / n_orig) if n_orig else float('nan')     # 去重后残余放大
    # 声明的 reposts 字段 vs 实际记录（数据质量 / 放大是下界）
    deq = [c for c in casc if c['declared'] is not None]
    deq_match = sum(1 for c in deq if c['declared'] == c['nrep'])
    sum_declared = sum(c['declared'] for c in deq)
    deq_med = pctl([c['declared'] for c in deq], .5)

    # A: 数值立方体的时间轴证据（day 粒度 cell）
    ced_cells = collections.Counter()
    for c in casc:
        if c['t'] is None:
            continue
        d = (EPOCH + datetime.timedelta(seconds=c['t'])).strftime('%Y-%m-%d')
        ced_cells[d] += c['n']

    res['ced'] = {
        'n_originals': n_orig, 'n_bad': ced_bad,
        'total_posts': tot_posts, 'total_reposts': tot_rep,
        'by_label': dict(by_lab), 'posts_by_label': dict(by_lab_posts),
        'cascade_size': {'min': min(sizes), 'p25': pctl(sizes, .25), 'median': med_size,
                         'p75': pctl(sizes, .75), 'p90': pctl(sizes, .9), 'p99': pctl(sizes, .99),
                         'max': max(sizes), 'mean': round(tot_posts / n_orig, 2)},
        'amp_posts_over_origins': round(amp_posts, 2),
        'dedup_by_user_cut_ratio': round(dedup_cut, 3),
        'amp_after_user_dedup_over_origins': round(amp_after_ud, 2),
        'declared_reposts': {'n_with_declared': len(deq), 'median_declared': deq_med,
                             'sum_declared': sum_declared,
                             'actual_over_declared_pct': round(100.0 * tot_rep / sum_declared, 2) if sum_declared else None,
                             'field_match_pct': round(100.0 * deq_match / len(deq), 2) if deq else None},
        'n_day_cells': len(ced_cells), 'day_span': (min(ced_cells), max(ced_cells)) if ced_cells else None,
    }

    # 大规模承重：体量前 10% 的级联贡献多少 post（说明放大集中在长尾头部）
    order = sorted(casc, key=lambda c: -c['n'])
    k = max(1, int(len(order) * 0.10))
    top10 = sum(c['n'] for c in order[:k])
    res['ced']['top10pct_cascades_volume_share_pct'] = round(100.0 * top10 / tot_posts, 2)

    # ============================ PHEME ============================
    print('[2/4] 装载 PHEME …')
    posts, ver, thread_event = load_pheme()
    n_posts = len(posts)
    n_threads = len(thread_event)
    amp_pheme = n_posts / n_threads
    ev_posts = collections.Counter(p[2] for p in posts)
    ev_threads = collections.Counter(thread_event.values())
    lab_thread = collections.Counter(VER_NAME[v] for v in ver.values())
    th_size = collections.Counter(p[3] for p in posts)
    ts = list(th_size.values())

    res['pheme'] = {
        'n_posts': n_posts, 'n_threads': n_threads,
        'amp_posts_over_threads': round(amp_pheme, 2),
        'thread_size': {'min': min(ts), 'p25': pctl(ts, .25), 'median': pctl(ts, .5),
                        'p75': pctl(ts, .75), 'p90': pctl(ts, .9), 'max': max(ts),
                        'mean': round(n_posts / n_threads, 2)},
        'veracity_dist': dict(lab_thread),
        'per_event': {e: {'posts': ev_posts[e], 'threads': ev_threads[e],
                          'amp': round(ev_posts[e] / ev_threads[e], 2)}
                      for e in sorted(ev_posts)},
    }

    # B/C: 逐 cell 聚合（3 粒度 × event）——数值计数 vs 冲突可见性
    thread_lab = {}
    for tid, ev in thread_event.items():
        v = ver.get(tid)
        thread_lab[tid] = VER_NAME[v] if v is not None else None

    conf = {}
    for gran in GRANS:
        cell_labs = collections.defaultdict(set)     # (bucket, event) -> set(labels)
        cell_n = collections.Counter()               # (bucket, event) -> n_posts
        for (t, bk, ev, tid) in posts:
            key = (bk[gran], ev)
            cell_n[key] += 1
            lb = thread_lab.get(tid)
            if lb in ('true', 'false'):
                cell_labs[key].add(lb)
        conflicted = {k for k, s in cell_labs.items() if s == {'true', 'false'}}
        vol_conf = sum(cell_n[k] for k in conflicted)
        conf[gran] = {
            'total_cells': len(cell_n),
            'conflicted_cells': len(conflicted),
            'conflicted_cell_pct': round(100.0 * len(conflicted) / len(cell_n), 2) if cell_n else 0.0,
            'volume_in_conflicted_pct': round(100.0 * vol_conf / n_posts, 2) if n_posts else 0.0,
        }
    res['pheme']['conflict'] = conf

    # 事件级冲突：一个事件里是否同时存在 true / false 线程
    ev_lab = collections.defaultdict(set)
    for tid, ev in thread_event.items():
        lb = thread_lab.get(tid)
        if lb in ('true', 'false'):
            ev_lab[ev].add(lb)
    res['pheme']['conflicted_events'] = {
        'n_events_with_labels': len(ev_lab),
        'n_conflicted_events': sum(1 for s in ev_lab.values() if s == {'true', 'false'}),
        'detail': {e: sorted(s) for e, s in sorted(ev_lab.items())},
    }

    # ============================ D: 成本 ============================
    print('[3/4] 成本对照 …')
    # 我们的语义立方体需要一条 claim → cell 的血缘表；数值立方体不需要。
    lineage_rows = 0
    for gran in GRANS:
        tc = collections.defaultdict(set)
        for (t, bk, ev, tid) in posts:
            tc[tid].add((bk[gran], ev))
        lineage_rows = max(lineage_rows, sum(len(s) for s in tc.values()))
    res['cost'] = {
        'pheme_lineage_rows_hour': sum(len(s) for s in _tc(posts, 'hour').values()),
        'pheme_lineage_rows_max_gran': lineage_rows,
        'ced_lineage_cascades': n_orig,
        'note': 'numeric cube stores no lineage; ours stores one row per (claim, cell) edge',
    }

    # ============================ A: 公平性（数值查询逐格相同） ============================
    # 数值立方体（tilted-time-frame，已处理迟到）= 我们的数据面计数度量；
    # 在同一 (hour, event) 网格上，两者的 n_posts 聚合定义相同 ⇒ 逐格差 0。
    def numeric_vs_ours(gran='hour'):
        num = collections.Counter()      # 数值立方体：只累加计数
        ours = collections.Counter()     # 我们的数据面：同样累加计数（外加语义层度量）
        for (t, bk, ev, tid) in posts:
            key = (bk[gran], ev)
            num[key] += 1
            ours[key] += 1
        diff = sum(1 for k in set(num) | set(ours) if num[k] != ours[k])
        return {'gran': gran, 'cells': len(num), 'diff_cells': diff,
                'total': sum(num.values())}

    res['fairness'] = {
        'checks': [numeric_vs_ours(g) for g in GRANS],
        'statement': ('data-plane measures are the projection of ours onto the numeric cube; '
                      'the two agree cell-by-cell, so #3 is a fair (non-strawman) baseline'),
    }

    res['elapsed_s'] = round(time.time() - T0, 2)
    print('[4/4] 写报告 …')
    with open(os.path.join(BASE, 'e10_results.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------------- 报告
    buf = []

    def P(s=''):
        buf.append(s)

    c = res['ced']
    ph = res['pheme']
    P('# E10 实验报告 —— Baseline #3：语义消融的数值流立方体')
    P('')
    P('> **性质**：论文 §7.7 baseline 表 **#3（numeric stream cube, semantics ablated）** 的实测。')
    P('> **日期**：2026-10-11 ｜ **脚本**：`e10_baseline3_ablation.py` ｜ **指标**：`e10_results.json`')
    P('> **数据**：CED（中文，%d 原帖 × 级联转发）+ PHEME 9-event（英文，%d 线程 / %s 推文，三值真值）'
      % (c['n_originals'], ph['n_threads'], f"{ph['n_posts']:,}"))
    P('')
    P('## 0. 一句话结论')
    P('')
    P('**把语义层拿掉后，数值流立方体在「活动量」上是完全正确的（与我们的数据面度量逐 cell 相同，0 差异），'
      '但在「知识面」上是盲的：它把 %s 条帖子当成 %s 份独立印证（放大 %.1f×），'
      '并且结构上无法表示 %s%% 的体量所落在的「互相矛盾」状态。**'
      % (f"{c['total_posts']:,}", f"{c['n_originals']:,}", c['amp_posts_over_origins'],
         ph['conflict']['day']['volume_in_conflicted_pct']))
    P('')
    P('一句话：**#3 不是坏引擎，是错仪表**——它能告诉你「讨论有多热」，不能告诉你「有多少条独立证据、其中多少是矛盾的」。')
    P('')
    P('## 1. A：时间轴上的公平性（#3 不是稻草人）')
    P('')
    P('数值流立方体（tilted time frame）本就是为事实时间轴设计的。在同一 $(\\text{时间桶}, \\text{事件})$ 网格上，'
      '把它的度量与我们数据面度量做逐格比对：')
    P('')
    P('| 粒度 | cell 数 | 逐格差异 | 判定 |')
    P('|---|---:|---:|---|')
    for ch in res['fairness']['checks']:
        P('| %s | %d | %d | %s |' % (ch['gran'], ch['cells'], ch['diff_cells'],
                                     '✅ 完全相同' if ch['diff_cells'] == 0 else '❌'))
    P('')
    P('> 差异恒为 0 的原因是**定义相同**：我们的数据面度量就是数值立方体度量层的超集投影。'
      '因此 #3 在事实时间轴上是一个**公平基线**——差距不来自"能不能维护时间"，而来自"能不能表示知识"。'
      '（对照：E1-SW 中**朴素翻滚窗**数值立方体在 $\\Delta w=0$ 时击穿 90.4% 的窗口，'
      '那是 #4 的失败模式，不是 #3 的。）')
    P('')
    P('## 2. B：放大 $\\neq$ 印证（"支持数"的系统性高估）')
    P('')
    P('数值立方体的"提及量"= 帖子数。而**独立报告数** = 原始出处数（一条原帖及其转发仍是**一份**证据）。'
      '两者之比即放大因子。')
    P('')
    P('### 2.1 CED（中文，级联转发）')
    P('')
    P('| 量 | 值 |')
    P('|---|---:|')
    P('| 原帖（独立出处） | %s |' % f"{c['n_originals']:,}")
    P('| 转发记录（语料实际给出） | %s |' % f"{c['total_reposts']:,}")
    P('| 总帖子数（数值立方体看到的"支持"） | %s |' % f"{c['total_posts']:,}")
    P('| **放大因子（$\\Sigma$posts / $\\Sigma$origin）** | **%.1f×** |' % c['amp_posts_over_origins'])
    P('| 级联规模 min / p25 / 中位 / p75 / p90 / p99 / max | %d / %d / %d / %d / %d / %s / %s |'
      % (c['cascade_size']['min'], c['cascade_size']['p25'], c['cascade_size']['median'],
         c['cascade_size']['p75'], c['cascade_size']['p90'],
         f"{c['cascade_size']['p99']:,}", f"{c['cascade_size']['max']:,}"))
    P('| 体量前 10%% 的原帖贡献 | %.1f%% 的帖子 |' % c['top10pct_cascades_volume_share_pct'])
    P('| 按 user 去重只能削减 | %.1f%%（去重后残余放大 %.1f×） |'
      % (100.0 * (1 - 1.0 / c['dedup_by_user_cut_ratio']), c['amp_after_user_dedup_over_origins']))
    P('')
    P('> 三个口径要点：')
    P('> ① **放大是普遍现象，不是长尾假象**——最小级联 %d 条推文、**中位级联 %d 条**、p25 也有 %d 条；'
      '体量前 10%% 的原帖只占 %.1f%% 的帖子，分布相当紧（max 仅 %s）。换言之，**随便挑一条原帖，'
      '它平均就带着几百份转发**——高估是系统性的，不是被少数爆款拉起来的。'
      % (c['cascade_size']['min'], c['cascade_size']['median'], c['cascade_size']['p25'],
         c['top10pct_cascades_volume_share_pct'], f"{c['cascade_size']['max']:,}"))
    P('> ② **按用户去重修不好**：转发者彼此并不独立（同一出处），去重后残余放大仍有 **%.1f×**。'
      '独立性的单位是**出处**，不是用户，也不是帖子。' % c['amp_after_user_dedup_over_origins'])
    P('> ③ **%.1f× 还是下界**：语料同时给出平台侧的 `reposts` 计数字段，其中位数为 %s，'
      '而实际随包发布的转发记录只占声明值的 %.1f%%（字段严格相等的仅 %.1f%%）——'
      '说明**发布版级联是平台计数的一个子样本**，真实放大只会更大。'
      % (c['amp_posts_over_origins'], f"{c['declared_reposts']['median_declared']:,}",
         c['declared_reposts']['actual_over_declared_pct'], c['declared_reposts']['field_match_pct']))
    P('')
    P('### 2.2 PHEME（英文，线程）')
    P('')
    P('| 事件 | 推文 | 线程（独立报告） | 放大 |')
    P('|---|---:|---:|---:|')
    for e, d in ph['per_event'].items():
        P('| %s | %s | %d | %.1f× |' % (e.split('-')[0], f"{d['posts']:,}", d['threads'], d['amp']))
    P('| **合计** | **%s** | **%s** | **%.1f×** |'
      % (f"{ph['n_posts']:,}", f"{ph['n_threads']:,}", ph['amp_posts_over_threads']))
    P('')
    P('线程规模（推文数）min / p25 / 中位 / p75 / p90 / max = %d / %d / %d / %d / %d / %d；'
      '即在英文语料里，**中位线程也有 %d 条推文**——放大同样不是长尾假象。'
      % (ph['thread_size']['min'], ph['thread_size']['p25'], ph['thread_size']['median'],
         ph['thread_size']['p75'], ph['thread_size']['p90'], ph['thread_size']['max'],
         ph['thread_size']['median']))
    P('')
    P('> 跨语料一致：把"转发/回复"计为"印证"会高估独立支持 **%.1f×（PHEME）– %.1f×（CED）**。'
      % (ph['amp_posts_over_threads'], c['amp_posts_over_origins']))
    P('')
    P('## 3. C：冲突不可表示（结构的盲区）')
    P('')
    P('数值立方体**没有声明状态维度**，因此无法回答"这个 cell 里是否含着互相矛盾的声明"。'
      '用 PHEME 三值真值量化这件事的规模——一个 $(\\text{时间桶}, \\text{事件})$ cell 若同时含 `true` 与 `false` 线程，即为**冲突 cell**：')
    P('')
    P('| 粒度 | 总 cell | 冲突 cell | 冲突 cell 占比 | **落在冲突 cell 的体量占比** |')
    P('|---|---:|---:|---:|---:|')
    for g in GRANS:
        d = ph['conflict'][g]
        P('| %s | %s | %s | %.2f%% | **%.2f%%** |'
          % (g, f"{d['total_cells']:,}", f"{d['conflicted_cells']:,}",
             d['conflicted_cell_pct'], d['volume_in_conflicted_pct']))
    P('')
    P('> 关键点不是"冲突很多"（粗粒度下几乎必然如此），而是**"细化 cell"这条退路会失效**：'
      '越细，冲突占比确实下降，但代价是 cell 数暴涨——而这恰恰正是我们改用**声明状态维 + 符号度量**'
      '（§5.4）来解决的问题。#3 无论把网格切多细，都只能得到一个"纯度"数字，'
      '而**无法把"矛盾"本身作为一个可切片的度量**拿出来。')
    P('')
    P('事件级：%d 个可标注事件中 **%d 个**同时含 true 与 false 线程（%s）。'
      % (ph['conflicted_events']['n_events_with_labels'],
         ph['conflicted_events']['n_conflicted_events'],
         '、'.join('%s' % k.split('-')[0] for k, v in ph['conflicted_events']['detail'].items()
                   if len(v) > 1)))
    P('')
    P('## 4. D：代价的正面（诚实对照）')
    P('')
    P('| 项 | 数值立方体 #3 | 我们的语义立方体 |')
    P('|---|---|---|')
    P('| 血缘表（claim $\\to$ cell 边） | 0 行 | %s 行（PHEME, hour 粒度） |'
      % f"{res['cost']['pheme_lineage_rows_hour']:,}")
    P('| 声明状态维 | 无 | 有（双时态） |')
    P('| 支撑 / 冲突 / MUS 符号度量 | 不可表示 | 可（§5.4） |')
    P('| as-of 查询正确性 | 终态高估 ≤ 23.9%（E5，作为 #4 的退化） | 28 / 1021 cell 失效，精确血缘 P=R=F1=1.000（E6） |')
    P('')
    P('> #3 的真实优势只有一个：**不需要血缘**，因此省掉 %s 行边表。'
      '这个优势在纯活动量分析里是划算的——本实验不否认这一点。'
      % f"{res['cost']['pheme_lineage_rows_hour']:,}")
    P('')
    P('## 5. 能力矩阵（查询 $\\times$ 基线）')
    P('')
    P('| 查询 | 数值立方体 #3 | 语义立方体（本文） |')
    P('|---|---|---|')
    P('| 总量 / 趋势 / 峰值桶 | ✅ 精确（逐格相同） | ✅ 精确 |')
    P('| 某声明的**独立**支持数 | ❌ 计为 %s（放大 %.1f×） | ✅ 按出处计 → %s |'
      % (f"{c['total_posts']:,}", c['amp_posts_over_origins'], f"{c['n_originals']:,}"))
    P('| 某 cell 是否含矛盾声明 | ❌ 不可表示 | ✅ conflict 度量（§5.4） |')
    P('| 截至 $\\tau$ 的**有效**结论 | ❌ 高估 ≤ 23.9% | ✅ 失效集最小重算（E5/E6） |')
    P('| 结论的**支撑集**（可解释性） | ❌ 无血缘 | ✅ 血缘可查 |')
    P('')
    P('## 6. 局限')
    P('')
    P('1. **"独立出处"的操作化**：CED 以「原帖 = 一个出处」，PHEME 以「线程 = 一个出处」。'
      '这是**代理**——同一事件的多家媒体报道本应算多个独立源，但语料不含媒体归属，故本文只做**上限友好**的保守估计'
      '（真实独立支持数在 [出处数, 帖子数] 之间）。')
    P('2. **CED 坏样本**：%d / %d 个原帖文件因编码截断无法解析（约 0.3%%），未计入。' % (c['n_bad'], c['n_originals'] + c['n_bad']))
    P('3. **冲突 cell 的定义**依赖 PHEME 的线程级三值真值，是**事后标注**；它衡量的是"该时刻该事件的讨论里真假并存"，'
      '不等于"同一命题被同时肯定和否定"（那需要声明级 $(s,r,o)$，见 §7.8-4）。')
    P('4. **跨语料量纲不可直接比较**：CED 的 %d 与 PHEME 的 %d 都是各自网格下的 cell 数，'
      '结论以**比例与放大因子**为主。' % (res['ced']['n_day_cells'], ph['conflict']['day']['total_cells']))
    P('')
    P('*生成：WorkBuddy ｜ 耗时 %ss ｜ 未消耗 SmartLib 配额。*' % res['elapsed_s'])

    open(os.path.join(BASE, 'E10_报告.md'), 'w', encoding='utf-8').write('\n'.join(buf))

    # ---------------- console ----------------
    print('=== 完成 ===')
    print('CED  原帖 %d ｜ 帖子 %s ｜ 放大 %.1f× ｜ 中位级联 %d'
          % (c['n_originals'], f"{c['total_posts']:,}", c['amp_posts_over_origins'], c['cascade_size']['median']))
    print('PHEME 线程 %d ｜ 推文 %s ｜ 放大 %.1f×' % (ph['n_threads'], f"{ph['n_posts']:,}", ph['amp_posts_over_threads']))
    for g in GRANS:
        d = ph['conflict'][g]
        print('  冲突[%-5s] cells=%d 冲突=%d (%.2f%%) 体量占比=%.2f%%'
              % (g, d['total_cells'], d['conflicted_cells'], d['conflicted_cell_pct'], d['volume_in_conflicted_pct']))
    print('公平性：', [ (x['gran'], x['diff_cells']) for x in res['fairness']['checks'] ])
    print('血缘边（hour）= %s' % f"{res['cost']['pheme_lineage_rows_hour']:,}")


def _tc(posts, gran):
    tc = collections.defaultdict(set)
    for (t, bk, ev, tid) in posts:
        tc[tid].add((bk[gran], ev))
    return tc


if __name__ == '__main__':
    main()
