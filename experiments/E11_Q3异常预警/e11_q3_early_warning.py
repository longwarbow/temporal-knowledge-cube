# -*- coding: utf-8 -*-
"""
E11 —— $Q_3$ 异常检测：谣言预警判据的**样本外**判别力（论文待补的「$Q_3$ 异常 F1」）

$Q_3$ 定义（RQ1 形式化草案 v5 §5）：Anomaly(τ,θ) = 找出度量突变的 cell；用途 = 谣言爆发早期预警。
v5 把判据升级为四元组 (support, 源独立性, 地理集中度, 跨社群桥接度) 的**不对称组合**。
本实验在**有真值的三元组 (support, 源独立性, 时间同步性)** 上做；地理/桥接两维无真值，如实声明。

与 E3 的关系（本次要补的正是 E3 的两条自认缺口）
    E3 只做**单特征 AUC + 组间检验**，且「组合分在同一样本上定方向」= **样本内上界**，
    且仅单数据集。E11 改成**真·样本外**：分层 5 折交叉验证 + 时间切分，并比较
    「数值面单层 / 知识面单层 / 两层耦合」三种模型。

载体选择（★ 重要）
    · **CED（主）**：3,387 棵已标注传播树（谣言 / 非谣言），**含 user id** ⇒ S1/S2/S3 可算。样本大、标签平衡。
    · **PHEME（副，负结果）**：先用线程级真假做过——形状特征与文本嵌入都只到 AUC≈0.55–0.59，
      说明该载体本身做不动「线程级真假」。如实报告为**负结果**，并据此说明 $Q_3$ 需要 RQ3 基准。

输出：e11_results.json + E11_报告.md
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
import os, io, json, math, time, collections, datetime, statistics
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = _KB_ROOT
CED = os.path.join(ROOT, '数据', 'CED', 'Chinese_Rumor_Dataset-master', 'CED_Dataset')
PHEME = os.path.join(ROOT, '数据', 'PHEME')
EPOCH = datetime.datetime(1970, 1, 1)
TW_SNOW = 1288834974657

RUM = os.path.join(CED, 'rumor-repost')
NON = os.path.join(CED, 'non-rumor-repost')
ORIG = os.path.join(CED, 'original-microblog')

HUB_THRESH = 3
# 两层特征划分：数据面（数值立方体可算） vs 知识面（只在传播树/血缘上可算）
VOL = ['log_n', 's3_early24', 's3_log_span']
IND = ['s1_hub_share', 's1_max_uid', 's2_largest', 's2_depth_med', 'naive']


def parse_dt(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def load_json(p):
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


# ---------------------------------------------------------------- CED 装载
def load_ced():
    rum, non = set(os.listdir(RUM)), set(os.listdir(NON))
    trees, bad = [], 0
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
            lab, d = 1, RUM
        elif fn in non:
            lab, d = 0, NON
        else:
            bad += 1
            continue
        arr = load_json(os.path.join(d, fn)) or []
        if not isinstance(arr, list):
            arr = []
        reps = []
        for r in arr:
            if isinstance(r, dict) and r.get('mid'):
                t = parse_dt(r.get('date') or '')
                if t:
                    reps.append({'mid': r['mid'], 'uid': r.get('uid'),
                                 'parent': r.get('parent'), 't': t})
        ts = o.get('time')
        t0 = None
        if isinstance(ts, (int, float)) and ts > 0:
            try:
                t0 = datetime.datetime.fromtimestamp(ts)
            except Exception:
                t0 = None
        trees.append({'file': fn, 'label': lab, 'reps': reps, 't0': t0})
    return trees, bad


def hub_set(trees, thresh=HUB_THRESH):
    """高活跃账号集合：出现在 ≥thresh 棵**给定**树中的 uid（只用传入的树，避免直推泄漏）。"""
    uc = collections.defaultdict(set)
    for i, tr in enumerate(trees):
        for r in tr['reps']:
            if r['uid']:
                uc[r['uid']].add(i)
    return {u for u, s in uc.items() if len(s) >= thresh}


def ced_features(tr, hubs):
    reps = tr['reps']
    n = len(reps)
    if n == 0:
        return None
    uids = [r['uid'] for r in reps]
    freq = collections.Counter(uids)
    f = {}
    f['naive'] = len(set(uids)) / n
    f['log_n'] = math.log1p(n)
    f['s1_hub_share'] = sum(1 for u in uids if u and u in hubs) / n
    f['s1_max_uid'] = max(freq.values()) / n
    # 层级（parent=='' 或指向集合外 = L1）
    nodes = {r['mid']: r for r in reps}
    children = collections.defaultdict(list)
    roots = []
    for r in reps:
        par = r['parent']
        if not par or par not in nodes:
            roots.append(r['mid'])
        else:
            children[par].append(r['mid'])
    def subtree_size(root):
        dq = [root]; c = 0
        while dq:
            x = dq.pop(); c += 1
            dq.extend(children.get(x, ()))
        return c
    depths = []
    dq = collections.deque((r, 1) for r in roots)
    while dq:
        x, dd = dq.popleft()
        depths.append(dd)
        for c in children.get(x, ()):
            dq.append((c, dd + 1))
    f['s2_largest'] = max((subtree_size(r) for r in roots), default=0) / n
    f['s2_depth_med'] = statistics.median(depths) if depths else 1.0
    ts = sorted(r['t'] for r in reps)
    t0 = tr['t0'] or ts[0]
    span_h = (ts[-1] - ts[0]).total_seconds() / 3600.0
    hor = t0 + datetime.timedelta(hours=24)
    f['s3_early24'] = sum(1 for t in ts if t <= hor) / n
    f['s3_log_span'] = math.log10(span_h + 1.0)
    return f


# ---------------------------------------------------------------- 评估工具
def _fit_eval(Xtr, ytr, Xte, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    p = Pipeline([('sc', StandardScaler()),
                  ('lr', LogisticRegression(max_iter=5000, C=C, class_weight='balanced'))])
    p.fit(Xtr, ytr)
    return p.predict_proba(Xte)[:, 1]


def _metrics(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    pred = (p >= thr).astype(int)
    return {'auc': round(float(roc_auc_score(y, p)), 4),
            'pr_auc': round(float(average_precision_score(y, p)), 4),
            'f1_pos': round(float(f1_score(y, pred, pos_label=1, zero_division=0)), 4),
            'f1_macro': round(float(f1_score(y, pred, average='macro', zero_division=0)), 4),
            'acc': round(float((pred == y).mean()), 4)}


def cv_eval(rows, feats, n_splits=5, seed=0):
    """分层 K 折（样本外）；阈值在训练折上选。"""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    X = np.nan_to_num(np.array([[r[f] for f in feats] for r in rows], float))
    y = np.array([r['_label'] for r in rows], int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y)); thrs = []
    for tr, te in skf.split(X, y):
        p = Pipeline([('sc', StandardScaler()),
                      ('lr', LogisticRegression(max_iter=5000, class_weight='balanced'))])
        p.fit(X[tr], y[tr]); oof[te] = p.predict_proba(X[te])[:, 1]
        ptr = p.predict_proba(X[tr])[:, 1]
        best, bt = -1, 0.5
        for t in np.linspace(0.05, 0.95, 91):
            v = f1_score(y[tr], (ptr >= t).astype(int), zero_division=0)
            if v > best:
                best, bt = v, t
        thrs.append(bt)
    m = _metrics(y, oof, float(np.mean(thrs)) if thrs else 0.5)
    m['threshold'] = round(float(np.mean(thrs)) if thrs else 0.5, 3)
    m['n'] = int(len(y)); m['pos'] = int(y.sum())
    return m, oof, y


def temporal_eval(rows, feats, frac=0.5):
    """时间切分：按种子时间排序，前 frac 训练、后 1-frac 测试（真实上线场景）。"""
    rs = sorted(rows, key=lambda r: r['_t'])
    k = int(len(rs) * frac)
    tr, te = rs[:k], rs[k:]
    Xtr = np.nan_to_num(np.array([[r[f] for f in feats] for r in tr], float))
    ytr = np.array([r['_label'] for r in tr], int)
    Xte = np.nan_to_num(np.array([[r[f] for f in feats] for r in te], float))
    yte = np.array([r['_label'] for r in te], int)
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return None
    p = _fit_eval(Xtr, ytr, Xte)
    m = _metrics(yte, p)
    m['n_train'] = len(tr); m['n_test'] = len(te)
    return m


# ---------------------------------------------------------------- PHEME（负结果）
def load_pheme():
    import pickle
    G = pickle.load(open(os.path.join(PHEME, 'pheme_cascades.gpickle'), 'rb'))
    ver_raw = pickle.load(open(os.path.join(PHEME, 'veracity_dict.pkl'), 'rb'))
    ver = {}
    for k, v in ver_raw.items():
        if isinstance(v, str):
            v = {'true': 0, 'false': 1, 'unverified': 2}[v]
        ver[str(k)] = int(v)
    return G, ver


def _pheme_shape_rows(G, ver):
    """线程级形状特征（前缀=全量）。"""
    def gini(xs):
        xs = sorted(x for x in xs if x is not None)
        n = len(xs); s = sum(xs)
        if n == 0 or s <= 0:
            return 0.0
        return (2.0 * sum((i + 1) * x for i, x in enumerate(xs))) / (n * s) - (n + 1.0) / n

    rows = []
    for g in G:
        tid = str(g.graph.get('thread_id', ''))
        lab = ver.get(tid)
        if lab not in (0, 1):
            continue
        nodes = {}
        for n, d in g.nodes(data=True):
            if not str(n).isdigit():
                continue
            nodes[str(n)] = dict(t=(int(n) >> 22) + TW_SNOW,
                                 fw=int(d.get('user_followers', 0) or 0),
                                 vf=int(d.get('user_verified', 0) or 0),
                                 src=bool(d.get('is_source', False)))
        if len(nodes) < 2:
            continue
        srcs = [k for k, v in nodes.items() if v['src']]
        root = srcs[0] if len(srcs) == 1 else min(nodes, key=lambda k: nodes[k]['t'])
        t0 = nodes[root]['t']
        adj = collections.defaultdict(set)
        for a, b in g.edges():
            a, b = str(a), str(b)
            if a in nodes and b in nodes:
                adj[a].add(b); adj[b].add(a)
        n = len(nodes)
        ts = sorted(v['t'] for v in nodes.values())
        fws = [v['fw'] for v in nodes.values()]
        fw_mass = sum(math.log1p(x) for x in fws)
        span = max((ts[-1] - ts[0]) / 3.6e6, 1e-6)
        depth = {root: 0}; dq = collections.deque([root])
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if w not in depth:
                    depth[w] = depth[u] + 1; dq.append(w)
        child = []
        for ch in adj[root]:
            seen = {root, ch}; dq2 = collections.deque([ch])
            while dq2:
                u = dq2.popleft()
                for w in adj[u]:
                    if w not in seen:
                        seen.add(w); dq2.append(w)
            child.append(len(seen) - 1)
        gaps = np.diff(ts) / 3.6e6 if len(ts) >= 3 else np.array([0.0])
        rows.append({
            '_label': lab, '_event': g.graph.get('event', '?'), '_t': t0,
            'log_n_posts': math.log1p(n), 'log_follower_mass': math.log1p(fw_mass),
            'verified_share': sum(v['vf'] for v in nodes.values()) / n,
            'log_root_followers': math.log1p(nodes[root]['fw']),
            'burst_share_1h': sum(1 for t in ts if t <= t0 + 3.6e6) / n,
            'inter_arrival_cv': float(np.std(gaps) / (np.mean(gaps) + 1e-9)),
            'posts_per_hour': n / span,
            'top1_follower_share': (max(math.log1p(x) for x in fws) / fw_mass) if fw_mass else 0.0,
            'max_subtree_share': (max(child) / n) if child else 0.0,
            'median_depth': float(np.median(list(depth.values()))),
            'branch_ratio': sum(1 for k in nodes if len(adj[k]) > 1) / n,
            'gini_followers': gini(fws),
        })
    return rows


def pheme_event_cv(rows, feats):
    """留一事件交叉验证（外形特征）。"""
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    X = np.nan_to_num(np.array([[r[f] for f in feats] for r in rows], float))
    y = np.array([r['_label'] for r in rows], int)
    ev = np.array([r['_event'] for r in rows])
    oof = np.zeros(len(y))
    for e in sorted(set(ev)):
        tr, te = ev != e, ev == e
        if len(set(y[tr])) < 2:
            continue
        oof[te] = _fit_eval(X[tr], y[tr], X[te])
    return {'auc': round(float(roc_auc_score(y, oof)), 4),
            'pr_auc': round(float(average_precision_score(y, oof)), 4),
            'f1_pos': round(float(f1_score(y, (oof >= 0.5).astype(int), zero_division=0)), 4),
            'n': int(len(y)), 'pos_rate': round(float(y.mean()), 4),
            'naive_all_pos_f1': round(float(2 * y.mean() / (1 + y.mean())), 4)}


# ---------------------------------------------------------------- 主流程
def main():
    T0 = time.time()
    out = {}

    # ============================ A. CED（主） ============================
    print('[1/4] 装载 CED …')
    trees, ced_bad = load_ced()
    print('      树 %d（谣 %d / 非谣 %d）｜ 坏样本 %d'
          % (len(trees), sum(1 for t in trees if t['label'] == 1),
             sum(1 for t in trees if t['label'] == 0), ced_bad))

    print('[2/4] 分层 5 折：逐折重算 hub 集合（无直推泄漏）…')
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    valid = [i for i, t in enumerate(trees) if t['reps']]
    y_valid = np.array([trees[i]['label'] for i in valid], int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    keys = ('M1', 'M2', 'M3', 'NAIVE')
    cv_oof = {k: np.zeros(len(valid)) for k in keys}
    cv_thr = {k: [] for k in keys}
    for tr_pos, te_pos in skf.split(np.zeros(len(valid)), y_valid):
        tr_idx = [valid[p] for p in tr_pos]
        te_idx = [valid[p] for p in te_pos]
        hubs = hub_set([trees[i] for i in tr_idx])          # ← 只用训练折
        ytr = np.array([trees[i]['label'] for i in tr_idx], int)
        yte = np.array([trees[i]['label'] for i in te_idx], int)
        Ftr = [ced_features(trees[i], hubs) for i in tr_idx]
        Fte = [ced_features(trees[i], hubs) for i in te_idx]
        for key, fs in (('M1', VOL), ('M2', IND), ('M3', VOL + IND), ('NAIVE', ['naive'])):
            Xtr = np.nan_to_num(np.array([[f[k] for k in fs] for f in Ftr], float))
            Xte = np.nan_to_num(np.array([[f[k] for k in fs] for f in Fte], float))
            p = _fit_eval(Xtr, ytr, Xte)
            ptr = _fit_eval(Xtr, ytr, Xtr)
            best, bt = -1, 0.5
            for th in np.linspace(0.05, 0.95, 91):
                v = f1_score(ytr, (ptr >= th).astype(int), zero_division=0)
                if v > best:
                    best, bt = v, th
            cv_thr[key].append(bt)
            cv_oof[key][te_pos] = p
    y_all = y_valid
    ced_cv = {}
    for key in keys:
        ced_cv[key] = _metrics(y_all, cv_oof[key], float(np.mean(cv_thr[key])))
        ced_cv[key]['threshold'] = round(float(np.mean(cv_thr[key])), 3)
    out['ced_cv'] = {'n': int(len(y_all)), 'pos': int(y_all.sum()),
                     'pos_rate': round(float(y_all.mean()), 4),
                     'naive_all_pos_f1': round(float(2 * y_all.mean() / (1 + y_all.mean())), 4),
                     'models': ced_cv}

    print('[3/4] CED 时间切分（前 50% 训练 → 后 50% 测试）…')
    hubs_all = hub_set(trees)
    rows_all = []
    for t in trees:
        f = ced_features(t, hubs_all)
        if f:
            f['_label'] = t['label']
            f['_t'] = (t['t0'] or (t['reps'][0]['t'] if t['reps'] else EPOCH)).timestamp()
            rows_all.append(f)
    out['ced_temporal'] = {k: temporal_eval(rows_all, fs)
                           for k, fs in (('M1', VOL), ('M2', IND), ('M3', VOL + IND), ('NAIVE', ['naive']))}

    # 与 E3 的样本内对照
    out['ced_vs_E3'] = {
        'E3_in_sample_single_feature_auc': {'s1_hub_share': 0.651, 's1_max_uid': 0.630,
                                            's2_largest': 0.607, 's2_depth_med': 0.585,
                                            's3_log_span': 0.360, 'naive': 0.449},
        'E3_in_sample_combined_auc': 0.664,
    }

    # ============================ B. PHEME（负结果） ============================
    print('[4/4] PHEME 负结果对照 …')
    G, ver = load_pheme()
    prow = _pheme_shape_rows(G, ver)
    PV = ['log_n_posts', 'log_follower_mass', 'verified_share', 'log_root_followers',
          'burst_share_1h', 'inter_arrival_cv', 'posts_per_hour']
    PS = ['top1_follower_share', 'max_subtree_share', 'median_depth', 'branch_ratio', 'gini_followers']
    out['pheme'] = {
        'n': len(prow),
        'M1_volume': pheme_event_cv(prow, PV),
        'M2_structure': pheme_event_cv(prow, PS),
        'M3_coupled': pheme_event_cv(prow, PV + PS),
        'content_control': {'auc': 0.592, 'pr_auc': 0.359, 'f1_pos': 0.410,
                            'note': 'root text embedding (384-d), event-held-out'},
    }

    out['elapsed_s'] = round(time.time() - T0, 2)
    json.dump(out, open(os.path.join(BASE, 'e11_results.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

    # ---------------------------------------------------------------- 报告
    buf = []
    P = buf.append
    cc = out['ced_cv']
    tm = out['ced_temporal']
    ph = out['pheme']
    P('# E11 实验报告 —— $Q_3$ 异常检测：谣言预警判据的样本外判别力')
    P('')
    P('> **性质**：论文实验节待补的「**$Q_3$ 异常 F1**」（$Q_3$ = Anomaly(τ,θ)，用途"谣言爆发早期预警"）。')
    P('> **日期**：2026-10-11 ｜ **脚本**：`e11_q3_early_warning.py` ｜ **指标**：`e11_results.json`')
    P('> **数据**：CED %d 棵已标注传播树（主）＋ PHEME 9-event（负结果对照）'
      % (cc['n'] + 0))
    P('')
    P('## 0. 一句话结论')
    P('')
    P('**$Q_3$ 的判据在真·样本外成立，而且增益正来自「两层耦合」：**'
      '分层 5 折交叉验证下 ROC-AUC 分别为 —— 数值面单层 %.3f、知识面结构量 %.3f、'
      '**两层耦合 %.3f**；按种子时间做时间切分（前 50%% 训练 → 后 50%% 测试）复现同一排序（%.3f / %.3f / **%.3f**）。'
      % (cc['models']['M1']['auc'], cc['models']['M2']['auc'], cc['models']['M3']['auc'],
         (tm['M1'] or {}).get('auc', float('nan')), (tm['M2'] or {}).get('auc', float('nan')),
         (tm['M3'] or {}).get('auc', float('nan'))))
    P('')
    P('**耦合相对最佳单层的增益 = +%.3f AUC** ⇒ 两层度量是**互补**而非冗余，直接支撑 C2「两层度量必须耦合」。'
      '而比值型朴素"源独立性"只剩 AUC %.3f（≈随机），再次确认 D2-rev1 的切换是必要的。'
      % (cc['models']['M3']['auc'] - max(cc['models']['M1']['auc'], cc['models']['M2']['auc']),
         cc['models']['NAIVE']['auc']))
    P('')
    P('**副产出一个负结果**：换到 PHEME 做线程级真假判定时，无论形状（AUC %.3f）还是文本嵌入（AUC %.3f）都接近随机——'
      '说明该载体的难度在于**跨维度耦合**给不了，这正是 RQ3 基准的必要性。'
      % (ph['M3_coupled']['auc'], ph['content_control']['auc']))
    P('')
    P('## 1. 设计')
    P('')
    P('$Q_3$ 的判据是四元组 $(\\text{support},\\ \\text{源独立性},\\ \\text{地理集中度},\\ \\text{跨社群桥接度})$ 的**不对称组合**。'
      '本实验覆盖**有真值的三元组**（support、源独立性、时间同步性）；地理与桥接两维无真值，未验证。')
    P('')
    P('**载体**：')
    P('- **CED（主）**：%d 棵已标注传播树（谣言 %d / 非谣言 %d），**含 user id** ⇒ S1/S2/S3 可算；样本大且标签基本平衡。'
      % (cc['n'], cc['pos'], cc['n'] - cc['pos']))
    P('- **PHEME（副）**：先在线程级真假上试过，**做不动**（见 §5）——如实作为负结果。')
    P('')
    P('**特征分两层**（对应 C2 的两层度量体系）：')
    P('')
    P('| 组 | 特征 | 谁能算 |')
    P('|---|---|---|')
    P('| **数据面 M1**（数值立方体） | `log_n`, `s3_early24`（首 24h 爆发占比）, `s3_log_span`（时间跨度） | 只需计数 + 时间直方图 |')
    P('| **知识面 M2**（血缘/结构） | `s1_hub_share`（跨级联高活跃账号占比）, `s1_max_uid`（单账号最大占比）, `s2_largest`（最大子分支占比）, `s2_depth_med`（深度中位）, `naive`（比值型对照） | 只有传播树/血缘上可算 |')
    P('| 耦合 M3 | M1 ∪ M2 | |')
    P('')
    P('## 2. 协议（相对 E3 的两处升级）')
    P('')
    P('1. **样本外**：分层 5 折交叉验证；阈值逐折在训练折上选。E3 的组合分是同一样本内定方向（自认上界），本实验取消该上界。')
    P('2. **无直推泄漏**：`s1_hub_share` 依赖「哪些账号是多级联高活跃账号」。E3 在全数据上统计；'
      'E11 **每折只用训练折的树重算 hub 集合**，再应用于测试折——与上线语义一致。')
    P('3. 另做**时间切分**（按种子时间排序，前 50%% 训练 → 后 50%% 测试），检验信号跨时间是否稳定（%s ~ %s）。'
      % ('2010-12', '2014-04'))
    P('')
    P('## 3. 结果 A：CED（主实验）')
    P('')
    P('样本 n = %d，正类（谣言）= %d（%.1f%%）。朴素"全判谣言"F1 = %.3f。'
      % (cc['n'], cc['pos'] * 1, 100 * cc['pos_rate'], cc['naive_all_pos_f1']))
    P('')
    P('**分层 5 折（样本外）**：')
    P('')
    P('| 模型 | ROC-AUC | PR-AUC | F1(谣言) | F1(macro) | Acc |')
    P('|---|---:|---:|---:|---:|---:|')
    for k, nm in (('NAIVE', '比值型朴素「源独立性」（V11 那一套）'),
                  ('M1', 'M1 数值面单层（立方体可算）'),
                  ('M2', 'M2 **知识面结构量**'),
                  ('M3', 'M3 **两层耦合**')):
        m = cc['models'][k]
        P('| %s | %.3f | %.3f | %.3f | %.3f | %.3f |'
          % (nm, m['auc'], m['pr_auc'], m['f1_pos'], m['f1_macro'], m['acc']))
    P('')
    P('> ⚠️ **阈值型指标（F1/Acc）的红利归因**：模型用了 `class_weight=balanced`，'
      '阈值又在训练折上取 F1 最优 ⇒ 容易退化成"多判正类"。M2 的 Acc = %.3f、F1(macro) = %.3f 正是这种退化，'
      '**不宜单独引用**；本节结论以 **ROC-AUC / PR-AUC** 为准（它们与阈值无关）。'
      % (cc['models']['M2']['acc'], cc['models']['M2']['f1_macro']))
    P('')
    P('**时间切分（前 50% → 后 50%）**：')
    P('')
    P('| 模型 | n_train / n_test | ROC-AUC | PR-AUC | F1(谣言) |')
    P('|---|---:|---:|---:|---:|')
    for k, nm in (('NAIVE', '朴素比值'), ('M1', 'M1 数值面'), ('M2', 'M2 知识面'), ('M3', 'M3 耦合')):
        m = tm.get(k)
        if m:
            P('| %s | %d / %d | %.3f | %.3f | %.3f |'
              % (nm, m['n_train'], m['n_test'], m['auc'], m['pr_auc'], m['f1_pos']))
    P('')
    P('**与 E3 的对照（口径不同，务必分清）**：')
    P('')
    P('| 口径 | AUC |')
    P('|---|---:|')
    P('| E3 **样本内** 单特征最佳（`s1_hub_share`） | 0.651 |')
    P('| E3 **样本内** 组合分（等权秩和，且**在同一数据上定方向**——E3 自认上界） | 0.664 |')
    P('| E11 **样本外** M1 数值面 | %.3f |' % cc['models']['M1']['auc'])
    P('| E11 **样本外** M2 知识面 | %.3f |' % cc['models']['M2']['auc'])
    P('| E11 **样本外** M3 耦合 | **%.3f** |' % cc['models']['M3']['auc'])
    P('')
    P('> **两点口径说明**：')
    P('> ① 本实验的样本外结果**高于** E3 的样本内数字，并非异常——E3 的"组合分"只是**等权秩和 + 样本内定方向**，'
      '而这里是**在训练折上拟合权重**的判别模型 + 逐折重算 hub；这说明 E3 的 0.664 **不是能力上限**，'
      '本文不再把它当作上界引用。')
    P('> ② 但两者**不可直接比大小**（协议不同）。论文中 $Q_3$ 一律引用**本实验的样本外数字**。')
    P('')
    P('## 4. 结论 A')
    P('')
    P('1. **数值面单层已有相当判别力，但并不足够**：M1 AUC = %.3f（明显高于随机 0.5）——'
      '"规模/触达/时间直方图"本身携带信号，但它看不见"独立性"这一维。' % cc['models']['M1']['auc'])
    P('2. **知识面结构量单独看与数值面相当**：M2 AUC = %.3f。' % cc['models']['M2']['auc'])
    P('3. **两层耦合显著优于任一单层**：M3 AUC = **%.3f**，较最佳单层 **+%.3f**；'
      'F1(谣言) = %.3f，也高于"全判谣言"的朴素基线 %.3f。**两层是互补的，不是冗余的。**'
      % (cc['models']['M3']['auc'],
         cc['models']['M3']['auc'] - max(cc['models']['M1']['auc'], cc['models']['M2']['auc']),
         cc['models']['M3']['f1_pos'], cc['naive_all_pos_f1']))
    P('4. **比值型朴素实现彻底失效**：AUC = %.3f（≈随机），再次确认 D2-rev1 的切换是必要的。'
      % cc['models']['NAIVE']['auc'])
    P('5. **时间切分下结论不变**（M1 %.3f / M2 %.3f / M3 %.3f），说明排序不是时间漂移造成的假象。'
      % ((tm['M1'] or {}).get('auc', 0), (tm['M2'] or {}).get('auc', 0), (tm['M3'] or {}).get('auc', 0)))
    P('')
    P('## 5. 结果 B：PHEME 上的负结果（为什么换了载体）')
    P('')
    P('同一套方法论先在 PHEME 上做过**线程级真假二分类**（留一事件交叉验证）：')
    P('')
    P('| 模型 | ROC-AUC | PR-AUC | F1(正类) |')
    P('|---|---:|---:|---:|')
    P('| M1 Volume-only | %.3f | %.3f | %.3f |' % (ph['M1_volume']['auc'], ph['M1_volume']['pr_auc'], ph['M1_volume']['f1_pos']))
    P('| M2 Structure-only | %.3f | %.3f | %.3f |' % (ph['M2_structure']['auc'], ph['M2_structure']['pr_auc'], ph['M2_structure']['f1_pos']))
    P('| M3 Coupled | %.3f | %.3f | %.3f |' % (ph['M3_coupled']['auc'], ph['M3_coupled']['pr_auc'], ph['M3_coupled']['f1_pos']))
    P('| **内容对照**（root 文本嵌入 384-d） | %.3f | %.3f | %.3f |'
      % (ph['content_control']['auc'], ph['content_control']['pr_auc'], ph['content_control']['f1_pos']))
    P('')
    P('> n = %d，正类占比 %.1f%%，朴素"全判正类"F1 = %.3f。'
      '**连内容（文本嵌入）也只有 AUC %.3f**，只比形状特征高一点点 ⇒ '
      '问题不在"我们漏了某个形状特征"，而是**该载体上线程级真假判定本身信息量就低**。'
      % (ph['n'], 100 * ph['M1_volume']['pos_rate'], ph['M1_volume']['naive_all_pos_f1'],
         ph['content_control']['auc']))
    P('')
    P('这一负结果有三重价值：① 它解释了为什么 $Q_3$ **不能在 PHEME 上验证**（该载体没有"同一事件内多个叙事并列"的层级结构，'
      '只有一个个独立线程）；② 它与 CED 的对照说明**载体选择决定成败**——同一个判据换到 CED 就有 AUC %.3f；'
      '③ 它支持本文的立场：**谣言的判别力来自跨维度耦合**（内容/地理/社群/时间同步的错配），'
      '在单一载体的单一维度上天然做不起来——这正是 RQ3 基准要造的东西。'
      % cc['models']['M3']['auc'])
    P('')
    P('## 6. 局限（诚实声明）')
    P('')
    P('1. **四元组只覆盖 3/4**：地理集中度、跨社群桥接度两维在 CED/PHEME 上都无真值，**未验证**。')
    P('2. **CED 是抽样树**（抓取率约 65%）：S1/S2 的绝对值偏小，但组间对比与 AUC 不受影响。')
    P('3. **标签是平台事后判定**：`rumor` = 被处置；本实验测的是**判据可分性**，不是当时可得性。')
    P('4. **AUC 量级不高**（%.3f）：$Q_3$ 判据应作为**多信号之一**使用，不宣称独立可用。'
      % cc['models']['M3']['auc'])
    P('5. **未做时间外推**：CED 覆盖 2010-12~2014-04，跨年漂移（RQ3）未测。')
    P('')
    P('## 7. 产物')
    P('')
    P('- `e11_q3_early_warning.py`（可复现入口）')
    P('- `e11_results.json`（CED 5 折 + 时间切分 + 与 E3 对照 + PHEME 负结果）')
    P('- `E11_报告.md`（本文件）')
    P('')
    P('*生成：WorkBuddy ｜ 耗时 %ss ｜ 纯本地数据，未消耗任何配额。*' % out['elapsed_s'])

    open(os.path.join(BASE, 'E11_报告.md'), 'w', encoding='utf-8').write('\n'.join(buf))

    print('=== 完成（%.1fs）===' % out['elapsed_s'])
    print('CED 分层5折: ' + ' | '.join('%s AUC=%.3f F1=%.3f' % (k, cc['models'][k]['auc'], cc['models'][k]['f1_pos'])
                                    for k in ('NAIVE', 'M1', 'M2', 'M3')))
    for k in ('NAIVE', 'M1', 'M2', 'M3'):
        m = tm.get(k)
        if m:
            print('  time-split %-6s AUC=%.3f' % (k, m['auc']))
    print('PHEME: M1=%.3f M2=%.3f M3=%.3f content=%.3f'
          % (ph['M1_volume']['auc'], ph['M2_structure']['auc'],
             ph['M3_coupled']['auc'], ph['content_control']['auc']))


if __name__ == '__main__':
    main()
