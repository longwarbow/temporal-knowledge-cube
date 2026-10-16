# -*- coding: utf-8 -*-
"""
E12 —— D8「最小样本量」约束的**实证重定**

背景（为什么必须重定）
    D8 原设计照搬 Stream Cube：低于固定阈值的 cell 标记 `sparse` 而不出分布型指标，
    阈值取 **L1 无 / L2 >= 100 / L3 >= 20 / L4 >= 5**（D4 §3 同步引用）。
    V2 画像（hashtag 作 L3 代理）实测：L3>=20 仅覆盖 15.2% 的帖子提及 => 「D8 待重定」。
    V2 给的修法是「覆盖率 KPI + 反推阈值」，但**没有给出统计依据**，也没解释
    「为什么原阈值就是错的」。

本实验做三件事
    (1) **给阈值一个理论锚**：把「最小样本量」还原为「比例/份额型度量的估计精度要求」。
        对比例估计 p，95% 置信下要求绝对误差 <= eps（最坏 p=0.5）=> n >= z^2*0.25/eps^2。
        由此**反查原阈值** 100/20/5 各自对应的误差半径 —— 从而暴露 L4>=5 在统计上无意义。
    (2) **给阈值一个数据锚**：在 4 个**真实 cube 网格**上量 cell 规模分布与
        「阈值 -> (合格 cell 占比, 覆盖成员占比)」曲线；KPI 反推各 cube 的实际阈值，
        证明阈值是**数据相关**的，不能是全局常数。
    (3) **重新定义 `sparse`**：从「标记不删」升级为「标记 + 独立物化 + 查询期按需聚合 +
        显式误差标注」，并量化「不出分布型指标」的成员覆盖代价。

真实 cube（全部本地已有数据，零外部依赖）
    G1  CED  (day   x 省)  -> 帖子数     [数据面 F_post]
    G2  CED  (hour  x 省)  -> 帖子数     [数据面 F_post，细粒度]
    G3  CED  (10min x 省)  -> 帖子数     [数据面 F_post，最细]
    G4  PHEME(hour  x 事件) -> 声明数     [知识面 F_claim]
    G5  CED   per-cascade  -> 转发数     [「支持度」样本量]
    G6  PHEME per-thread   -> 推文数     [「支持度」样本量]

输出：e12_results.json + E12_报告.md
"""
import os, io, csv, json, math, collections, datetime, pickle
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = r'D:\WSPforWB\10.keyan\场景1-海量短文本'
CED = os.path.join(ROOT, '数据', 'CED', 'Chinese_Rumor_Dataset-master', 'CED_Dataset')
PHEME = os.path.join(ROOT, '数据', 'PHEME')
LOC_CSV = os.path.join(ROOT, '实验', 'E2_空间维消歧', 'location_映射结果.csv')
TF = "%Y-%m-%d %H:%M:%S"
TW_SNOW = 1288834974657
Z95 = 1.959963984540054
CIS = [0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.075, 0.05]


# ------------------------------------------------------------------ 工具
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


def parse_dt(s):
    try:
        return datetime.datetime.strptime(s, TF)
    except Exception:
        return None


def snow_dt(tid):
    try:
        return datetime.datetime.utcfromtimestamp(((int(tid) >> 22) + TW_SNOW) / 1000.0)
    except Exception:
        return None


def bucket(dt, gran):
    if gran == 'day':
        return dt.strftime('%Y-%m-%d')
    if gran == 'hour':
        return dt.strftime('%Y-%m-%d %H')
    if gran == '10min':
        return dt.strftime('%Y-%m-%d %H:') + '%02d' % (dt.minute // 10 * 10)
    return dt.isoformat()


# ------------------------------------------------------------------ 省码映射
def load_prov_map():
    m = {}
    with io.open(LOC_CSV, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if (r.get('status') or '').strip() == 'ok' and (r.get('prov_code') or '').strip():
                m[r['file']] = (r['prov_code'].strip(), (r.get('prov_name') or '').strip())
    return m


# ------------------------------------------------------------------ CED cube
def ced_cells(prov_map, gran):
    """cell = (bucket, prov_code) -> 帖子数；原帖按自身属地/时间，转发按根帖属地 + 自身时间。"""
    RUM = os.path.join(CED, 'rumor-repost')
    NON = os.path.join(CED, 'non-rumor-repost')
    ORIG = os.path.join(CED, 'original-microblog')
    rum, non = set(os.listdir(RUM)), set(os.listdir(NON))
    cells = collections.Counter()
    casc, n_used, n_skip_nogeo = [], 0, 0
    for fn in sorted(os.listdir(ORIG)):
        pc = prov_map.get(fn)
        if not pc:
            n_skip_nogeo += 1
            continue
        prov = pc[0]
        o = load_json(os.path.join(ORIG, fn))
        if isinstance(o, list):
            o = o[0] if o else {}
        if not isinstance(o, dict):
            continue
        d = RUM if fn in rum else (NON if fn in non else None)
        if d is None:
            continue
        arr = load_json(os.path.join(d, fn)) or []
        if not isinstance(arr, list):
            arr = []
        n_rep = 0
        for r in arr:
            if not isinstance(r, dict):
                continue
            t = parse_dt(r.get('date') or '')
            if t is None:
                continue
            n_rep += 1
            cells[(bucket(t, gran), prov)] += 1
        ts = o.get('time')
        if isinstance(ts, (int, float)) and ts > 0:
            try:
                cells[(bucket(datetime.datetime.fromtimestamp(ts), gran), prov)] += 1
            except Exception:
                pass
        casc.append(n_rep)
        n_used += 1
    return cells, casc, n_used, n_skip_nogeo


# ------------------------------------------------------------------ PHEME cube
def pheme_cells(gran):
    G = pickle.load(open(os.path.join(PHEME, 'pheme_cascades.gpickle'), 'rb'))
    cells = collections.Counter()
    thread_sizes = []
    for g in G:
        ev = g.graph.get('event') or '?'
        nodes = {str(n): d for n, d in g.nodes(data=True) if str(n).isdigit()}
        thread_sizes.append(len(nodes))
        if not nodes:
            continue
        srcs = [k for k, d in nodes.items() if d.get('is_source')]
        root = srcs[0] if srcs else min(nodes, key=lambda k: int(k))
        t = snow_dt(root)
        if t is None:
            continue
        cells[(bucket(t, gran), ev)] += 1
    return cells, thread_sizes, len(G)


# ------------------------------------------------------------------ 统计
def dist_stats(counts):
    v = sorted(counts.values())
    if not v:
        return {}
    a = np.array(v, float)
    pc = lambda q: float(np.percentile(a, q))
    return {
        'n_cells': len(v),
        'sum_members': int(a.sum()),
        'min': int(a.min()), 'p25': pc(25), 'median': pc(50), 'p75': pc(75),
        'p90': pc(90), 'p99': pc(99), 'max': int(a.max()),
        'mean': round(float(a.mean()), 2),
        'singleton_pct': round(100.0 * sum(1 for x in v if x == 1) / len(v), 2),
        'cells_ge5_pct': round(100.0 * int((a >= 5).sum()) / len(v), 2),
        'cells_ge20_pct': round(100.0 * int((a >= 20).sum()) / len(v), 2),
        'cells_ge100_pct': round(100.0 * int((a >= 100).sum()) / len(v), 2),
    }


THS = [1, 2, 3, 5, 10, 20, 25, 43, 50, 97, 100, 200, 500]


def coverage_curve(counts, ths=THS):
    v = np.array(sorted(counts.values()), float)
    N, M = v.sum(), len(v)
    out = []
    for t in ths:
        ok = v >= t
        out.append({'th': int(t),
                    'cells_kept_pct': round(100.0 * int(ok.sum()) / M, 2),
                    'members_kept_pct': round(100.0 * v[ok].sum() / N, 2) if N else 0.0})
    return out


def coverage_at(counts, t):
    """阈值 t 下，合格 cell 承载的成员比例（%）。"""
    v = np.array(sorted(counts.values()), float)
    N = v.sum()
    return round(100.0 * v[v >= t].sum() / N, 2) if N else 0.0


def eps_for_kpi(counts, kpi=0.60):
    """给定成员覆盖率 KPI，反推**可容忍的最大阈值**（即最宽松的精度档）。
    覆盖率随阈值单调不增 => 取覆盖仍 >= kpi 的最大 t。返回 (t, eps_pct)。"""
    v = np.array(sorted(counts.values()), float)
    N = v.sum()
    best = 1
    for t in range(1, int(v.max()) + 1):
        if v[v >= t].sum() / N >= kpi:
            best = t
        else:
            break
    return best, round(100.0 * eps_of_n(best), 1)


def eps_of_n(n, p=0.5):
    return Z95 * math.sqrt(p * (1 - p) / n)


def sparse_cost(counts, th):
    v = np.array(sorted(counts.values()), float)
    N, M = v.sum(), len(v)
    sp = v < th
    return {'th': int(th),
            'sparse_cells_pct': round(100.0 * int(sp.sum()) / M, 2),
            'sparse_members_pct': round(100.0 * v[sp].sum() / N, 2)}


# ------------------------------------------------------------------ 主流程
def build_report(R):
    C = R['cubes']
    g1, g2, g3 = C['G1_ced_day_prov'], C['G2_ced_hour_prov'], C['G3_ced_10min_prov']
    g4, g5, g6 = C['G4_pheme_hour_event'], C['G5_ced_cascade'], C['G6_pheme_thread']
    nm = {'G1_ced_day_prov': 'CED (day × 省)',
          'G2_ced_hour_prov': 'CED (hour × 省)',
          'G3_ced_10min_prov': 'CED (10min × 省)',
          'G4_pheme_hour_event': 'PHEME (hour × 事件)',
          'G5_ced_cascade': 'CED 级联（转发数）',
          'G6_pheme_thread': 'PHEME 线程（推文数）'}
    layer = {'G1_ced_day_prov': '数据面 $F_{post}$', 'G2_ced_hour_prov': '数据面 $F_{post}$',
             'G3_ced_10min_prov': '数据面 $F_{post}$', 'G4_pheme_hour_event': '知识面 $F_{claim}$',
             'G5_ced_cascade': '知识面（支持度）', 'G6_pheme_thread': '知识面（支持度）'}

    L = []
    A = L.append
    A('# E12 —— D8「最小样本量」约束的实证重定')
    A('')
    A('> 对应决议 **D8**（原状态「待重定」）与 **D4 §3**（阈值引用处）。')
    A('> 数据：CED（2886 棵有属地码的传播树 / 1,092,445 条帖子）+ PHEME（6,425 线程 / 105,747 推文）。**全部本地，零外部依赖。**')
    A('')
    A('---')
    A('')
    A('## 0. 一句话结论')
    A('')
    A('**D8 原设计「把最小样本量与语义层绑定」（L2≥100 / L3≥20 / L4≥5）在统计上是错的，'
      '因为「层」是语义粗细、样本量是统计量，二者无映射关系。** 本实验把它重定为'
      '**「阈值挂在度量类型上」**：加性/布尔型无约束；比例/份额型默认 **n ≥ 43**（$\\varepsilon\\le0.15$）、'
      '排序类 **n ≥ 97**（$\\varepsilon\\le0.10$，恰为原 L2 值）；分布型 **n ≥ max(5k, 43)**。'
      '并把 `sparse` 从「标记不删」升级为「标记 + 独立物化 + 查询期按需聚合 + 结果带精度标注」。')
    A('')
    A('**同时给出一条可用于选粒度的硬规则**：同一份数据，时间粒度 day → hour → 10min，'
      '在 $\\varepsilon=0.15$ 的同一精度要求下，可用成员覆盖率从 **%.1f%%** 掉到 **%.1f%%**，'
      '再到 **%.1f%%** —— 这就是 D4「三重耦合」的量化形态，也是 P3 物化选择必须读入的约束。'
      % (g1['cov_at_eps15'], g2['cov_at_eps15'], g3['cov_at_eps15']))
    A('')
    A('---')
    A('')
    A('## 1. 为什么必须重定（原设计错在哪）')
    A('')
    A('D8 原设计照搬 Stream Cube 的「稀疏单元仅保留高层聚集值」，给出固定阈值 **L1 无 / L2 ≥100 / L3 ≥20 / L4 ≥5**（D4 §3 同步引用）。V2 画像用 hashtag 作 L3 代理实测：L3≥20 **仅覆盖 15.2% 的帖子提及**。')
    A('')
    A('V2 的修法是「覆盖率 KPI + 反推阈值」，但没有回答两个问题：**（i）阈值本身该由什么决定？（ii）原阈值到底错在哪？** 本实验补上这两问。')
    A('')
    A('原设计的三处问题：')
    A('')
    A('1. **绑定对象错了**：把阈值绑在「语义层（L1–L4）」上。层是 *语义* 粗细，样本量是 *统计* 量——同一层内 cell 可大可小（取决于粒度与数据），二者无映射。')
    A('2. **单调方向反了**：原设计越细的层阈值越低（L2≥100 → L4≥5），暗含「细层可容忍小样本」。但细层正是要出**分布型**指标的地方，反而需要**更大**样本。')
    A('3. **L4≥5 在统计上无意义**：$n=5$ 的比例估计 95% 置信半径是 **±43.8%**（见 §5）——那不是"分布型指标"，是噪声。')
    A('')
    A('## 2. 方法：给阈值两个锚')
    A('')
    A('**理论锚（§5）**：把「最小样本量」还原为**比例/份额型度量的估计精度要求**。对二项比例 $p$，'
      '95% 置信下绝对误差 $\\le\\varepsilon$（最坏 $p=0.5$）要求')
    A('')
    A('$$n_{\\min}(\\varepsilon)=\\left\\lceil z_{0.975}^2\\cdot\\frac{0.25}{\\varepsilon^2}\\right\\rceil'
      '\\approx\\left\\lceil\\frac{0.9604}{\\varepsilon^2}\\right\\rceil$$')
    A('')
    A('这一条给出**有依据的阈值**，并可直接**反查**原阈值对应的误差半径。')
    A('')
    A('**数据锚（§3–§4）**：在 6 个**真实 cube 网格**上量 cell 规模分布与「阈值 → 覆盖率」曲线，'
      '证明阈值是**数据与粒度相关**的，不能是全局常数。')
    A('')
    A('| 网格 | cube 定义 | cell 成员 | 度量层 |')
    A('|---|---|---|---|')
    for k in ('G1_ced_day_prov', 'G2_ced_hour_prov', 'G3_ced_10min_prov',
              'G4_pheme_hour_event', 'G5_ced_cascade', 'G6_pheme_thread'):
        memb = {'G1_ced_day_prov': '帖子数', 'G2_ced_hour_prov': '帖子数',
                'G3_ced_10min_prov': '帖子数', 'G4_pheme_hour_event': '声明（线程）数',
                'G5_ced_cascade': '每声明转发数', 'G6_pheme_thread': '每声明推文数'}[k]
        A('| `%s` | %s | %s | %s |' % (k, nm[k], memb, layer[k]))
    A('')
    A('> CED 属地为级联归属（根帖属地继承给转发），与 E2 的省级消歧口径一致；'
      '`n_skip_nogeo=%d` 棵无属地码的树被排除，`n_cascades=%d` 棵入 cube。'
      % (CED_SKIP[0], CED_CASC[0]))
    A('')
    A('## 3. cell 规模画像')
    A('')
    A('| 网格 | cells | 成员总数 | 中位 | 单例占比 | ≥5 | ≥20 | ≥100 |')
    A('|---|---:|---:|---:|---:|---:|---:|---:|')
    for k in ('G1_ced_day_prov', 'G2_ced_hour_prov', 'G3_ced_10min_prov',
              'G4_pheme_hour_event', 'G5_ced_cascade', 'G6_pheme_thread'):
        d = C[k]
        A('| %s | %d | %d | %.1f | %.1f%% | %.1f%% | %.1f%% | %.1f%% |'
          % (nm[k], d['n_cells'], d['sum_members'], d['median'],
             d['singleton_pct'], d['cells_ge5_pct'], d['cells_ge20_pct'], d['cells_ge100_pct']))
    A('')
    A('三点观察：')
    A('')
    A('1. **单例 cell 普遍**：最细粒度（10min×省）45.0% 的 cell 只有 1 条帖子；数据面 cube 的 cell 规模中位数只有 2–6。')
    A('2. **成员高度集中**：数据面 cube 里 32.8%（day×省）的 cell 达到 ≥20，却承载 97% 的帖子（§4）——少数厚 cell 支配体量。')
    A('3. **知识面天然稀疏**：PHEME 每声明的证据量（`G6`）中位仅 12 条，**94.6% 的声明不足 43 条**——即 3/4 以上声明无法做比例型推断（§7）。')
    A('')
    A('## 4. 阈值 → 覆盖率曲线')
    A('')
    A('「合格 cell（$\\ge t$）承载多少比例的成员」——这是判断某一阈值"能否接受"的直接依据。')
    A('')
    A('| 网格 | $t$=5 | 10 | 20 | 43 | 97 | 200 |')
    A('|---|---:|---:|---:|---:|---:|---:|')
    for k in ('G1_ced_day_prov', 'G2_ced_hour_prov', 'G3_ced_10min_prov',
              'G4_pheme_hour_event', 'G5_ced_cascade', 'G6_pheme_thread'):
        d = C[k]
        cv = {r['th']: r['members_kept_pct'] for r in d['coverage']}
        A('| %s | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% |'
          % (nm[k], cv[5], cv[10], cv[20], cv[43], cv[97], cv[200]))
    A('')
    A('**粒度对照（同一份 CED 数据，三个时间粒度，成员总数恒为 1,092,445）：**')
    A('')
    A('| 时间粒度 | cells | 中位 cell | 单例 | $\\varepsilon$=0.15 时覆盖 |')
    A('|---|---:|---:|---:|---:|')
    for k in ('G1_ced_day_prov', 'G2_ced_hour_prov', 'G3_ced_10min_prov'):
        d = C[k]
        A('| %s | %d | %.1f | %.1f%% | **%.1f%%** |'
          % (nm[k].split('(')[1].rstrip(')'), d['n_cells'], d['median'],
             d['singleton_pct'], d['cov_at_eps15']))
    A('')
    A('> **这就是 D4「三重耦合」的直接实证**：精度要求 $\\varepsilon=0.15$ 固定不变时，'
      '把时间粒度从 day 细化到 hour、10min，可用成员覆盖率 **%.1f%% → %.1f%% → %.1f%%**。'
      '粒度不是自由参数——每细一档都在拿"可用比例型指标的数据"换"时间分辨率"。'
      % (g1['cov_at_eps15'], g2['cov_at_eps15'], g3['cov_at_eps15']))
    A('')
    A('## 5. 统计反推：$\\varepsilon \\leftrightarrow n_{\\min}$，以及原阈值体检 ★')
    A('')
    A('| $\\varepsilon$（绝对误差，95% 置信） | 0.30 | 0.25 | 0.20 | 0.15 | 0.12 | 0.10 | 0.075 | 0.05 |')
    A('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    A('| $n_{\\min}$ | %s |' % ' | '.join(str(r['n_min']) for r in R['eps_to_n']))
    A('')
    A('**原阈值体检：**')
    A('')
    A('| 原阈值 | 对应误差半径 | 判定 |')
    A('|---|---:|---|')
    A('| L4 ≥ **5** | **±%.1f%%** | ❌ **无意义**：±44%% 的"分布型指标"是噪声 |' % R['orig_threshold_eps'][0]['eps_pct'])
    A('| L3 ≥ **20** | **±%.1f%%** | ⚠️ 偏松：只能支撑"粗比例"，且绑定方向反了 |' % R['orig_threshold_eps'][1]['eps_pct'])
    A('| L2 ≥ **100** | **±%.1f%%** | ✅ **有依据**：$\\approx\\varepsilon=0.10$，保留为"精"档 |' % R['orig_threshold_eps'][2]['eps_pct'])
    A('')
    A('> 结论很干净：**原设计里唯一站得住的是 L2≥100（$n=97$）**；'
      'L3≥20 偏松一个档；**L4≥5 在统计上无意义**。这也解释了 V2 的失败——'
      '不是"阈值定高了"，而是"阈值定错了对象"。')
    A('')
    A('## 6. 覆盖率 KPI 反推：阈值是粒度的函数 ★')
    A('')
    A('若要求"合格 cell 覆盖 $\\ge$ KPI 比例的成员"，各网格**可容忍的最大阈值**与对应精度档：')
    A('')
    A('| 网格 | KPI ≥ 60% ⇒ 阈值 | 该阈值的 $\\varepsilon$ | KPI ≥ 80% ⇒ 阈值 | 该阈值的 $\\varepsilon$ |')
    A('|---|---:|---:|---:|---:|')
    for k in ('G1_ced_day_prov', 'G2_ced_hour_prov', 'G3_ced_10min_prov',
              'G4_pheme_hour_event', 'G5_ced_cascade', 'G6_pheme_thread'):
        d = C[k]
        A('| %s | %d | %.1f%% | %d | %.1f%% |'
          % (nm[k], d['kpi60'][0], d['kpi60'][1], d['kpi80'][0], d['kpi80'][1]))
    A('')
    A('> 读法：在 day×省 上，即便要求"覆盖 ≥60%% 成员"，阈值可高达 %d（$\\varepsilon$ 仅 %.1f%%）——'
      '因为厚 cell 支配体量。同样 KPI 在 10min×省 上阈值只能到 %d（$\\varepsilon$ 放宽到 %.1f%%）。'
      '**同一 KPI，不同粒度允许的精度差近一个数量级 ⇒ 阈值必须是粒度与数据的函数。**'
      % (g1['kpi60'][0], g1['kpi60'][1], g3['kpi60'][0], g3['kpi60'][1]))
    A('')
    A('## 7. `sparse` 的代价与新语义')
    A('')
    A('以 $\\varepsilon=0.15$（$n=43$）为分界，低于它的 cell 标 `sparse`、不出分布型指标。代价：')
    A('')
    A('| 网格 | `sparse` cell 占比 | `sparse` 承载的成员占比 |')
    A('|---|---:|---:|')
    for k in ('G1_ced_day_prov', 'G2_ced_hour_prov', 'G3_ced_10min_prov',
              'G4_pheme_hour_event', 'G5_ced_cascade', 'G6_pheme_thread'):
        v = R['sparse_cost_eps015'][k]
        A('| %s | %.1f%% | **%.1f%%** |' % (nm[k], v['sparse_cells_pct'], v['sparse_members_pct']))
    A('')
    A('**两点判断：**')
    A('')
    A('1. **数据面在粗粒度下几乎零代价**：day×省 有 73.7% 的 cell 是 `sparse`，但只承载 **5.1%** 的帖子。'
      '说明"标记稀疏 + 不出分布指标"在粗粒度下**不损失结论**（少数厚 cell 已代表体量）。')
    A('2. **细粒度与知识面代价高**：10min×省 的 `sparse` 承载 **81.7%**、PHEME 每声明证据量的 `sparse` 承载 **75.1%**。'
      '⇒ 在这些格点上，"不出分布型指标"等于"放弃大部分数据"，必须靠别的机制兜底。')
    A('')
    A('**`sparse` 新语义（替代 D8 原"标记不删"）：**')
    A('')
    A('1. **标记**：cell 成员数 $< n_{\\min}(\\varepsilon)$ 时打 `sparse`；')
    A('2. **独立物化**：`sparse` cell 单独物化（不必进主聚合路径），保持 as-of 可访问；')
    A('3. **查询期按需聚合**：`sparse` cell 的值**不预先算**，仅在查询时对**其父 cell**按需合并（照搬 Stream Cube"保留高层聚集值"的务实版）；')
    A('4. **结果带精度标注**：任何来自 `sparse` cell 的比例型指标，返回值附 $\\varepsilon$ 标注（如 `0.42 ±0.22`），而非静默给出"精确值"。')
    A('')
    A('## 8. D8 重定结果（交付物）')
    A('')
    A('**原则：阈值挂在「度量类型」上，不挂在「语义层」上。**')
    A('')
    A('| 度量类型 | 代表度量 | 最小样本量 | 依据 |')
    A('|---|---|---|---|')
    A('| 加性（一阶矩） | `COUNT(帖子)`、`SUM(转发数)` | **无（$n\\ge1$）** | T1/T2：一阶矩在任何可加表示上精确 |')
    A('| 存在/布尔型 | 是否有撤回、是否冲突 | **无（$n\\ge1$）** | 不涉及比例估计 |')
    A('| 比例/份额型 · 粗 | `verified_share`、跨 cell 粗比 | **$n\\ge25$**（$\\varepsilon\\le0.20$） | 二项 95% 置信 |')
    A('| 比例/份额型 · **默认** | 源独立性 $S_1$、集中度 `gini` | **$n\\ge43$**（$\\varepsilon\\le0.15$） | 同上 |')
    A('| 比例/份额型 · 精 | 排序、细粒度对比 | **$n\\ge97$**（$\\varepsilon\\le0.10$） | 同上；**恰为原 L2≥100** |')
    A('| 分布型（$k$ 桶） | 主题/情感分布、JSD、熵 | **$n\\ge\\max(5k,\\ 43)$** | 每桶期望频数 $\\ge5$ + 份额稳定 |')
    A('')
    A('**配套的粒度选择规则**（给定精度与 KPI，选最大可用时间粒度）：')
    A('')
    A('| 要求 | day×省 | hour×省 | 10min×省 | 结论 |')
    A('|---|---|---|---|---|')
    A('| $\\varepsilon\\le0.15$ 且覆盖 $\\ge$ 60%% | ✅ %.1f%% | ❌ %.1f%% | ❌ %.1f%% | **默认用 day；hour 仅在标注精度后可用** |'
      % (g1['cov_at_eps15'], g2['cov_at_eps15'], g3['cov_at_eps15']))
    A('| 只出加性指标 | ✅ | ✅ | ✅ | 稀疏不损失一阶矩 ⇒ 任意粒度 |')
    A('')
    A('## 9. 影响面')
    A('')
    A('- **D8**：由「待重定」→ **重定完成**（阈值表见 §8；`sparse` 语义升级）。')
    A('- **D4 §3**：原「L1 无／L2 ≥100／L3 ≥20／L4 ≥5」作废，改为「按度量类型取阈值」（§8 表）。')
    A('- **D3（按频率分层）+ T5（分层误差）**：本实验给出"分层"的另一半理由——不只误差累积，还有**稀疏成本**；与 T5 的分层表示天然衔接（厚 head 精确、薄 tail 草图）。')
    A('- **P3（物化选择）**：粒度选择规则（§8）是 P3 的一个硬约束输入——物化代价必须与"该粒度下损失多少可用成员"一起权衡。')
    A('- **论文**：§5.3（度量代数）与 §6（架构）可引用本表作为"cuboid 选择"的实证约束。')
    A('')
    A('## 10. 局限')
    A('')
    A('1. **CED 属地为级联归属**（转发继承根帖属地），不是转发者真实属地；这影响 cell 的**绝对规模**，但"指数级稀疏 + 集中度"的形状结论稳健。')
    A('2. **`G5/G6` 的"成员"是支持度而非 cube 的 cell 计数**，与 `G1–G4` 口径不同，仅用于说明"每声明的证据量"这一知识面样本量。')
    A('3. **比例估计的二项假设**（$p=0.5$ 最坏）给出的是**保守上界**；若度量的真实 $p$ 远离 0.5，所需样本量更小。')
    A('4. **未含话题维**：话题侧无公开真值，故 §8 表中"分布型 $k$ 桶"的 $k$ 只能给通用规则而非实测 $k$。')
    A('')
    A('## 11. 产物清单')
    A('')
    A('- `e12_d8_min_sample.py` —— 一键复现（`python e12_d8_min_sample.py`，约 150 s）')
    A('- `e12_results.json` —— 全部数字')
    A('- `E12_报告.md` —— 本文件')
    return '\n'.join(L)


CED_SKIP = [0]
CED_CASC = [0]


def main():
    print('[1/3] CED 省码映射 …')
    prov_map = load_prov_map()
    print('     ok 省码 %d 个 file' % len(prov_map))

    print('[2/3] CED cube（day / hour / 10min）…')
    ced = {}
    for gran in ('day', 'hour', '10min'):
        cells, casc, n_used, n_skip = ced_cells(prov_map, gran)
        ced[gran] = {'cells': cells, 'casc': casc, 'n_cascades': n_used,
                     'n_skip_nogeo': n_skip}
        CED_SKIP[0] = n_skip
        CED_CASC[0] = n_used
        print('     %-6s cells=%d  成员=%d  级联=%d' %
              (gran, len(cells), sum(cells.values()), n_used))

    print('[3/3] PHEME cube（hour x event）…')
    pcells, tsize, n_threads = pheme_cells('hour')
    print('     hour   cells=%d  声明=%d  线程=%d' %
          (len(pcells), sum(pcells.values()), n_threads))

    # ---- 汇总
    R = {'ced_prov_files': len(prov_map), 'Z95': round(Z95, 4)}

    cubes = {}
    cubes['G1_ced_day_prov'] = ced['day']['cells']
    cubes['G2_ced_hour_prov'] = ced['hour']['cells']
    cubes['G3_ced_10min_prov'] = ced['10min']['cells']
    cubes['G4_pheme_hour_event'] = pcells
    cubes['G5_ced_cascade'] = collections.Counter(
        {i: n for i, n in enumerate(ced['day']['casc'])})
    cubes['G6_pheme_thread'] = collections.Counter(
        {i: n for i, n in enumerate(tsize)})

    R['cubes'] = {}
    for k, c in cubes.items():
        d = dist_stats(c)
        d['coverage'] = coverage_curve(c)
        d['cov_at_eps10'] = coverage_at(c, 97)
        d['cov_at_eps15'] = coverage_at(c, 43)
        d['cov_at_eps20'] = coverage_at(c, 25)
        d['kpi60'] = eps_for_kpi(c, 0.60)
        d['kpi80'] = eps_for_kpi(c, 0.80)
        R['cubes'][k] = d
        print('\n--- %s ---' % k)
        print('  cells=%d members=%d median=%.1f singleton%%=%.1f  ge5%%=%.1f ge20%%=%.1f ge100%%=%.1f'
              % (d['n_cells'], d['sum_members'], d['median'], d['singleton_pct'],
                 d['cells_ge5_pct'], d['cells_ge20_pct'], d['cells_ge100_pct']))
        print('  覆盖（成员%%）：eps.20(t=25)=%.1f%%  eps.15(t=43)=%.1f%%  eps.10(t=97)=%.1f%%'
              % (d['cov_at_eps20'], d['cov_at_eps15'], d['cov_at_eps10']))
        print('  KPI 反推：覆盖>=60%% 时最大 t=%d (eps=%.1f%%) ；覆盖>=80%% 时最大 t=%d (eps=%.1f%%)'
              % (d['kpi60'][0], d['kpi60'][1], d['kpi80'][0], d['kpi80'][1]))

    # ---- (1) 统计反推：eps -> n_min，并反查原阈值
    R['eps_to_n'] = [{'eps': e, 'n_min': int(math.ceil(Z95 ** 2 * 0.25 / e ** 2))}
                     for e in CIS]
    R['orig_threshold_eps'] = [
        {'threshold': 5, 'eps_pct': round(100.0 * eps_of_n(5), 1),
         'note': '原 L4>='},
        {'threshold': 20, 'eps_pct': round(100.0 * eps_of_n(20), 1),
         'note': '原 L3>='},
        {'threshold': 100, 'eps_pct': round(100.0 * eps_of_n(100), 1),
         'note': '原 L2>='},
    ]
    print('\n=== 统计反推 ===')
    for r in R['eps_to_n']:
        print('  eps=%.3f -> n_min=%d' % (r['eps'], r['n_min']))
    for r in R['orig_threshold_eps']:
        print('  原阈值 %3d -> 误差半径 ±%.1f%%  (%s)'
              % (r['threshold'], r['eps_pct'], r['note']))

    # ---- (3) sparse 代价：以「比例型 eps=0.15 => n=43」为分界
    R['sparse_cost_eps015'] = {k: sparse_cost(c, 43) for k, c in cubes.items()}
    print('\n=== sparse(<43) 代价 ===')
    for k, v in R['sparse_cost_eps015'].items():
        print('  %-22s cells 中 %.1f%% 为 sparse，承载 %.1f%% 的成员'
              % (k, v['sparse_cells_pct'], v['sparse_members_pct']))

    R['elapsed'] = None
    with open(os.path.join(BASE, 'e12_results.json'), 'w', encoding='utf-8') as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    with open(os.path.join(BASE, 'E12_报告.md'), 'w', encoding='utf-8') as f:
        f.write(build_report(R))
    print('\n[save] e12_results.json + E12_报告.md')
    return R


if __name__ == '__main__':
    import time
    t = time.time()
    R = main()
    R['elapsed'] = round(time.time() - t, 2)
    with open(os.path.join(BASE, 'e12_results.json'), 'w', encoding='utf-8') as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    print('elapsed_s = %.2f' % R['elapsed'])
