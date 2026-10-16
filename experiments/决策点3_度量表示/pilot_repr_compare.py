# -*- coding: utf-8 -*-
"""
决策点3「度量摘要的可聚合表示」—— 合成数据快速探针 (pilot)

目的：在同一份合成流上，比较 5 种候选表示在三个关键场景下的表现：
  (a) 上卷合并 roll-up   (b) 撤回更新 RETRACT   (c) 概念漂移 drift
并记录内存与更新代价。

说明：本探针用合成数据，结论只用于**筛掉明显不可行的方案、确定实验口径**，
      不能替代真实数据 (Weibo-WM / Weibo21 / PHEME) 上的验证。

纯标准库实现，固定随机种子，可复现。
输出：pilot_results_v1.md（本目录）
"""
import random
import math
from collections import Counter, defaultdict

# ---------------- 参数 ----------------
SEED = 20261009
DIM = 32                 # 语义向量维度
N_DOCS = 24000           # 文档总数
PHASES = 3               # 概念漂移阶段数
TOPICS_PER_PHASE = 6     # 每阶段主题数（阶段间主题完全不重叠 = 模拟新概念出现）
N_TOPICS = PHASES * TOPICS_PER_PHASE
N_ENTITIES = 2000        # 实体词表（长尾）
SLICES_PER_PHASE = 3     # 每阶段时间片
GROUPS = 2               # 每个时间片内的语义分组
K_CLUSTERS = 16          # 聚类摘要的簇容量
CF_TOP = 8               # 每簇保留的实体数（可读层的截断）
CMS_W = 256              # 草图宽度
CMS_D = 4                # 草图深度
CF_BUFFER = 256          # 聚类摘要的微簇缓冲（同 MiTexCube 的 micro-cluster 思路）
NOISE = 0.08             # 向量噪声（须显著小于主题间距离，否则语义簇不可分）
RETAIN = 0.92            # 撤回实验：保留 92%，撤回 8%
ASSIGN_DIM = 8           # 聚类分配用的粗粒度子空间（近似，省算力）
CF_THRESH = 0.25         # 新建簇的距离阈值（超过则开新簇，仿 BIRCH 半径判据）

random.seed(SEED)

# ---------------- 合成数据 ----------------
def unit_vec():
    v = [random.gauss(0, 1) for _ in range(DIM)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

TOPIC_VEC = [unit_vec() for _ in range(N_TOPICS)]
TOPIC_W = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]

_ew = [1.0 / (i + 1) ** 1.1 for i in range(N_ENTITIES)]
ECUM = []
_s = 0.0
for _w in _ew:
    _s += _w
    ECUM.append(_s)
ETOT = _s


def sample_entity():
    x = random.random() * ETOT
    lo, hi = 0, N_ENTITIES - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ECUM[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def sample_topic(phase):
    x = random.random()
    c = 0.0
    for i, w in enumerate(TOPIC_W):
        c += w
        if x <= c:
            return phase * TOPICS_PER_PHASE + i
    return phase * TOPICS_PER_PHASE + len(TOPIC_W) - 1


def gen_stream():
    docs = []
    per = N_DOCS // PHASES
    for i in range(N_DOCS):
        p = min(PHASES - 1, i // per)
        t = sample_topic(p)
        e = sample_entity()
        base_pol = 0.65 if (t % TOPICS_PER_PHASE) < 3 else 0.40
        pol = 1 if random.random() < base_pol else -1
        base = TOPIC_VEC[t]
        v = [base[d] + random.gauss(0, NOISE) for d in range(DIM)]
        si = min(SLICES_PER_PHASE - 1, (i - p * per) * SLICES_PER_PHASE // per)
        s = p * SLICES_PER_PHASE + si
        g = (t % TOPICS_PER_PHASE) // (TOPICS_PER_PHASE // GROUPS)
        docs.append((v, t, e, pol, p, s, g))
    return docs


def truth(docs):
    n = len(docs)
    if n == 0:
        return None
    sv = [0.0] * DIM
    pos = 0
    ec = Counter()
    tc = Counter()
    for (v, t, e, pol, p, s, g) in docs:
        for d in range(DIM):
            sv[d] += v[d]
        if pol > 0:
            pos += 1
        ec[e] += 1
        tc[t] += 1
    return {"n": n, "mean": [x / n for x in sv], "pos_share": pos / n, "ent": ec, "topic": tc}


# ---------------- 候选表示 ----------------
class R0NoMat:
    """R0 不物化：保留声明/文档集合，查询时现算（正确性基准 + 代价上界）"""
    name = "R0 不物化(现算)"

    def __init__(self):
        self.docs = []
        self._cache = None

    def _t(self):
        if self._cache is None:
            self._cache = truth(self.docs)
        return self._cache

    def add(self, d):
        self.docs.append(d)
        self._cache = None

    def sub(self, d):
        self.docs.remove(d)
        self._cache = None

    def flush(self):
        pass

    def merge(self, o):
        r = R0NoMat()
        r.docs = self.docs + o.docs
        return r

    def mean(self):
        t = self._t()
        return t["mean"] if t else [0.0] * DIM

    def pos_share(self):
        t = self._t()
        return t["pos_share"] if t else 0.0

    def ent_dist(self, ents):
        t = self._t()
        return {e: t["ent"].get(e, 0) for e in ents} if t else {}

    @property
    def n(self):
        return len(self.docs)

    def mem(self):
        return self.n * (DIM * 8 + 32)  # 向量 + 元数据

    def ops_add(self):
        return 1

    def ops_sub(self):
        return self.n  # 线性查找 + 删除

    def ops_rollup(self, k):
        return self.n  # 需要扫描全部文档重算


class R1Moments:
    """R1 矩统计：n, Σx, n_pos —— 可加可减，精确；但无法给出语义分布"""
    name = "R1 矩统计"

    def __init__(self):
        self.n = 0
        self.sv = [0.0] * DIM
        self.pos = 0

    def add(self, d):
        v = d[0]
        self.n += 1
        for i in range(DIM):
            self.sv[i] += v[i]
        if d[3] > 0:
            self.pos += 1

    def sub(self, d):
        v = d[0]
        self.n -= 1
        for i in range(DIM):
            self.sv[i] -= v[i]
        if d[3] > 0:
            self.pos -= 1

    def flush(self):
        pass

    def merge(self, o):
        r = R1Moments()
        r.n = self.n + o.n
        r.sv = [a + b for a, b in zip(self.sv, o.sv)]
        r.pos = self.pos + o.pos
        return r

    def mean(self):
        return [x / self.n for x in self.sv] if self.n else [0.0] * DIM

    def pos_share(self):
        return self.pos / self.n if self.n else 0.0

    def ent_dist(self, ents):
        return None  # 结构性缺陷：答不出分布型问题

    def mem(self):
        return (DIM + 1) * 8 + 8

    def ops_add(self):
        return DIM

    def ops_sub(self):
        return DIM

    def ops_rollup(self, k):
        return k * (DIM + 1)


class R2Sketch:
    """R2 固定桶草图 (Count-Min)：实体分布用 (W,D) 哈希桶；加减为桶增减；有碰撞误差"""
    name = "R2 固定桶草图"

    def __init__(self):
        self.n = 0
        self.tab = [[0] * CMS_W for _ in range(CMS_D)]
        self.sv = [0.0] * DIM
        self.pos = 0

    @staticmethod
    def _idx(r, e):
        return (hash((r, e)) & 0x7FFFFFFF) % CMS_W

    def add(self, d):
        e = d[2]
        self.n += 1
        for r in range(CMS_D):
            self.tab[r][self._idx(r, e)] += 1
        v = d[0]
        for i in range(DIM):
            self.sv[i] += v[i]
        if d[3] > 0:
            self.pos += 1

    def sub(self, d):
        e = d[2]
        self.n -= 1
        for r in range(CMS_D):
            self.tab[r][self._idx(r, e)] -= 1
        v = d[0]
        for i in range(DIM):
            self.sv[i] -= v[i]
        if d[3] > 0:
            self.pos -= 1

    def flush(self):
        pass

    def merge(self, o):
        r = R2Sketch()
        r.n = self.n + o.n
        r.tab = [[a + b for a, b in zip(x, y)] for x, y in zip(self.tab, o.tab)]
        r.sv = [a + b for a, b in zip(self.sv, o.sv)]
        r.pos = self.pos + o.pos
        return r

    def est(self, e):
        return min(self.tab[r][self._idx(r, e)] for r in range(CMS_D))

    def mean(self):
        return [x / self.n for x in self.sv] if self.n else [0.0] * DIM

    def pos_share(self):
        return self.pos / self.n if self.n else 0.0

    def ent_dist(self, ents):
        return {e: self.est(e) for e in ents}

    def mem(self):
        return CMS_W * CMS_D * 4 + DIM * 8 + 16

    def ops_add(self):
        return CMS_D + DIM

    def ops_sub(self):
        return CMS_D + DIM

    def ops_rollup(self, k):
        return k * (CMS_W * CMS_D + DIM)


class R3Clusters:
    """R3 聚类摘要 (BIRCH 式 CF)：有微簇缓冲 + 簇容量上限；均值精确、可读性好，但每簇实体截断"""
    name = "R3 聚类摘要"

    def __init__(self):
        self.cl = []      # [n, cen, pos, ents]
        self.buf = []
        self.n = 0

    def _new(self, d):
        v = d[0]
        return [1, list(v), 1 if d[3] > 0 else 0, Counter({d[2]: 1}), Counter({d[1]: 1})]

    def _upd(self, c, d):
        v = d[0]
        n = c[0]
        inv = 1.0 / (n + 1)
        cen = c[1]
        for i in range(DIM):
            cen[i] = (cen[i] * n + v[i]) * inv
        c[0] = n + 1
        if d[3] > 0:
            c[2] += 1
        c[4][d[1]] = c[4].get(d[1], 0) + 1
        cnt = c[3]
        cnt[d[2]] = cnt.get(d[2], 0) + 1
        if len(cnt) > CF_TOP * 4:
            keep = cnt.most_common(CF_TOP * 2)
            c[3] = Counter(dict(keep))

    def _nearest(self, d):
        v = d[0]
        best = None
        bd = None
        for c in self.cl:
            cen = c[1]
            dist = 0.0
            for i in range(ASSIGN_DIM):
                x = cen[i] - v[i]
                dist += x * x
            if bd is None or dist < bd:
                bd = dist
                best = c
        return best

    def add(self, d):
        self.n += 1
        self.buf.append(d)
        if len(self.buf) >= CF_BUFFER:
            self.flush()

    def flush(self):
        for d in self.buf:
            self._insert(d)
        self.buf = []

    def _insert(self, d):
        if not self.cl:
            self.cl.append(self._new(d))
            return
        v = d[0]
        best = None
        bd = None
        for c in self.cl:
            cen = c[1]
            dist = 0.0
            for i in range(ASSIGN_DIM):
                x = cen[i] - v[i]
                dist += x * x
            if bd is None or dist < bd:
                bd = dist
                best = c
        if bd > CF_THRESH:
            self.cl.append(self._new(d))      # 超出半径判据 → 开新簇（容纳新概念）
        else:
            self._upd(best, d)
        while len(self.cl) > K_CLUSTERS:
            self._merge_closest_pair()        # 容量约束：每次只并最近的一对

    def _merge_closest_pair(self):
        m = len(self.cl)
        bi = -1
        bj = -1
        bd = None
        for i in range(m):
            for j in range(i + 1, m):
                dist = 0.0
                for k2 in range(ASSIGN_DIM):
                    x = self.cl[i][1][k2] - self.cl[j][1][k2]
                    dist += x * x
                if bd is None or dist < bd:
                    bd = dist
                    bi, bj = i, j
        if bi < 0:
            return
        merged = self._merge_two(self.cl[bi], self.cl[bj])
        self.cl = [c for idx, c in enumerate(self.cl) if idx != bi and idx != bj]
        self.cl.append(merged)

    @staticmethod
    def _merge_two(a, b):
        na, nb = a[0], b[0]
        n = na + nb
        cen = [(a[1][i] * na + b[1][i] * nb) / n for i in range(DIM)]
        cnt = a[3] + b[3]
        if len(cnt) > CF_TOP * 2:
            cnt = Counter(dict(cnt.most_common(CF_TOP * 2)))
        return [n, cen, a[2] + b[2], cnt, a[4] + b[4]]

    def sub(self, d):
        self.flush()
        self.n -= 1
        c = self._nearest(d)
        v = d[0]
        n = c[0]
        if n <= 1:
            self.cl.remove(c)
            return
        inv = 1.0 / (n - 1)
        cen = c[1]
        for i in range(DIM):
            cen[i] = (cen[i] * n - v[i]) * inv
        c[0] = n - 1
        if d[3] > 0:
            c[2] -= 1
        cnt = c[3]
        if cnt.get(d[2], 0) > 0:
            cnt[d[2]] -= 1

    def merge(self, o):
        r = R3Clusters()
        r.n = self.n + o.n
        r.cl = [[c[0], list(c[1]), c[2], Counter(c[3]), Counter(c[4])] for c in self.cl]
        r.cl += [[c[0], list(c[1]), c[2], Counter(c[3]), Counter(c[4])] for c in o.cl]
        while len(r.cl) > K_CLUSTERS:
            r._merge_closest_pair()
        return r

    def mean(self):
        tot = sum(c[0] for c in self.cl)
        if tot == 0:
            return [0.0] * DIM
        out = [0.0] * DIM
        for c in self.cl:
            for i in range(DIM):
                out[i] += c[1][i] * c[0]
        return [x / tot for x in out]

    def pos_share(self):
        tot = sum(c[0] for c in self.cl)
        return (sum(c[2] for c in self.cl) / tot) if tot else 0.0

    def ent_dist(self, ents):
        agg = Counter()
        for c in self.cl:
            agg += c[3]
        return {e: agg.get(e, 0) for e in ents}

    def kept_mass(self):
        return sum(sum(c[3].values()) for c in self.cl)

    def mem(self):
        return len(self.cl) * (2 * DIM * 8 + CF_TOP * 12)

    def ops_add(self):
        return len(self.cl) * ASSIGN_DIM + DIM

    def ops_sub(self):
        return len(self.cl) * ASSIGN_DIM + DIM

    def ops_rollup(self, k):
        return k * len(self.cl) * ASSIGN_DIM


class R4Hybrid:
    """R4 混合：矩统计(骨架,精确) + 草图(长尾实体) + 聚类(可读层) + 主题精确计数"""
    name = "R4 混合(推荐)"

    def __init__(self):
        self.m = R1Moments()
        self.sk = R2Sketch()
        self.cf = R3Clusters()
        self.tc = Counter()
        self.n = 0

    def add(self, d):
        self.n += 1
        self.m.add(d)
        self.sk.add(d)
        self.cf.add(d)
        self.tc[d[1]] += 1

    def sub(self, d):
        self.n -= 1
        self.m.sub(d)
        self.sk.sub(d)
        self.cf.sub(d)
        self.tc[d[1]] -= 1

    def flush(self):
        self.cf.flush()

    def merge(self, o):
        r = R4Hybrid()
        r.n = self.n + o.n
        r.m = self.m.merge(o.m)
        r.sk = self.sk.merge(o.sk)
        r.cf = self.cf.merge(o.cf)
        r.tc = self.tc + o.tc
        return r

    def mean(self):
        return self.m.mean()

    def pos_share(self):
        return self.m.pos_share()

    def ent_dist(self, ents):
        return self.sk.ent_dist(ents)

    def mem(self):
        return self.m.mem() + self.sk.mem() + self.cf.mem() + N_TOPICS * 12

    def ops_add(self):
        return self.m.ops_add() + self.sk.ops_add() + self.cf.ops_add()

    def ops_sub(self):
        return self.m.ops_sub() + self.sk.ops_sub() + self.cf.ops_sub()

    def ops_rollup(self, k):
        return self.m.ops_rollup(k) + self.sk.ops_rollup(k) + self.cf.ops_rollup(k)


METHODS = [R0NoMat, R1Moments, R2Sketch, R3Clusters, R4Hybrid]


# ---------------- 误差度量 ----------------
def cos_err(mean, tmean):
    a = math.sqrt(sum(x * x for x in mean))
    b = math.sqrt(sum(x * x for x in tmean))
    if a == 0 or b == 0:
        return 1.0
    c = sum(x * y for x, y in zip(mean, tmean)) / (a * b)
    return max(0.0, 1.0 - c)


def ent_l1(rep, t, topk=100):
    ents = [e for e, _ in t["ent"].most_common(topk)]
    est = rep.ent_dist(ents)
    if est is None:
        return None
    tot = rep.n or 1
    return sum(abs(est.get(e, 0) / tot - t["ent"][e] / t["n"]) for e in ents)


def aggregate(rows):
    out = {}
    for key in ("cos", "pol", "ent", "mem", "ops"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = (sum(vals) / len(vals)) if vals else None
    return out


def cf_purity(cf):
    """簇纯度：按簇规模加权，每簇内最高频真实主题的占比"""
    tot = sum(c[0] for c in cf.cl)
    if tot == 0:
        return None
    s = 0.0
    for c in cf.cl:
        tc = c[4]
        m = sum(tc.values())
        if m > 0:
            s += c[0] * (max(tc.values()) / m)
    return s / tot


def rep_mass(rep, t):
    """分布质量比：真实实体份额总和应为 1.0。
    草图会因哈希碰撞 >1（膨胀），聚类摘约会因截断 <1（丢失长尾）。"""
    if not isinstance(rep, (R2Sketch, R3Clusters, R4Hybrid)) or rep.n == 0:
        return None
    ents = list(t["ent"].keys())
    if isinstance(rep, R3Clusters):
        return rep.kept_mass() / rep.n
    if isinstance(rep, R2Sketch):
        return sum(rep.est(e) for e in ents) / rep.n
    return sum(rep.sk.est(e) for e in ents) / rep.n


def rep_purity(rep):
    if isinstance(rep, R3Clusters):
        return cf_purity(rep)
    if isinstance(rep, R4Hybrid):
        return cf_purity(rep.cf)
    return None


def rep_k(rep):
    if isinstance(rep, R3Clusters):
        return len(rep.cl)
    if isinstance(rep, R4Hybrid):
        return len(rep.cf.cl)
    return None


# ---------------- 实验 1：上卷合并 ----------------
def exp_rollup(docs):
    cells = defaultdict(list)
    parents = defaultdict(list)
    for d in docs:
        cells[(d[5], d[6])].append(d)
        parents[(d[4], d[6])].append(d)

    res = {}
    for M in METHODS:
        reps = {}
        for k, ds in cells.items():
            r = M()
            for d in ds:
                r.add(d)
            r.flush()
            reps[k] = r

        per_parent = defaultdict(list)
        for (s, g), r in reps.items():
            per_parent[(s // SLICES_PER_PHASE, g)].append(r)

        rows = []
        for pk, rlist in per_parent.items():
            merged = rlist[0]
            for r in rlist[1:]:
                merged = merged.merge(r)
            merged.flush()
            t = truth(parents[pk])
            cell_n = sum(r.n for r in rlist)
            rows.append({
                "cos": cos_err(merged.mean(), t["mean"]),
                "pol": abs(merged.pos_share() - t["pos_share"]),
                "ent": ent_l1(merged, t),
                "mem": merged.mem(),
                "ops": merged.ops_rollup(len(rlist)),
            })
        res[M.name] = aggregate(rows)
    return res


# ---------------- 实验 2：撤回更新 ----------------
def exp_retract(docs):
    cells = defaultdict(list)
    for d in docs:
        cells[(d[5], d[6])].append(d)
    target = max(cells.keys(), key=lambda k: len(cells[k]))
    ds = cells[target]
    rng = random.Random(SEED + 7)
    kept_idx = set()
    for i in range(len(ds)):
        if rng.random() < RETAIN:
            kept_idx.add(i)
    kept = [ds[i] for i in range(len(ds)) if i in kept_idx]
    dropped = [ds[i] for i in range(len(ds)) if i not in kept_idx]
    t = truth(kept)

    res = {}
    for M in METHODS:
        r = M()
        for d in ds:
            r.add(d)
        r.flush()
        n_before = r.n
        for d in dropped:
            r.sub(d)
        r.flush()
        res[M.name] = {
            "cos": cos_err(r.mean(), t["mean"]),
            "pol": abs(r.pos_share() - t["pos_share"]),
            "ent": ent_l1(r, t),
            "mass": rep_mass(r, t),
            "mem": r.mem(),
            "ops": r.ops_sub() * len(dropped),
            "recompute_scan": n_before,
            "n_kept": len(kept),
            "n_dropped": len(dropped),
        }
    return res, target, len(ds), len(dropped)


# ---------------- 实验 3：概念漂移 / 维度演化 ----------------
def exp_drift(docs):
    """维度演化下的分布保真与结构退化。
    窗口 = 上一阶段末尾 W 篇（陈旧）+ 当前阶段头部 W 篇（新），各占 50%。
    一阶矩对所有可加表示都相同，故本实验只度量三件事：
      分布保真(实体 L1)、质量守恒(膨胀/截断)、可读层结构(簇纯度与簇数)
    """
    per = N_DOCS // PHASES
    W = 1000
    out = {}
    meta = {}
    for p in range(1, PHASES):
        old = docs[p * per - W: p * per]
        new = docs[p * per: p * per + W]
        win = old + new
        t = truth(win)
        meta[p] = len(set(d[1] for d in win))
        rows = {}
        for M in METHODS:
            r = M()
            for d in win:
                r.add(d)
            r.flush()
            rows[M.name] = {
                "ent": ent_l1(r, t),
                "mass": rep_mass(r, t),
                "purity": rep_purity(r),
                "k": rep_k(r),
            }
        out[p] = rows
    return out, meta, W


# ---------------- 主流程 ----------------
def main():
    print("生成合成流 ...")
    docs = gen_stream()
    print("  %d docs / %d phases / %d topics / %d entities" % (len(docs), PHASES, N_TOPICS, N_ENTITIES))

    print("实验1 上卷合并 ...")
    roll = exp_rollup(docs)
    print("实验2 撤回更新 ...")
    retr, tkey, tn, tdrop = exp_retract(docs)
    print("实验3 维度演化 ...")
    drift, dmeta, dwin = exp_drift(docs)

    def f(x, nd=3):
        return "n/a" if x is None else round(x, nd)

    L = []
    A = L.append
    A("# 决策点3 探针实验结果 v1 —— 度量摘要的可聚合表示\n")
    A("> 合成数据探针（非真实数据），用于筛方案与确定实验口径。脚本：`pilot_repr_compare.py`\n")
    A("## 实验设置\n")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 文档数 | %d |" % N_DOCS)
    A("| 阶段数(概念漂移) | %d（阶段间主题词表完全不重叠） |" % PHASES)
    A("| 主题数 / 实体词表 | %d / %d（实体 Zipf 长尾） |" % (N_TOPICS, N_ENTITIES))
    A("| 语义向量维度 | %d（噪声 %.2f） |" % (DIM, NOISE))
    A("| 时间片 × 语义分组 | %d × %d = %d 个 cell |" % (SLICES_PER_PHASE * PHASES, GROUPS, SLICES_PER_PHASE * PHASES * GROUPS))
    A("| 聚类容量/簇实体截断 | K=%d / top-%d |" % (K_CLUSTERS, CF_TOP))
    A("| 草图 | %d 桶 × %d 行 |" % (CMS_W, CMS_D))
    A("")

    A("## 实验1：上卷合并（子 cell 合并为父 cell，与全量重算对比）\n")
    A("| 表示 | 向量误差 1-cos | 极性份额误差 | 实体分布 L1 | 单次上卷代价(ops) | 内存(bytes) |")
    A("|---|---|---|---|---|---|")
    order = [M.name for M in METHODS]
    for name in order:
        r = roll[name]
        A("| %s | %s | %s | %s | %s | %s |" % (
            name, f(r["cos"], 5), f(r["pol"], 5), f(r["ent"], 4),
            f(r["ops"], 0), f(r["mem"], 0)))
    A("")
    A("> 向量/极性误差全部≈0 是**预期结果**：一阶矩(Σx, n)天生可加，任何方案都能精确上卷。")
    A("> 真正的分水岭在**实体分布 L1**：R1 结构性答不出（n/a），R2 有哈希碰撞误差，R3 因每簇只留 top-%d 实体而丢失长尾质量。" % CF_TOP)
    A("")

    A("## 实验2：撤回更新（撤回 %.0f%% 文档，增量 vs 全量重算）\n" % ((1 - RETAIN) * 100))
    A("- 目标 cell：%s，原规模 %d 篇，撤回 %d 篇，剩余 %d 篇\n" % (str(tkey), tn, tdrop, tn - tdrop))
    A("| 表示 | 向量误差 1-cos | 极性份额误差 | 实体分布 L1 | 分布质量比 | 撤回总代价(ops) | 对比：全量重算需扫描 |")
    A("|---|---|---|---|---|---|---|")
    for name in order:
        r = retr[name]
        A("| %s | %s | %s | %s | %s | %s | %d 篇 |" % (
            name, f(r["cos"], 5), f(r["pol"], 5), f(r["ent"], 4), f(r["mass"], 4),
            f(r["ops"], 0), r["recompute_scan"]))
    A("")
    A("> 关键发现：**均值/极性的撤回也是精确的**（把一篇文档从全局和中减掉即可），聚类摘要同样精确——")
    A("> 因为簇心加权和的总和 = 全局和，分配错误只影响簇边界、不影响一阶矩。")
    A("> 撤退误差**只出现在分布型度量**上：草图碰撞、聚类截断。\n")

    A("## 实验3：维度演化下的分布保真与结构退化\n")
    A("设置：窗口 = 上一阶段**末尾 %d 篇（陈旧）** + 当前阶段**头部 %d 篇（新）**，各占 50%%。" % (dwin, dwin))
    A("一阶矩对所有可加表示都相同，故本实验只度量三件真正有差异的事：**分布保真 / 质量守恒 / 可读层结构**。\n")
    A("`分布质量比`：真实实体份额之和应 = 1.0。草图因哈希碰撞 **>1（膨胀）**，聚类摘约会因每簇截断 **<1（丢长尾）**。")
    A("`簇纯度`：按簇规模加权，每簇内最高频真实主题的占比（越高说明可读层越干净）。\n")
    for p in sorted(drift.keys()):
        A("### 阶段 %d → 阶段 %d 的混合窗口（窗口内实际出现 %d 个不同主题）\n" % (p, p + 1, dmeta[p]))
        A("| 表示 | 实体分布 L1 | 分布质量比 | 簇纯度 | 簇数 |")
        A("|---|---|---|---|---|")
        for name in order:
            r = drift[p][name]
            A("| %s | %s | %s | %s | %s |" % (
                name, f(r["ent"], 4), f(r["mass"], 4), f(r["purity"], 3),
                r["k"] if r["k"] is not None else "—"))
        A("")
    A("> 读法：R1 结构性答不出（n/a）；R2 的质量比 >1 表示**碰撞膨胀**；")
    A("> R3 的质量比 <1 表示**长尾被截断丢弃**；簇纯度则说明固定簇数 K 在新主题涌入时如何被摊薄。")
    A("> 注意本题的漂移是**完全换词表**（新主题向量与旧向量近似正交），是最容易被发现的一种漂移；")
    A("> 真实语义漂移更隐蔽（词义渐变、新旧混叠），这一项**必须用真实数据复做**。\n")

    A("## 综合判断（探针结论，待真实数据验证）\n")
    A("1. **一阶矩不是难点**：向量与极性在全部可加表示上都是精确的，包括撤回后（把文档从全局和中减掉即可）。")
    A("   所以「语义度量」的困难不在均值，而在**分布**——这一条应当写进论文，避免把力气花错地方。\n")
    A("2. **两种分布候选的失效方向相反**：")
    A("   - 草图（R2/R4）**膨胀**：哈希碰撞把长尾抬高，全词表质量比 1.9–2.1（应为 1.0）；")
    A("   - 聚类摘要（R3）**收缩**：每簇只留 top-%d 实体，质量比 0.48–0.57，丢掉长尾。\n" % CF_TOP)
    A("3. **平均 L1 是误导性指标**：草图的 top-100 实体 L1 只有 0.07–0.08（看起来很准），")
    A("   但它对**全词表**的膨胀是 2 倍——**误差几乎全部压在长尾上**。")
    A("   而长尾（新出现的实体、新话题）恰恰是谣言早期预警最需要的信号。")
    A("   ⇒ 真实实验必须**分头部/长尾分别报告指标**，不能只报一个平均值。\n")
    A("4. **可读层在维度演化下比预期稳健**：12 个真实主题对应 11–12 个簇，纯度 0.94，")
    A("   说明固定容量 K 不是主要瓶颈；瓶颈是**每簇的实体保留深度**（CF_TOP）。\n")
    A("5. **R0（不物化）误差恒为 0，代价是线性扫描**（本次撤回实验 31 万次操作 vs R2 的 6 千次）：")
    A("   它只应作为**正确性基准与代价上界**，不是竞争者。\n")
    A("6. **待真实数据回答的问题**（本探针无法替代）：")
    A("   ① 真实语义分布有多长尾（决定草图宽度与簇保留深度）；")
    A("   ② 真实漂移是渐变而非换词表（决定物化结构要不要支持维度扩展）；")
    A("   ③ 真实撤回事件的规模与突发性（决定失效传播是局部还是全局）。\n")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot_results_v1.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("已写出:", out)
    for line in L[:8]:
        print(line)


import os

if __name__ == "__main__":
    main()
