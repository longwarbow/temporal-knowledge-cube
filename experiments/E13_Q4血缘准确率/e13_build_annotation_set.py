# -*- coding: utf-8 -*-
"""
E13 — $Q_4$ 血缘准确率：**人工标注集构造 + 分层抽样**（Tier B）

产出：
  · E13_标注集.csv       —— 待标注样本（分层抽样，含判定所需全部上下文；标签列留空）
  · E13_抽样清单.json    —— 抽样参数与分层统计（可复现）

设计要点
--------
**标注单元**：一对断言 $(p, q)$（p 早于 q），标注员判断 **q 相对 p 的关系**：
  DERIVE       q 是 p 的转述/引用/转发（同一断言，q 派生自 p）
  SUPPORT      q 独立地断言与 p 相同的命题（无引用关系）
  CONFLICT     q 断言与 p 不相容的命题（极性相反 / 互斥细节）
  UNRELATED    q 在说别的事
标签集与 `符号语义度量` §3（support / conflict / derivation）一一对应，
因此标注结果可直接作为 §5.4 声明图边的真值，并成为 RQ3（`StreamRumor-Bench`）
**血缘标注子集**的种子规范。

**分层**（避免只采"容易的"样本）：
  S1 same-thread reply      —— 结构上的派生关系（应多为 DERIVE / CONFLICT）
  S2 cross-thread high-sim  —— 同事件、跨线程、嵌入高相似（应多为 SUPPORT）
  S3 cross-thread opp-ver   —— 同事件、跨线程、veracity 相反（CONFLICT 的高危层）
  S4 cross-thread random    —— 同事件、跨线程、随机（对照层，应多为 UNRELATED）

**样本量**：每层 n = 97，取 `E12`（D8-rev1）反推出的「比例·精」档
（$n_{\\min}\\approx0.96/\\varepsilon^2$，$\\varepsilon=0.10$）⇒ 合计 388。

**系统预测列**：留空。预测接口见 `E13_标注协议.md` §5；一旦系统产出，
`e13_score.py` 直接对 `system_pred` 与 `label_relation` 打分。
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
import os
import csv
import json
import random
import pickle
import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = _KB('数据', 'PHEME')

RNG = random.Random(20261011)
N_PER_STRATUM = 97        # E12: ε=0.10 ⇒ n_min ≈ 97（比例·精）
CAP_PER_EVENT = 30        # 单事件在单层内的上限，保证事件覆盖均衡（避免混淆）
MAXLEN = 320              # 文本截断（保留完整语义）


def clean(t):
    if not t:
        return ''
    t = ' '.join(t.split())
    return t[:MAXLEN]


def main():
    print('[1/3] 装载 PHEME（文本 + 嵌入 + veracity）…')
    G = pickle.load(open(os.path.join(DATA, 'pheme_cascades.gpickle'), 'rb'))
    ver_raw = pickle.load(open(os.path.join(DATA, 'veracity_dict.pkl'), 'rb'))
    VER = {str(k): int(v if not isinstance(v, str) else {'true': 0, 'false': 1, 'unverified': 2}[v])
           for k, v in ver_raw.items()}
    VNAME = {0: 'true', 1: 'false', 2: 'unverified'}

    # 结构化：线程 → (节点表, 父映射, 事件, veracity)
    threads = {}
    for g in G:
        tid = str(g.graph.get('thread_id', ''))
        nd = {}
        for n, d in g.nodes(data=True):
            s = str(n)
            if not s.isdigit():
                continue
            nd[s] = {'text': clean(d.get('text')), 'emb': d.get('text_embedding')}
        if not nd:
            continue
        parent = {str(b): str(a) for a, b in g.edges()}
        threads[tid] = {'nodes': nd, 'parent': parent,
                        'event': g.graph.get('event', '?'),
                        'veracity': VER.get(tid)}
    print('    线程 %d ｜ 有 veracity %d' %
          (len(threads), sum(1 for t in threads.values() if t['veracity'] is not None)))

    rows = []

    def emit(stratum, tid_p, tid_q, p_id, q_id, p, q, note=''):
        rows.append({
            'pair_id': 'E13-%04d' % (len(rows) + 1),
            'stratum': stratum,
            'event': threads[tid_p]['event'],
            'p_thread': tid_p, 'q_thread': tid_q,
            'p_tweet_id': p_id, 'q_tweet_id': q_id,
            'p_veracity': VNAME.get(threads[tid_p]['veracity'], 'na'),
            'q_veracity': VNAME.get(threads[tid_q]['veracity'], 'na'),
            'p_text': p, 'q_text': q,
            'system_prior': note,
            'system_pred': '',            # 由系统填写（DERIVE/SUPPORT/CONFLICT/UNRELATED）
            'label_relation': '',         # 人工标注（主）
            'label_relation_2': '',       # 第二标注员（20% 重叠）
            'label_confidence': '',       # 1=低 2=中 3=高
            'label_notes': '',
        })

    # ---------- S1: same-thread reply（父→子） ----------
    print('[2/3] 抽样 …')
    s1 = 0
    need = N_PER_STRATUM * 8     # 候选池
    cand = []
    for tid, T in threads.items():
        for c, p in T['parent'].items():
            pt = T['nodes'].get(p, {}).get('text')
            ct = T['nodes'].get(c, {}).get('text')
            if pt and ct and len(pt) >= 15 and len(ct) >= 15:
                cand.append((tid, p, c, pt, ct))
    RNG.shuffle(cand)
    ev_cnt = {}
    for tid, p, c, pt, ct in cand:
        if s1 >= N_PER_STRATUM:
            break
        ev = threads[tid]['event']
        if ev_cnt.get(ev, 0) >= CAP_PER_EVENT:
            continue
        emit('S1_same_thread_reply', tid, tid, p, c, pt, ct,
             note='structural: q replies to p')
        ev_cnt[ev] = ev_cnt.get(ev, 0) + 1
        s1 += 1

    # ---------- 跨线程池：按事件分组 ----------
    by_event = {}
    for tid, T in threads.items():
        if T['veracity'] is None:
            continue
        # 线程代表文本 = 根帖
        roots = [k for k in T['nodes'] if k not in T['parent']]
        if not roots:
            continue
        r = min(roots, key=lambda k: int(k))
        rt = T['nodes'][r]['text']
        if rt and len(rt) >= 15:
            by_event.setdefault(T['event'], []).append((tid, r, rt, T['veracity']))

    n3 = n2 = n4 = 0
    ev_cnt3 = {}
    for ev, lst in by_event.items():
        RNG.shuffle(lst)
        # S3: 相反 veracity（true vs false）
        trues = [x for x in lst if x[3] == 0]
        falses = [x for x in lst if x[3] == 1]
        RNG.shuffle(trues)
        RNG.shuffle(falses)
        for i in range(min(len(trues), len(falses))):
            if n3 >= N_PER_STRATUM or ev_cnt3.get(ev, 0) >= CAP_PER_EVENT:
                break
            tp = trues[i]
            fq = falses[i]
            emit('S3_cross_thread_opposite_veracity', tp[0], fq[0], tp[1], fq[1],
                 tp[2], fq[2], note='veracity: p=true, q=false')
            ev_cnt3[ev] = ev_cnt3.get(ev, 0) + 1
            n3 += 1

    # S2: 跨线程高相似；S4: 跨线程随机
    sim_cand = []
    rnd_cand = []
    for ev, lst in by_event.items():
        if len(lst) < 4:
            continue
        for _ in range(2000):
            a, b = RNG.sample(lst, 2)
            ea, eb = threads[a[0]]['nodes'][a[1]]['emb'], threads[b[0]]['nodes'][b[1]]['emb']
            sim_cand.append((a, b, ea, eb))
            rnd_cand.append((a, b))
    # 计算相似度
    try:
        import numpy as np

        def cosine(ea, eb):
            a = np.asarray(ea, float)
            b = np.asarray(eb, float)
            d = (np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / d) if d > 0 else 0.0

        scored = [(cosine(ea, eb), a, b) for a, b, ea, eb in sim_cand]
        scored.sort(key=lambda x: -x[0])
        used = set()
        ev_cnt2 = {}
        for s, a, b in scored:
            if n2 >= N_PER_STRATUM:
                break
            key = tuple(sorted([a[0], b[0]]))
            if key in used or s < 0.5:
                continue
            ev = threads[a[0]]['event']
            if ev_cnt2.get(ev, 0) >= CAP_PER_EVENT:
                continue
            used.add(key)
            ev_cnt2[ev] = ev_cnt2.get(ev, 0) + 1
            emit('S2_cross_thread_high_sim', a[0], b[0], a[1], b[1], a[2], b[2],
                 note='cos(thread roots)=%.3f' % s)
            n2 += 1
    except Exception as ex:
        print('    (S2 跳过：%s)' % ex)

    ev_cnt4 = {}
    for a, b in rnd_cand:
        if n4 >= N_PER_STRATUM:
            break
        ev = threads[a[0]]['event']
        if ev_cnt4.get(ev, 0) >= CAP_PER_EVENT:
            continue
        emit('S4_cross_thread_random', a[0], b[0], a[1], b[1], a[2], b[2],
             note='random same-event pair')
        ev_cnt4[ev] = ev_cnt4.get(ev, 0) + 1
        n4 += 1

    print('    S1=%d  S2=%d  S3=%d  S4=%d  合计 %d' % (s1, n2, n3, n4, len(rows)))

    # ---------- 落盘 ----------
    print('[3/3] 写盘 …')
    cols = ['pair_id', 'stratum', 'event', 'p_thread', 'q_thread',
            'p_tweet_id', 'q_tweet_id', 'p_veracity', 'q_veracity',
            'p_text', 'q_text', 'system_prior', 'system_pred',
            'label_relation', 'label_relation_2', 'label_confidence', 'label_notes']
    with open(os.path.join(BASE, 'E13_标注集.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    manifest = {
        'n_per_stratum': N_PER_STRATUM,
        'cap_per_event': CAP_PER_EVENT,
        'total': len(rows),
        'strata': {'S1_same_thread_reply': s1, 'S2_cross_thread_high_sim': n2,
                   'S3_cross_thread_opposite_veracity': n3, 'S4_cross_thread_random': n4},
        'event_distribution': dict(sorted(
            __import__('collections').Counter(r['event'] for r in rows).items(),
            key=lambda kv: -kv[1])),
        'label_set': ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED'],
        'sample_size_rationale': 'n=97 取自 E12/D8-rev1「比例·精」档 '
                                 '(n_min≈0.96/eps^2, eps=0.10)，按层独立估计比例',
        'overlap_for_IAA': '前 20%（≈78 对）由第二标注员独立标注，用于 Cohen kappa',
        'text_truncation_chars': MAXLEN,
        'note': 'system_pred 留空，由系统填写；label_* 由人工填写。见 E13_标注协议.md',
    }
    with open(os.path.join(BASE, 'E13_抽样清单.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print('[save] E13_标注集.csv / E13_抽样清单.json')
    return manifest


if __name__ == '__main__':
    main()
