# -*- coding: utf-8 -*-
"""
E13 — 全量标注集 **v3**（S6-strict 加抽至 ≈270；协议 v1.4 §4.4b + 用户 2026-10-12 拍板）

与 v2 的唯一变量
----------------
**只追加 S6-strict 对 ≈173 条**（97 → ≈270），其余 582 行**原样保留**：
  · S1–S4  pair_id E13-0001…0388 不变（先导真值可对齐）；
  · S5     E13-0389…0485 不变；
  · S6     E13-0486…0582 不变（其中 strict 77 / polarity 12 / vague 8，先导已测产量）；
  · 新增   E13-0583… 续编，全部为 S6_same_slot_conflicting / strict。

卫生规则逐条沿用 v2：
  · 模板/机器人频次过滤（归一文本重复 >3 剔除）；
  · 配对防 hub 退化（新批内单节点 ≤2 次）；
  · 跨线程独立断言；取值互斥（无交集）；排除 v2 已用过的任何对；
  · 新批事件上限 40/事件（9 事件 × 40 = 360 ≥ 173，避免单事件霸占）。
  · RNG = Random(20261012)，与 v2 的 20261011 独立。

产出
----
  E13_标注集_v3.csv        7 层口径（582 + ≈173 ≈ 755 行，含元数据，**不得外发**）
  E13_盲版标注表_v3.csv    去泄露盲版（列同 v2）
  E13_抽样清单_v3.json     增量参数与层统计（可复现）
"""
import os
import csv
import json
import random
import pickle
import collections

import e13_slot_extract as S
import e13_build_annotation_set_v2 as V2   # 复用 clean/normtext/char3/coverage/ts_of/_mk 与常量

BASE = V2.BASE
DATA = V2.DATA

RNG = random.Random(20261012)
N_STRICT_EXTRA = 173          # 97 → 270
EXTRA_EVENT_CAP = 40
MAX_NODE_REUSE = V2.MAX_NODE_REUSE
S6_CLASSES = ['DEAD', 'INJURED', 'HOSTAGE', 'SUSPECT', 'AIRPLANE', 'BUILDING']


def main():
    print('[1/4] 装载 PHEME + 模板卫生（与 v2 逐字节同一实现）…')
    G = pickle.load(open(os.path.join(DATA, 'pheme_cascades.gpickle'), 'rb'))
    ver_raw = pickle.load(open(os.path.join(DATA, 'veracity_dict.pkl'), 'rb'))
    VER = {str(k): ({'true': 0, 'false': 1, 'unverified': 2}[v] if isinstance(v, str) else int(v))
           for k, v in ver_raw.items()}
    VNAME = {0: 'true', 1: 'false', 2: 'unverified', None: 'na'}
    NODES = {}
    for g in G:
        tid = str(g.graph.get('thread_id', ''))
        ev = g.graph.get('event', '?')
        v = VER.get(tid)
        for n, d in g.nodes(data=True):
            s = str(n)
            if not s.isdigit():
                continue
            t = V2.clean(d.get('text'))
            if len(t) < 10:
                continue
            NODES[s] = {'text': t, 'thread': tid, 'event': ev,
                        'ts': V2.ts_of(s), 'ver': VNAME.get(v, 'na')}
    nz_all = collections.Counter(V2.normtext(v['text']) for v in NODES.values())
    templates = set(nz for nz, c in nz_all.items() if c > V2.TEMPLATE_MAX)
    n_before = len(NODES)
    NODES = {k: v for k, v in NODES.items() if V2.normtext(v['text']) not in templates}
    print('    节点 %d（模板剔除 %d）' % (len(NODES), n_before - len(NODES)))

    # v2 全量已用对（防止任何重复）
    v2_rows = list(csv.DictReader(open(os.path.join(BASE, 'E13_标注集_v2.csv'), encoding='utf-8-sig')))
    used_pairs = set()
    for r in v2_rows:
        used_pairs.add(tuple(sorted([r['p_tweet_id'], r['q_tweet_id']])))
    print('[2/4] 继承 v2 全部 %d 行（pair_id 不变）' % len(v2_rows))
    rows = [dict(r) for r in v2_rows]
    next_id = 1 + max(int(r['pair_id'].split('-')[1]) for r in v2_rows)
    assert next_id == 583, 'v2 末行应为 E13-0582，实测 %d' % (next_id - 1)

    # ---------------- 仅追加 S6-strict ----------------
    print('[3/4] 生成 S6-strict 增量批（%d 条，RNG seed=%d）…' % (N_STRICT_EXTRA, 20261012))
    inv = collections.defaultdict(list)
    for k, v in NODES.items():
        sl, _ = S.slots_of(v['text'])
        for c, vals in sl.items():
            if c.startswith('RAW:') or c not in S6_CLASSES:
                continue
            inv[c].append((k, frozenset(vals)))

    byev = collections.defaultdict(list)
    for c, items in inv.items():
        for k, vals in items:
            byev[(c, NODES[k]['event'])].append((k, vals))

    cands = []
    for (c, ev), lst in byev.items():
        if len(lst) > V2.S6_GROUP_CAP:
            lst = RNG.sample(lst, V2.S6_GROUP_CAP)
        n = len(lst)
        for i in range(n):
            ki, vi = lst[i]
            for j in range(i + 1, n):
                kj, vj = lst[j]
                if ki == kj or NODES[ki]['thread'] == NODES[kj]['thread']:
                    continue
                if vi & vj:
                    continue
                pr = tuple(sorted([ki, kj]))
                if pr in used_pairs:
                    continue
                if not (all(x.isdigit() for x in vi) and all(x.isdigit() for x in vj)):
                    continue          # ★ 只收 strict（双侧均数值）
                p, q = (ki, kj) if NODES[ki]['ts'] <= NODES[kj]['ts'] else (kj, ki)
                vp, vq = (vi, vj) if p == ki else (vj, vi)
                cands.append((c, ev, p, q, sorted(vp), sorted(vq)))
    RNG.shuffle(cands)
    print('    strict 候选 %d 条（去重/去 v2 已用后）' % len(cands))

    ev_cnt, used_node = collections.Counter(), collections.Counter()
    added = []
    for c, ev, p, q, vp, vq in cands:
        if len(added) >= N_STRICT_EXTRA:
            break
        if ev_cnt[ev] >= EXTRA_EVENT_CAP:
            continue
        if used_node[p] >= MAX_NODE_REUSE or used_node[q] >= MAX_NODE_REUSE:
            continue
        d = V2._mk(next_id, 'S6_same_slot_conflicting', 'strict',
                   NODES[p], NODES[q], p, q,
                   note='slot=%s %s|%s (v3-extra)' % (c, ','.join(vp), ','.join(vq)))
        added.append(d)
        rows.append(d)
        used_pairs.add(tuple(sorted([p, q])))
        used_node[p] += 1
        used_node[q] += 1
        ev_cnt[ev] += 1
        next_id += 1
    print('    追加 %d 条（pair_id E13-%04d…E13-%04d）｜事件分布 %s'
          % (len(added), next_id - len(added), next_id - 1, dict(ev_cnt)))

    # ---------------- 落盘 ----------------
    with open(os.path.join(BASE, 'E13_标注集_v3.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=V2.FULL_COLS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print('[save] E13_标注集_v3.csv（%d 行）' % len(rows))

    blind = [{k: r.get(k, '') for k in V2.BLIND_COLS} for r in rows]
    with open(os.path.join(BASE, 'E13_盲版标注表_v3.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=V2.BLIND_COLS)
        w.writeheader()
        w.writerows(blind)
    print('[save] E13_盲版标注表_v3.csv（%d 行）' % len(blind))

    st = collections.Counter(r['stratum'] for r in rows)
    sub = collections.Counter(r['subtype'] for r in rows if r['stratum'].startswith('S6'))
    manifest = {
        'version': 'v3',
        'rng_seed': 20261012,
        'total': len(rows),
        'strata': dict(st),
        'S6_subtypes_all': dict(sub),
        'increment': {'only': 'S6-strict', 'n_added': len(added),
                      'pair_id_range': 'E13-%04d…E13-%04d' % (next_id - len(added), next_id - 1),
                      'extra_event_cap': EXTRA_EVENT_CAP,
                      'node_reuse_in_extra_batch': MAX_NODE_REUSE,
                      'excluded_all_v2_pairs': True},
        'inherited': {'v2_rows': len(v2_rows), 'pair_ids_unchanged': True},
        'candidate_pool_strict_after_dedupe': len(cands),
        'label_set': ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED'],
        'rationale': ('S6-strict 产量 11.1%（v1.6 口径）⇒ 97 条只能得 ≈11 条 CONFLICT 真值；'
                      '加抽至 ≈270 才能支撑逐类 F1（30÷0.111≈270，协议 §4.4b）。'
                      '用户 2026-10-12 拍板「加抽至 ≈270」+「跑全量探针标注」。'),
        'hygiene': V2_hygiene_snapshot(n_before, len(NODES)),
    }
    with open(os.path.join(BASE, 'E13_抽样清单_v3.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print('[save] E13_抽样清单_v3.json')
    print('\n层分布：%s\nS6 子型：%s' % (dict(st), dict(sub)))
    return manifest


def V2_hygiene_snapshot(n_before, n_after):
    return {'template_max_repeat': V2.TEMPLATE_MAX,
            'n_nodes_dropped_as_template': n_before - n_after,
            'n_nodes_after_hygiene': n_after,
            's6_classes_used': S6_CLASSES}


if __name__ == '__main__':
    main()
