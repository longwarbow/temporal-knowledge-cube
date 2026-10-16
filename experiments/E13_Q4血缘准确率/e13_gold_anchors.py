# -*- coding: utf-8 -*-
"""
E13 — $Q_4$（血缘回溯）准确率：**客观锚点**（无需人工标注）

上游定义：
  · RQ1 §5  $Q_4 = Lineage(c or p)$：返回支撑证据链（document → claim → cell）；
  · RQ1 §8.3 指标「血缘准确率」：$Q_4$ 与人工标注匹配；
  · `符号语义度量` §6：$Q_4$ = 支撑集 / why-provenance（Green et al. PODS'07）。

本脚本只做**客观可判定**的部分（人工部分见 `E13_标注协议.md`）：

  A1  血缘真值画像 —— PHEME 回复树 = document→claim 层的权威血缘；
      99.2% 的回复文本含 `@` 提及 ⇒ **结构通道与文本通道互为独立见证**。
  A2  **无特权结构时的血缘可恢复性** —— 只用 384 维文本嵌入的余弦，
      能否把一条回复的**父边**从同事件的干扰节点里挑出来？（直接对应 D24
      「不依赖单一平台 API」：若拿不到平台的回复树，血缘还能不能重建？）
  A3  时间近邻基线 —— 「最近的更早节点即父节点」的准确率（廉价启发式上界/下界）。
  A4  Wikidata 互反撤回对（承 E8 声明级样本）= 官方*共撤回/耦合*边锚点。

输出：e13_gold_anchors.json
"""
import os
import json
import math
import random
import pickle
import collections
import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = r'D:\WSPforWB\10.keyan\场景1-海量短文本\数据\PHEME'
E8_SAMPLE = r'D:\WSPforWB\10.keyan\场景1-海量短文本\实验\E8_Wikidata声明级撤回\E8_声明级样本.csv'

TW_SNOW = 1288834974657
EPOCH = datetime.datetime(1970, 1, 1)
RNG = random.Random(20261011)


def ts_of(nid):
    return (int(nid) >> 22) + TW_SNOW


def cos(a, b):
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.dot(a, a))
    nb = float(np.dot(b, b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b)) / math.sqrt(na * nb)


def auc(pos, neg):
    """秩和 AUC（Mann-Whitney），tie 取 0.5。"""
    if not pos or not neg:
        return float('nan')
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = {}
    i = 0
    # 处理 tie：同值取平均秩
    vals = [v for v, _ in allv]
    j = 0
    rank_of = [0.0] * len(vals)
    while j < len(vals):
        k = j
        while k + 1 < len(vals) and vals[k + 1] == vals[j]:
            k += 1
        avg = (j + k) / 2.0 + 1.0
        for t in range(j, k + 1):
            rank_of[t] = avg
        j = k + 1
    r_pos = sum(rank_of[t] for t in range(len(allv)) if allv[t][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def ancestor_set(g, node, parent):
    out = set()
    cur = parent.get(node)
    while cur is not None and cur not in out:
        out.add(cur)
        cur = parent.get(cur)
    return out


def descendant_set(children, node):
    out = set()
    stack = list(children.get(node, ()))
    while stack:
        x = stack.pop()
        if x in out:
            continue
        out.add(x)
        stack.extend(children.get(x, ()))
    return out


def main():
    print('[1/4] 装载 PHEME …')
    G = pickle.load(open(os.path.join(DATA, 'pheme_cascades.gpickle'), 'rb'))

    # ---------- A1 结构画像 ----------
    print('[2/4] A1 血缘真值画像（结构 + 文本双通道）…')
    n_threads = 0
    n_nodes = 0
    n_edges = 0
    depths = []
    sizes = []
    edges_with_at = 0
    mention_counts = []
    a2_pos = []          # 正例：cos(child, parent)
    a3_hit = 0
    a3_tot = 0
    neg_pool_used = 0
    a2_neg = []          # 负例：cos(child, 同事件非祖先节点)

    for gi, g in enumerate(G):
        nd = {str(n): d for n, d in g.nodes(data=True)}
        if not nd:
            continue
        n_threads += 1
        n_nodes += len(nd)
        sizes.append(len(nd))

        ev = g.graph.get('event', '?')
        parent = {}
        children = collections.defaultdict(list)
        for a, b in g.edges():
            a, b = str(a), str(b)
            parent[b] = a
            children[a].append(b)
            n_edges += 1
            tb = (nd.get(b, {}).get('text') or '')
            if '@' in tb:
                edges_with_at += 1

        # 深度
        roots = [k for k in nd if k not in parent]
        for r in roots:
            stack = [(r, 0)]
            seen = set()
            while stack:
                x, d = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                d2 = d
                for ch in children.get(x, ()):
                    stack.append((ch, d2 + 1))
                    depths.append(d2 + 1)

        # 嵌入缓存
        emb = {}
        for k, d in nd.items():
            e = d.get('text_embedding')
            if e is not None and len(e) > 0:
                emb[k] = e
            t = d.get('text')
            if t:
                mention_counts.append(t.count('@'))

        # 时间序（用索引表，避免 O(n^2)）
        order = sorted([k for k in nd if k.isdigit()], key=lambda k: int(k))
        pos_in_order = {k: i for i, k in enumerate(order)}

        # A2/A3 —— 用「有父节点且父子嵌入皆存在」的边
        for b, a in parent.items():
            if a not in emb or b not in emb:
                continue
            a2_pos.append(cos(emb[b], emb[a]))
            # A3：父节点是否是「最近的更早节点」
            i = pos_in_order.get(b)
            if i is not None and i > 0:
                a3_tot += 1
                if order[i - 1] == a:
                    a3_hit += 1

        # 负例：同线程非祖先/后代节点（同事件、同话题，是最难的干扰项）
        keys = list(emb.keys())
        if len(keys) >= 3:
            want = min(len(keys) * 2, 8)
            got = 0
            tries = 0
            while got < want and tries < want * 20:
                b = RNG.choice(keys)
                o = RNG.choice(keys)
                tries += 1
                if o == b:
                    continue
                if o in ancestor_set(g, b, parent):
                    continue
                if o in descendant_set(children, b):
                    continue
                a2_neg.append(cos(emb[b], emb[o]))
                neg_pool_used += 1
                got += 1

    depths.sort()
    sizes.sort()
    mention_counts.sort()

    def pctl(v, q):
        if not v:
            return None
        i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
        return v[i]

    A1 = {
        'n_threads': n_threads,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'edges_with_mention_pct': round(100.0 * edges_with_at / n_edges, 2) if n_edges else None,
        'thread_size_median': pctl(sizes, .5),
        'thread_size_p90': pctl(sizes, .9),
        'thread_size_max': sizes[-1] if sizes else None,
        'depth_median': pctl(depths, .5),
        'depth_p90': pctl(depths, .9),
        'depth_max': depths[-1] if depths else None,
        'mentions_per_post_median': pctl(mention_counts, .5),
        'mentions_per_post_p90': pctl(mention_counts, .9),
    }

    print('    线程 %d ｜ 节点 %d ｜ 回复边 %d ｜ 含@边 %.2f%%' %
          (n_threads, n_nodes, n_edges, A1['edges_with_mention_pct']))
    print('    线程规模 中位 %s / p90 %s ｜ 深度 中位 %s / p90 %s' %
          (A1['thread_size_median'], A1['thread_size_p90'],
           A1['depth_median'], A1['depth_p90']))

    # ---------- A2/A3 ----------
    print('[3/4] A2 内容通道恢复父边 / A3 时间近邻基线 …')
    a2 = {
        'n_pos': len(a2_pos),
        'n_neg': len(a2_neg),
        'cos_pos_mean': round(sum(a2_pos) / len(a2_pos), 4) if a2_pos else None,
        'cos_neg_mean': round(sum(a2_neg) / len(a2_neg), 4) if a2_neg else None,
        'auc_cos': round(auc(a2_pos, a2_neg), 4),
        'a3_nearest_earlier_accuracy': round(a3_hit / a3_tot, 4) if a3_tot else None,
        'a3_n': a3_tot,
    }
    print('    A2 正例 %d / 负例 %d ｜ cos(parent)=%.4f vs cos(非祖先)=%.4f ｜ **AUC=%.3f**' %
          (a2['n_pos'], a2['n_neg'], a2['cos_pos_mean'], a2['cos_neg_mean'], a2['auc_cos']))
    print('    A3 「最近更早节点=父节点」准确率 = %.3f (n=%d)' %
          (a2['a3_nearest_earlier_accuracy'], a2['a3_n']))

    # ---------- A4 Wikidata 互反撤回对 ----------
    print('[4/4] A4 Wikidata 互反（共撤回）边锚点 …')
    A4 = {'available': False}
    try:
        import csv
        rows = list(csv.DictReader(open(E8_SAMPLE, encoding='utf-8-sig')))
        p26 = [r for r in rows if r['pid'] == 'P26']
        E = {}
        for r in p26:
            a = r['item'].rsplit('/', 1)[-1]
            b = r['value'].rsplit('/', 1)[-1]
            E[(a, b)] = r
        rec = [(a, b) for (a, b) in E if (b, a) in E and a < b]

        def rs(x, y):
            return (E[(x, y)].get('reason') or '').rsplit('/', 1)[-1]

        # 与 E8 口径对齐：同因 = **双方原因均非空**且相同
        same_reason = sum(1 for a, b in rec if rs(a, b) and rs(a, b) == rs(b, a))
        both_empty = sum(1 for a, b in rec if not rs(a, b) and not rs(b, a))
        one_empty = len(rec) - same_reason - both_empty
        same_time = sum(1 for a, b in rec
                        if (E[(a, b)].get('endtime') or '') and
                        (E[(a, b)].get('endtime') == E[(b, a)].get('endtime')))
        A4 = {
            'available': True,
            'n_spouse_statements_in_sample': len(p26),
            'n_distinct_edges': len(E),
            'n_reciprocal_pairs': len(rec),
            'same_reason_pairs_E8_criterion': same_reason,   # 双双非空且相同（= E8 的 46）
            'both_reason_empty_pairs': both_empty,           # 双空（不计入同因）
            'one_reason_missing_pairs': one_empty,
            'same_endtime_pairs': same_time,
            'criterion_note': 'E8 口径：同因要求双方 reason 均非空且相同；本表额外披露“双空”计数以消除歧义',
            'examples': [{'a': a, 'b': b,
                          'reason_a': E[(a, b)].get('reason'),
                          'reason_b': E[(b, a)].get('reason'),
                          'end_a': E[(a, b)].get('endtime'),
                          'end_b': E[(b, a)].get('endtime')} for a, b in rec[:5]],
        }
        print('    P26 声明 %d 条 ｜ 互反对 **%d** 对 ｜ 同因(E8口径) %d ｜ 双空 %d ｜ 同刻 %d' %
              (len(p26), len(rec), same_reason, both_empty, same_time))
    except Exception as ex:
        print('    (跳过 A4：%s)' % ex)

    res = {
        'A1_lineage_ground_truth': A1,
        'A2_content_recovery': a2,
        'A3_temporal_baseline': {'nearest_earlier_accuracy': a2['a3_nearest_earlier_accuracy'],
                                 'n': a2['a3_n']},
        'A4_wikidata_coupled_edges': A4,
        'note': ('A1/A2/A3 = document→claim 层血缘（客观可判定）；'
                 'claim 层的语义边（DERIVE/SUPPORT/CONFLICT）需人工标注，见 E13_标注协议.md'),
    }
    with open(os.path.join(BASE, 'e13_gold_anchors.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('\n[save] e13_gold_anchors.json')
    return res


if __name__ == '__main__':
    main()
