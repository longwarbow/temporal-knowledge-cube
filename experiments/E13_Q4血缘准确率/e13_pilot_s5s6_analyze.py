# -*- coding: utf-8 -*-
"""
E13 — $Q_4$：**S5 / S6 先导分析**（产量 yield + 通道一致性 + 系统评分）

判据为**冻结版 v1.3**，且本样本**未参与任何判据迭代** ⇒ 这是一次
**样本外**的产量测量（不同于先导 50 条的 in-sample 校准）。

产出：E13_先导S5S6_分析.json / E13_先导S5S6_最终真值.csv
"""
import os
import csv
import json
import collections

BASE = os.path.dirname(os.path.abspath(__file__))
LAB = ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED']


def kappa(y1, y2, labels):
    n = len(y1)
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    pe = 0.0
    for L in labels:
        pe += (sum(1 for a in y1 if a == L) / n) * (sum(1 for b in y2 if b == L) / n)
    k = (po - pe) / (1 - pe) if pe < 1 else float('nan')
    return po, pe, k


def gwet_ac1(y1, y2, labels):
    """Gwet AC1（抗偏斜）：在标签高度不平衡时比 κ 稳健。"""
    n = len(y1)
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    q = len(labels)
    pj = [(sum(1 for a in y1 if a == L) + sum(1 for b in y2 if b == L)) / (2 * n) for L in labels]
    pe = sum(p * (1 - p) for p in pj) / (q - 1) if q > 1 else 0.0
    return (po - pe) / (1 - pe) if pe < 1 else float('nan')


def boot_ci(y1, y2, labels, B=2000, seed=20261011):
    import random
    rng = random.Random(seed)
    n = len(y1)
    ks = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        a = [y1[i] for i in idx]
        b = [y2[i] for i in idx]
        _, _, k = kappa(a, b, labels)
        if k == k:
            ks.append(k)
    ks.sort()
    if not ks:
        return None, None
    return round(ks[int(0.025 * len(ks))], 4), round(ks[int(0.975 * len(ks)) - 1], 4)


def wilson(k, n, z=1.96):
    """Wilson 区间（小样本比例的首选，比正态近似稳）。"""
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


def metrics(y_true, y_pred):
    res = {}
    for L in LAB:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == L and p == L)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != L and p == L)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == L and p != L)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        res[L] = {'n_true': sum(1 for t in y_true if t == L), 'n_pred': sum(1 for p in y_pred if p == L),
                  'P': round(pr, 4), 'R': round(rc, 4), 'F1': round(f1, 4)}
    res['macro_F1'] = round(sum(res[L]['F1'] for L in LAB) / 4, 4)
    cm = {t: collections.Counter() for t in LAB}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    res['confusion'] = {k: dict(v) for k, v in cm.items()}
    return res


def main():
    blind = {r['pair_id']: r for r in csv.DictReader(
        open(os.path.join(BASE, 'E13_先导S5S6_盲版.csv'), encoding='utf-8-sig'))}
    idx = json.load(open(os.path.join(BASE, 'E13_先导S5S6_索引.json'), encoding='utf-8'))['map']
    G = {r['pair_id']: r for r in csv.DictReader(
        open(os.path.join(BASE, 'E13_先导S5S6_通道G.csv'), encoding='utf-8-sig'))}
    H = {r['pair_id']: r for r in csv.DictReader(
        open(os.path.join(BASE, 'E13_先导S5S6_通道H.csv'), encoding='utf-8-sig'))}
    pids = [p for p in blind if p in G and p in H]
    print('可比较 %d 条' % len(pids))

    yg = [G[p]['label_relation'] for p in pids]
    yh = [H[p]['label_relation'] for p in pids]
    po, pe, k = kappa(yg, yh, LAB)
    ac1 = gwet_ac1(yg, yh, LAB)
    lo, hi = boot_ci(yg, yh, LAB)
    dis = [p for p in pids if G[p]['label_relation'] != H[p]['label_relation']]
    print('一致性 p_o=%.3f  κ=%.3f (95%%CI %.3f–%.3f)  AC1=%.3f  分歧 %d/%d'
          % (po, k, lo, hi, ac1, len(dis), len(pids)))

    # 共识（一致项直接取；分歧项取保守：若任一方 UNRELATED 则取另一方？——这里**只报告**，
    # 共识仅用一致项；分歧项留空并在报告中单列）
    consensus = {p: (yg[i] if yg[i] == yh[i] else '') for i, p in enumerate(pids)}

    # ---------------- 产量 yield ----------------
    def yield_of(keys, label):
        sub = [p for p in pids if consensus[p] and keys(p)]
        hit = sum(1 for p in sub if consensus[p] == label)
        lo_, hi_ = wilson(hit, len(sub))
        return {'n_labelled_agree': len(sub), 'hit': hit,
                'yield': round(hit / len(sub), 4) if sub else None,
                'yield_ci95_wilson': [lo_, hi_],
                'dist': dict(collections.Counter(consensus[p] for p in sub))}

    yld = {
        'S5_all': yield_of(lambda p: idx[p]['stratum'].startswith('S5'), 'DERIVE'),
        'S6_all': yield_of(lambda p: idx[p]['stratum'].startswith('S6'), 'CONFLICT'),
        'S6_strict': yield_of(lambda p: idx[p]['subtype'] == 'strict', 'CONFLICT'),
        'S6_polarity': yield_of(lambda p: idx[p]['subtype'] == 'polarity', 'CONFLICT'),
        'S6_vague_NEGCTRL': yield_of(lambda p: idx[p]['subtype'] == 'vague', 'CONFLICT'),
    }
    print('\n[产量]')
    for kk, vv in yld.items():
        print('  %-18s n=%-3s 命中=%-3s yield=%-6s CI95=%-16s 分布=%s'
              % (kk, vv['n_labelled_agree'], vv['hit'], vv['yield'],
                 str(vv['yield_ci95_wilson']), vv['dist']))
    baseline = {'note': 'S1–S4 随机式分层（先导 50 条）上 CONFLICT 密度仅 1/50 = 2%，作为密度基线',
                'conflict_density': 0.02}
    print('  [基线] S1–S4 先导 50 条 CONFLICT 密度 = 2%')

    # 分歧明细（供报告引用）
    print('\n[分歧 %d 条]' % len(dis))
    for p in dis:
        print('  %s  G=%-9s H=%-9s  [%s/%s]' % (p, G[p]['label_relation'], H[p]['label_relation'],
                                                idx[p]['stratum'][:2], idx[p]['subtype']))
        print('     P:', (blind[p]['p_text'] or '')[:120])
        print('     Q:', (blind[p]['q_text'] or '')[:120])

    # ---------------- ★ 敏感性：S5 里"仅前导 @提及不同"的比例 ----------------
    # 冻结判据 §2.1 明确"覆盖率在**去掉 @/URL 之后**计算" ⇒ DERIVE 是 v1.3 下的正解；
    # 但两名通道都点名了这条缝。这里给出"若把这类改判 SUPPORT"的下界。
    import re as _re
    def _nonmention(t):
        t = _re.sub(r'http\S+|www\.\S+', '', t or '')
        t = _re.sub(r'@[A-Za-z0-9_]+', '', t)
        return _re.sub(r'[^a-z0-9]', '', t.lower())

    s5p = [p for p in pids if consensus[p] and idx[p]['stratum'].startswith('S5')]
    mention_only = [p for p in s5p if _nonmention(blind[p]['p_text']) == _nonmention(blind[p]['q_text'])]
    hit_now = sum(1 for p in s5p if consensus[p] == 'DERIVE')
    alt = [('SUPPORT' if p in mention_only else consensus[p]) for p in s5p]
    hit_alt = sum(1 for v in alt if v == 'DERIVE')
    sens = {
        'S5_n': len(s5p),
        'n_mention_only_variants': len(mention_only),
        'mention_only_ids': mention_only,
        'yield_if_mention_only_is_DERIVE': round(hit_now / len(s5p), 4) if s5p else None,
        'yield_if_mention_only_is_SUPPORT': round(hit_alt / len(s5p), 4) if s5p else None,
        'note': ('冻结判据 §2.1 规定覆盖率在去掉 @/URL 后计算 ⇒ 正解为 DERIVE；'
                 '此处给出"若改判 SUPPORT"的下界。两名通道**各自独立**点名了这条缝。'),
    }
    print('\n[S5 敏感性] "仅前导 @ 提及不同" %d/%d 条 ⇒ S5 yield 区间 [%.4f, %.4f]'
          % (len(mention_only), len(s5p),
             sens['yield_if_mention_only_is_SUPPORT'] or 0,
             sens['yield_if_mention_only_is_DERIVE'] or 0))

    # ---------------- 系统 v2 评分（对共识一致项） ----------------
    sysp = {r['pair_id']: r['system_pred'] for r in csv.DictReader(
        open(os.path.join(BASE, 'E13_系统预测_v2.csv'), encoding='utf-8-sig'))}
    scored = [p for p in pids if consensus[p]]
    yt = [consensus[p] for p in scored]
    yp = [sysp.get(p, 'UNRELATED') for p in scored]
    m = metrics(yt, yp)
    m['n'] = len(scored)
    print('\n[系统 v2 对共识 n=%d] macro-F1=%.4f' % (len(scored), m['macro_F1']))
    for L in LAB:
        print('  %-10s n_true=%-3d n_pred=%-3d P=%.3f R=%.3f F1=%.3f'
              % (L, m[L]['n_true'], m[L]['n_pred'], m[L]['P'], m[L]['R'], m[L]['F1']))
    if m['CONFLICT']['n_true'] < 30:
        print('  ⚠️ CONFLICT 真值 %d 条 < 30 ⇒ 按协议 §7 只报"存在性"，不报其 F1'
              % m['CONFLICT']['n_true'])

    # 逐条明细
    with open(os.path.join(BASE, 'E13_先导S5S6_最终真值.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pair_id', 'stratum', 'subtype', 'system_pred', 'G', 'H',
                    'consensus_label', 'agree', 'p_note_G', 'p_note_H'])
        for p in pids:
            w.writerow([p, idx[p]['stratum'], idx[p]['subtype'], sysp.get(p, ''),
                        G[p]['label_relation'], H[p]['label_relation'], consensus[p],
                        int(G[p]['label_relation'] == H[p]['label_relation']),
                        (G[p].get('label_notes') or '')[:160],
                        (H[p].get('label_notes') or '')[:160]])
    print('[save] E13_先导S5S6_最终真值.csv')

    res = {
        'n': len(pids),
        'agreement': {'p_o': round(po, 4), 'p_e': round(pe, 4), 'kappa': round(k, 4),
                      'kappa_ci95': [lo, hi], 'gwet_ac1': round(ac1, 4),
                      'n_disagreement': len(dis), 'disagreement_ids': dis},
        'channel_dist': {'G': dict(collections.Counter(yg)),
                         'H': dict(collections.Counter(yh))},
        'consensus_dist': dict(collections.Counter(v for v in consensus.values() if v)),
        'yield': yld,
        'baseline_conflict_density': baseline,
        'S5_sensitivity_mention_only': sens,
        'system_v2_vs_consensus': m,
        'caveat': ('通道同源（同一模型）⇒ 误差相关 ⇒ κ 偏乐观；合法用法是"否决性检验"。'
                   '本样本未参与判据迭代（冻结版 v1.3 事前固定）⇒ 产量测量是**样本外**的。'),
    }
    with open(os.path.join(BASE, 'E13_先导S5S6_分析.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('[save] E13_先导S5S6_分析.json')
    return res


if __name__ == '__main__':
    main()
