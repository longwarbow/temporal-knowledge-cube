# -*- coding: utf-8 -*-
"""E13 v6 = v5 + R1 评论对守卫 + R2 语义复述守卫（轮 9 探索）

R1（rule 53）：lab==SUPPORT 且**两侧皆为非命题评论**（无谓词族锚 + 有意见/疑问标记）
               ⇒ UNRELATED。
R2（rule 39）：lab==UNRELATED 且是**兜底**（rule 10）且 q/p 语义余弦 ≥ τ
               且非评论对、无数值冲突 ⇒ SUPPORT。

本脚本扫描 τ 与守卫组合，报告全量 755 的 macro-F1（对 v1.7r 真值）。
"""
import os
import re
import csv
import math
import pickle
import sys

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import e13_claim_graph_v5 as V5
import e13_slot_extract as SLOT

CATS = ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED']

_EMB = pickle.load(open(os.path.join(BASE, 'E13_v6_emb_cache.pkl'), 'rb'))

_OPINION = re.compile(
    r"\b(wtf|ridiculous|misleading|seriously|kidding|ur\b|u r\b|you're\b|"
    r"not buying|nonsense|shameful|shame|disgrace|whatever|"
    r"no\?|really\?|right\?|isn't it|at all)\b", re.I)


def cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return num / (na * nb) if na and nb else 0.0


def _has_family(text):
    return any(w in SLOT.TRIG2CLASS for w in re.findall(r'[a-z]+', (text or '').lower()))


def is_comment_pair(p, q):
    """两侧皆为非命题评论：两侧都无谓词族锚，且至少一侧有意见/疑问标记。"""
    if _has_family(p) or _has_family(q):
        return 0
    if _OPINION.search(p or '') or _OPINION.search(q or '') or '?' in (p or '') or '?' in (q or ''):
        return 1
    return 0


def score(pred, truth):
    out, f1s = {}, []
    for c in CATS:
        tp = sum(1 for p in truth if truth[p] == c and pred.get(p) == c)
        fp = sum(1 for p in truth if truth[p] != c and pred.get(p) == c)
        fn = sum(1 for p in truth if truth[p] == c and pred.get(p) != c)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        out[c] = {'n_true': tp + fn, 'P': round(P, 4), 'R': round(R, 4), 'F1': round(F, 4)}
        f1s.append(F)
    return round(sum(f1s) / len(f1s), 4), out


def main():
    rows = list(csv.DictReader(open(os.path.join(BASE, 'E13_标注集_v3.csv'), encoding='utf-8-sig')))
    truth = {r['pair_id']: r['truth'] for r in csv.DictReader(
        open(os.path.join(BASE, 'E13_全量真值_v5.csv'), encoding='utf-8-sig'))}

    # 预计算
    recs = []
    for r in rows:
        pid = r['pair_id']
        p, q = r['p_text'] or '', r['q_text'] or ''
        f = V5.features(p, q)
        struct = (r.get('system_prior') or '').startswith('structural')
        base_lab, base_rule = V5.predict(f, struct)
        ep, eq = _EMB.get(r['p_tweet_id']), _EMB.get(r['q_tweet_id'])
        c = cos(ep, eq) if ep and eq else 0.0
        cp = is_comment_pair(p, q)
        recs.append({'pid': pid, 'f': f, 'base': (base_lab, base_rule), 'cos': c, 'cp': cp})

    # v5 基线
    v5_pred = {x['pid']: x['base'][0] for x in recs}
    m5, d5 = score(v5_pred, truth)
    print('v5 基线 macro-F1 = %.4f' % m5)
    print('  R1候选(comment_pair 且 v5=SUPPORT):',
          [x['pid'] for x in recs if x['cp'] and x['base'][0] == 'SUPPORT'])
    print('  评论对总数:', sum(x['cp'] for x in recs))
    print()

    print('=== τ 扫描（加 R1，再逐 τ 开 R2）===')
    print('%-6s %-10s %-10s %-10s %-10s %-10s' % ('τ', 'macroF1', 'DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED'))
    best = None
    for ti in range(55, 86, 5):
        tau = ti / 100
        pred = {}
        n_r1 = n_r2 = 0
        for x in recs:
            lab, rule = x['base']
            if lab == 'SUPPORT' and x['cp']:
                lab, rule = 'UNRELATED', 53
                n_r1 += 1
            if lab == 'UNRELATED' and rule == 10 and x['cos'] >= tau and not x['cp'] \
                    and x['f']['v5_num_verdict'] != 'CONFLICT':
                lab, rule = 'SUPPORT', 39
                n_r2 += 1
            pred[x['pid']] = lab
        m, d = score(pred, truth)
        print('%-6.2f %-10.4f %-10.3f %-10.3f %-10.3f %-10.3f  (R1=%d R2=%d)' % (
            tau, m, d['DERIVE']['F1'], d['SUPPORT']['F1'], d['CONFLICT']['F1'], d['UNRELATED']['F1'], n_r1, n_r2))
        if best is None or m > best[1]:
            best = (tau, m, d, n_r1, n_r2)

    tau = best[0]
    print()
    print('★ 最佳 τ=%.2f  macro-F1=%.4f (v5=%.4f, Δ=%+.4f)' % (tau, best[1], m5, best[1] - m5))
    print('   逐类:', {c: best[2][c] for c in CATS})
    print('   R1 触发 %d 条 / R2 触发 %d 条' % (best[3], best[4]))

    # R5: 也试 τ=0.70 固定版（更保守）
    print()
    print('=== 固定 τ=0.70 明细 ===')
    pred = {}
    r1_list, r2_list = [], []
    for x in recs:
        lab, rule = x['base']
        if lab == 'SUPPORT' and x['cp']:
            lab, rule = 'UNRELATED', 53
            r1_list.append(x['pid'])
        if lab == 'UNRELATED' and rule == 10 and x['cos'] >= 0.70 and not x['cp'] \
                and x['f']['v5_num_verdict'] != 'CONFLICT':
            lab, rule = 'SUPPORT', 39
            r2_list.append((x['pid'], round(x['cos'], 3)))
        pred[x['pid']] = lab
    m, d = score(pred, truth)
    print('macro-F1=%.4f  Δ=%+.4f' % (m, m - m5))
    print('  逐类:', {c: d[c] for c in CATS})
    print('  R1 触发:', r1_list)
    print('  R2 触发:', r2_list)
    # 逐条审计：改了标签的
    flips = [(x['pid'], x['base'][0], pred[x['pid']], truth[x['pid']], x['cp'], round(x['cos'], 3))
             for x in recs if x['base'][0] != pred[x['pid']]]
    import collections
    print('  翻转 n=%d' % len(flips))
    audit = collections.Counter()
    for pid, a, b, t, cp, c in flips:
        audit['正确' if b == t else ('破例' if a == t else '中性')] += 1
    print('  翻转审计:', dict(audit))
    broke = [(pid, a, b, t) for pid, a, b, t, cp, c in flips if a == t and b != t]
    print('  破例明细:')
    for x in broke:
        print('    ', x)


if __name__ == '__main__':
    main()
