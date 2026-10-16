# -*- coding: utf-8 -*-
"""E13 v6 变体对照（轮 9）

对比：
  V5            基线
  V5+R2(τ)      仅语义复述守卫
  V5+R1+R2(τ)   R1（评论对，rule 8 限定）+ R2
  V5+R1*+R2(τ)  R1（更严：两侧都有强意见词）+ R2
判收：对照零漂移 + macro-F1 不降 + 破例可解释。
"""
import os
import re
import csv
import math
import pickle
import sys
import collections

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import e13_claim_graph_v5 as V5
import e13_slot_extract as SLOT

CATS = ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED']
_EMB = pickle.load(open(os.path.join(BASE, 'E13_v6_emb_cache.pkl'), 'rb'))

_STRONG_OP = re.compile(r"\b(wtf|ridiculous|misleading|seriously|kidding|not buying|"
                        r"how does that make you feel|make you feel|at all|isn't much)\b", re.I)


def cos(a, b):
    n = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return n / (na * nb) if na and nb else 0.0


def has_fam(t):
    return any(w in SLOT.TRIG2CLASS for w in re.findall(r'[a-z]+', (t or '').lower()))


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


rows = list(csv.DictReader(open(os.path.join(BASE, 'E13_标注集_v3.csv'), encoding='utf-8-sig')))
truth = {r['pair_id']: r['truth'] for r in csv.DictReader(open(os.path.join(BASE, 'E13_全量真值_v5.csv'), encoding='utf-8-sig'))}

recs = []
for r in rows:
    p, q = r['p_text'] or '', r['q_text'] or ''
    f = V5.features(p, q)
    struct = (r.get('system_prior') or '').startswith('structural')
    bl, br = V5.predict(f, struct)
    ep, eq = _EMB.get(r['p_tweet_id']), _EMB.get(r['q_tweet_id'])
    recs.append({'pid': r['pair_id'], 'f': f, 'base': (bl, br),
                 'cos': cos(ep, eq) if ep and eq else 0.0,
                 'nofam': (not has_fam(p)) and (not has_fam(q)),
                 'strong_op': bool(_STRONG_OP.search(p) or _STRONG_OP.search(q))})

m5, d5 = score({x['pid']: x['base'][0] for x in recs}, truth)
print('V5 基线: macro-F1=%.4f' % m5)


def run(tau, r1_mode):
    pred, flips = {}, []
    for x in recs:
        lab, rule = x['base']
        if r1_mode == 'rule8' and lab == 'SUPPORT' and rule == 8 and x['nofam']:
            lab = 'UNRELATED'
        elif r1_mode == 'strong' and lab == 'SUPPORT' and x['nofam'] and x['strong_op']:
            lab = 'UNRELATED'
        if lab == 'UNRELATED' and rule == 10 and x['cos'] >= tau and x['f']['v5_num_verdict'] != 'CONFLICT':
            lab = 'SUPPORT'
        pred[x['pid']] = lab
        if lab != x['base'][0]:
            flips.append((x['pid'], x['base'][0], lab, truth[x['pid']]))
    m, d = score(pred, truth)
    aud = collections.Counter('正确' if b == t else ('破例' if a == t else '中性') for _, a, b, t in flips)
    return m, d, flips, aud


print()
for mode in ('none', 'rule8', 'strong'):
    for tau in (0.65, 0.70, 0.75):
        m, d, flips, aud = run(tau, mode)
        print('R1=%-7s τ=%.2f  macro-F1=%.4f (%+.4f)  n_flip=%-3d %s  SUPPORT_F1=%.3f UNREL_F1=%.3f' % (
            mode, tau, m, m - m5, len(flips), dict(aud), d['SUPPORT']['F1'], d['UNRELATED']['F1']))

print()
print('=== 选定：R1=strong + R2(τ=0.70) 逐条翻转 ===')
m, d, flips, aud = run(0.70, 'strong')
print('macro-F1=%.4f (%+.4f)  逐类:' % (m, m - m5), {c: d[c]['F1'] for c in CATS})
for pid, a, b, t in flips:
    print('  %-11s %-9s -> %-9s truth=%-9s %s' % (pid, a, b, t, 'OK' if b == t else ('BREAK' if a == t else 'neutral')))
