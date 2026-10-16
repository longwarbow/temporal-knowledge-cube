# -*- coding: utf-8 -*-
"""E13 R2 语义复述缝分析（轮 9）

用 PHEME 自带的 384 维文本嵌入，在 755 对上算 q/p 余弦相似度，
看它能否把「真值 SUPPORT 但系统判 UNRELATED」的"无锚独立复述"与
「真值 UNRELATED 系统也 UNRELATED」的寻常无关对分开。

产出：E13_r2_semantic.json
"""
import os
import io
import csv
import json
import math
import pickle
import sys

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.abspath(__file__))
PHEME = r'D:\WSPforWB\10.keyan\场景1-海量短文本\数据\PHEME\pheme_cascades.gpickle'


def load_emb():
    obj = pickle.load(open(PHEME, 'rb'))
    emb = {}
    for G in obj:
        for n, d in G.nodes(data=True):
            s = str(n)
            if s.isdigit() and d.get('text_embedding') is not None:
                emb[s] = d['text_embedding']
    return emb


def cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return num / (na * nb) if na and nb else 0.0


emb = load_emb()
rows = list(csv.DictReader(open(os.path.join(BASE, 'E13_标注集_v3.csv'), encoding='utf-8-sig')))
t17r = {r['pair_id']: r['truth'] for r in csv.DictReader(open(os.path.join(BASE, 'E13_全量真值_v5.csv'), encoding='utf-8-sig'))}
v5 = {r['pair_id']: r['system_pred'] for r in csv.DictReader(open(os.path.join(BASE, 'E13_系统预测_v5_全量.csv'), encoding='utf-8-sig'))}

# 两个目标组
grp_sm = []   # truth=SUPPORT & v5=UNRELATED  （R2 想捞回的）
grp_un = []   # truth=UNRELATED & v5=UNRELATED（必须不被误捞）
grp_all = []
for r in rows:
    pid = r['pair_id']
    ep, eq = emb.get(r['p_tweet_id']), emb.get(r['q_tweet_id'])
    if ep is None or eq is None:
        continue
    c = cos(ep, eq)
    t, pr = t17r[pid], v5[pid]
    grp_all.append((pid, r['stratum'], t, pr, round(c, 4)))
    if t == 'SUPPORT' and pr == 'UNRELATED':
        grp_sm.append((pid, r['stratum'], round(c, 4), r['p_text'][:60], r['q_text'][:60]))
    elif t == 'UNRELATED' and pr == 'UNRELATED':
        grp_un.append((pid, r['stratum'], round(c, 4)))


def stats(xs):
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return {'n': n, 'mean': round(sum(xs) / n, 4), 'p10': xs[int(n * .1)],
            'median': xs[n // 2], 'p90': xs[int(n * .9)], 'max': xs[-1]}


s_sm = stats([x[2] for x in grp_sm])
s_un = stats([x[2] for x in grp_un])
print('=== 余弦相似度分布 ===')
print('组1 truth=SUPPORT & v5=UNRELATED (R2目标):', s_sm)
print('组2 truth=UNRELATED & v5=UNRELATED (对照) :', s_un)

# 阈值扫描
print()
print('=== 阈值扫描：若用「cos ≥ τ 且 判 UNRELATED ⇒ 改 SUPPORT」===')
print('%-6s %-10s %-10s %-10s' % ('τ', '捞回(R2)', '误捞(组2)', '净'))
best = None
for ti in range(30, 86, 5):
    tau = ti / 100
    rec = sum(1 for x in grp_sm if x[2] >= tau)
    fp = sum(1 for x in grp_un if x[2] >= tau)
    print('%-6.2f %-10s %-10s %-10d' % (tau, '%d/%d' % (rec, len(grp_sm)), '%d/%d' % (fp, len(grp_un)), rec - fp))
    if best is None or (rec - fp) > best[1]:
        best = (tau, rec - fp, rec, fp)
print('最佳阈值: τ=%.2f 捞回 %d/%d 误捞 %d/%d' % (best[0], best[2], len(grp_sm), best[3], len(grp_un)))

print()
print('=== R2 目标组明细（cos 降序）===')
for x in sorted(grp_sm, key=lambda z: -z[2]):
    print('  %-11s %-32s cos=%.3f' % (x[0], x[1], x[2]))
    print('      P:', x[3])
    print('      Q:', x[4])

json.dump({'stats': {'r2_target': s_sm, 'control_unrelated': s_un},
           'threshold_scan': [{'tau': ti / 100,
                               'recall_r2': sum(1 for x in grp_sm if x[2] >= ti / 100),
                               'fp_control': sum(1 for x in grp_un if x[2] >= ti / 100)}
                              for ti in range(30, 86, 5)],
           'r2_target': grp_sm, 'n_control': len(grp_un)},
          open(os.path.join(BASE, 'E13_r2_semantic.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('\n已写 E13_r2_semantic.json')
