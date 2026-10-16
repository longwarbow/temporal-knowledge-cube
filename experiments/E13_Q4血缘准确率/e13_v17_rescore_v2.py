# -*- coding: utf-8 -*-
"""
E13 — 三版真值评分（轮 9）
--------------------------------------------------------
对同一份系统预测（v4 / v5），用三版真值分别评分：
  v1.6 真值  = E13_全量真值_v3.csv   （原标，判据 v1.6）
  v1.7 真值  = E13_全量真值_v4.csv   （轮 7 重标，判据 v1.7）
  v1.7r 真值 = E13_全量真值_v5.csv   （轮 9 反向核验修正 4 条后）
输出：E13_v17_重标_评分.json（覆盖，含三版）
"""
import os
import csv
import json

BASE = os.path.dirname(os.path.abspath(__file__))
CATS = ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED']


def col(f, c):
    return {(r['pair_id'] or '').strip(): (r.get(c) or '').strip().upper()
            for r in csv.DictReader(open(os.path.join(BASE, f), encoding='utf-8-sig'))}


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
    T = {
        'v16': col('E13_全量真值_v3.csv', 'truth'),
        'v17': col('E13_全量真值_v4.csv', 'truth'),
        'v17r': col('E13_全量真值_v5.csv', 'truth'),
    }
    P = {'v4': col('E13_系统预测_v4_全量.csv', 'system_pred'),
         'v5': col('E13_系统预测_v5_全量.csv', 'system_pred')}

    res = {}
    for pn, pred in P.items():
        res[pn] = {}
        for tn, tr in T.items():
            m, d = score(pred, tr)
            res[pn][tn] = {'macro_F1': m, 'per_class': d}

    h = {
        'v4_v16': res['v4']['v16']['macro_F1'], 'v5_v16': res['v5']['v16']['macro_F1'],
        'v4_v17': res['v4']['v17']['macro_F1'], 'v5_v17': res['v5']['v17']['macro_F1'],
        'v4_v17r': res['v4']['v17r']['macro_F1'], 'v5_v17r': res['v5']['v17r']['macro_F1'],
        'gain_on_v16': round(res['v5']['v16']['macro_F1'] - res['v4']['v16']['macro_F1'], 4),
        'gain_on_v17': round(res['v5']['v17']['macro_F1'] - res['v4']['v17']['macro_F1'], 4),
        'gain_on_v17r': round(res['v5']['v17r']['macro_F1'] - res['v4']['v17r']['macro_F1'], 4),
        'dist_v17r': {c: res['v5']['v17r']['per_class'][c]['n_true'] for c in CATS},
    }
    res['headline'] = h
    json.dump(res, open(os.path.join(BASE, 'E13_v17_重标_评分.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    print('               v1.6真值   v1.7真值   v1.7r真值(修正)')
    print('  v4            %.4f     %.4f     %.4f' % (h['v4_v16'], h['v4_v17'], h['v4_v17r']))
    print('  v5            %.4f     %.4f     %.4f' % (h['v5_v16'], h['v5_v17'], h['v5_v17r']))
    print('  v5-v4 增益     +%.4f    +%.4f    +%.4f' % (h['gain_on_v16'], h['gain_on_v17'], h['gain_on_v17r']))
    print()
    print('  逐类 (v5 vs v1.7r):')
    for c in CATS:
        d = res['v5']['v17r']['per_class'][c]
        print('    %-10s n=%-4d P=%.3f R=%.3f F1=%.3f' % (c, d['n_true'], d['P'], d['R'], d['F1']))
    print()
    print('  v1.7r 分布:', h['dist_v17r'])


if __name__ == '__main__':
    main()
