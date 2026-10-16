# -*- coding: utf-8 -*-
"""
E13 — v1.7 重标后重新评分（轮 7）
--------------------------------------------------------
对同一份系统预测（v4 / v5），分别用**旧真值（v1.6，E13_全量真值_v3.csv）**
与**新真值（v1.7，E13_全量真值_v4.csv）**评分，量化"重标"对结论的影响。
输出：E13_v17_重标_评分.json
"""
import os, csv, json

BASE = os.path.dirname(os.path.abspath(__file__))
CATS = ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED']


def col(f, c):
    return {(r['pair_id'] or '').strip(): (r.get(c) or '').strip().upper()
            for r in csv.DictReader(open(os.path.join(BASE, f), encoding='utf-8-sig'))}


def score(pred, truth):
    n = len(truth)
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
    t16 = col('E13_全量真值_v3.csv', 'truth')
    t17 = col('E13_全量真值_v4.csv', 'truth')
    v4 = col('E13_系统预测_v4_全量.csv', 'system_pred')
    v5 = col('E13_系统预测_v5_全量.csv', 'system_pred')

    res = {}
    for name, pred in (('v4', v4), ('v5', v5)):
        m16, d16 = score(pred, t16)
        m17, d17 = score(pred, t17)
        res[name] = {'vs_v16': {'macro_F1': m16, 'per_class': d16},
                     'vs_v17': {'macro_F1': m17, 'per_class': d17}}
    res['headline'] = {
        'v4_v16': res['v4']['vs_v16']['macro_F1'], 'v5_v16': res['v5']['vs_v16']['macro_F1'],
        'v4_v17': res['v4']['vs_v17']['macro_F1'], 'v5_v17': res['v5']['vs_v17']['macro_F1'],
        'gain_v5_over_v4_on_v16': round(res['v5']['vs_v16']['macro_F1'] - res['v4']['vs_v16']['macro_F1'], 4),
        'gain_v5_over_v4_on_v17': round(res['v5']['vs_v17']['macro_F1'] - res['v4']['vs_v17']['macro_F1'], 4),
    }
    json.dump(res, open(os.path.join(BASE, 'E13_v17_重标_评分.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    h = res['headline']
    print('            v1.6 真值      v1.7 真值')
    print('  v4        %.4f        %.4f' % (h['v4_v16'], h['v4_v17']))
    print('  v5        %.4f        %.4f' % (h['v5_v16'], h['v5_v17']))
    print('  v5-v4 增益  +%.4f       +%.4f   <== 重标后增益几乎归零'
          % (h['gain_v5_over_v4_on_v16'], h['gain_v5_over_v4_on_v17']))


if __name__ == '__main__':
    main()
