# -*- coding: utf-8 -*-
"""E13 轮 9 回归：v6 vs v5（对 v1.7r 真值）

产出 E13_v9_回归.json
  - metrics: v4/v5/v6 对 v1.7r 真值的逐类 P/R/F1
  - flips: v5 → v6 翻转 + 审计（正确/破例/中性）
  - targets: R1/R2 靶向（真值侧确认的候选）
  - controls: 冻结对照（不得漂移）
"""
import os
import csv
import json
import collections

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


# 轮 9 靶向（真值侧确认、v5 判错、v6 应改对的）
TARGETS = {
    'E13-0086': 'UNRELATED', 'E13-0093': 'UNRELATED',           # R1
    'E13-0229': 'SUPPORT', 'E13-0255': 'SUPPORT', 'E13-0256': 'SUPPORT',
    'E13-0260': 'SUPPORT', 'E13-0261': 'SUPPORT', 'E13-0262': 'SUPPORT',
    'E13-0263': 'SUPPORT', 'E13-0267': 'SUPPORT', 'E13-0269': 'SUPPORT',
    'E13-0279': 'SUPPORT', 'E13-0285': 'SUPPORT', 'E13-0289': 'SUPPORT',
    'E13-0291': 'SUPPORT', 'E13-0346': 'SUPPORT', 'E13-0380': 'SUPPORT',
    'E13-0385': 'SUPPORT', 'E13-0545': 'SUPPORT', 'E13-0738': 'SUPPORT',
}
# 冻结对照（v5→v6 不得漂移；取 v1.7 轮对照中与 R2 无冲突者 + 正典）
CONTROLS = ['E13-0019', 'E13-0139', 'E13-0503', 'E13-0487', 'E13-0458',
            'E13-0491', 'E13-0534', 'E13-0539', 'E13-0543',
            'E13-0427', 'E13-0446', 'E13-0464', 'E13-0482']


def main():
    truth = col('E13_全量真值_v5.csv', 'truth')
    v4 = col('E13_系统预测_v4_全量.csv', 'system_pred')
    v5 = col('E13_系统预测_v5_全量.csv', 'system_pred')
    v6 = col('E13_系统预测_v6_全量.csv', 'system_pred')
    rule6 = col('E13_系统预测_v6_全量.csv', 'system_pred_rule')

    m4, o4 = score(v4, truth)
    m5, o5 = score(v5, truth)
    m6, o6 = score(v6, truth)

    flips = []
    for p in truth:
        if v5.get(p) != v6.get(p):
            flips.append({'pair_id': p, 'v5': v5.get(p), 'v6': v6.get(p),
                          'truth': truth[p], 'v6_rule': rule6.get(p)})
    aud = collections.Counter()
    for f in flips:
        if f['v5'] == f['truth'] and f['v6'] != f['truth']:
            aud['broke'] += 1
        elif f['v5'] != f['truth'] and f['v6'] == f['truth']:
            aud['correct'] += 1
        else:
            aud['neutral'] += 1

    t_ok = t_bad = 0
    for p, want in TARGETS.items():
        ok = v6.get(p) == want
        t_ok += ok
        t_bad += not ok
    c_bad = 0
    drift = []
    for p in CONTROLS:
        if v5.get(p) != v6.get(p):
            c_bad += 1
            drift.append((p, v5.get(p), v6.get(p)))
    c_ok = len(CONTROLS) - c_bad

    res = {
        'metrics': {'v4': m4, 'v5': m5, 'v6': m6, 'delta_v6_v5': round(m6 - m5, 4)},
        'per_class': {'v5': o5, 'v6': o6},
        'n_flips': len(flips), 'flip_audit': dict(aud), 'flips': flips,
        'targets': {'ok': t_ok, 'bad': t_bad}, 'controls': {'ok': c_ok, 'bad': c_bad},
        'control_drift': drift,
    }
    json.dump(res, open(os.path.join(BASE, 'E13_v9_回归.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    print('对 v1.7r 真值（n=755）:')
    print('  v4=%.4f  v5=%.4f  v6=%.4f  (v6-v5 = %+.4f)' % (m4, m5, m6, m6 - m5))
    print('  逐类:')
    for c in CATS:
        print('    %-10s v5 P%.3f R%.3f F1%.3f  ->  v6 P%.3f R%.3f F1%.3f' %
              (c, o5[c]['P'], o5[c]['R'], o5[c]['F1'], o6[c]['P'], o6[c]['R'], o6[c]['F1']))
    print('  翻转 %d 条: %s' % (len(flips), dict(aud)))
    print('  靶向 %d/%d  对照 %d/%d（漂移 %s）' % (t_ok, t_ok + t_bad, c_ok, c_ok + c_bad, drift or '无'))
    print('[save] E13_v9_回归.json')


if __name__ == '__main__':
    main()
