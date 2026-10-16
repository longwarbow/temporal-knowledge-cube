# -*- coding: utf-8 -*-
"""
E13 — 全量 v3 双通道合并与一致性分析
输入：E13_标注集_v3.csv（元数据）+ 通道 M/N 各 4 分片
输出：E13_全量v3_一致性.json + E13_全量v3_分歧清单.csv + E13_全量v3_通道M.csv / _通道N.csv（合并后单文件）
"""
import os
import csv
import json
import collections

BASE = os.path.dirname(os.path.abspath(__file__))
LABELS = ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED']


def load(p):
    return [{(k or '').lstrip('\ufeff'): v for k, v in r.items()}
            for r in csv.DictReader(open(p, encoding='utf-8-sig'))]


def kappa(y1, y2, labels=LABELS):
    n = len(y1)
    po = sum(a == b for a, b in zip(y1, y2)) / n
    pe = sum(sum(a == l for a in y1) * sum(b == l for b in y2) for l in labels) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def ac1(y1, y2, labels=LABELS):
    """Gwet's AC1 —— 对极端偏斜分布稳健。"""
    n = len(y1)
    pi_l = {l: (sum(a == l for a in y1) + sum(b == l for b in y2)) / (2 * n) for l in labels}
    po = sum(a == b for a, b in zip(y1, y2)) / n
    pe = sum(pi_l.values()) - sum(v * v for v in pi_l.values())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    meta = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_标注集_v3.csv'))}
    M, N = {}, {}
    for part in range(1, 5):
        for ch, store in (('M', M), ('N', N)):
            for r in load(os.path.join(BASE, 'E13_全量v3_通道%s_part%d.csv' % (ch, part))):
                store[r['pair_id']] = r
    assert set(M) == set(meta) and set(N) == set(meta), '通道覆盖不全 M=%d N=%d meta=%d' % (len(M), len(N), len(meta))
    # 合并单文件
    for ch, store in (('M', M), ('N', N)):
        with open(os.path.join(BASE, 'E13_全量v3_通道%s.csv' % ch), 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['pair_id', 'label_relation', 'label_confidence', 'label_notes'])
            for pid in sorted(meta):
                r = store[pid]
                w.writerow([pid, r['label_relation'], r['label_confidence'], r.get('label_notes', '')])

    pids = sorted(meta)
    ym = [M[p]['label_relation'] for p in pids]
    yn = [N[p]['label_relation'] for p in pids]
    k, po = kappa(ym, yn)
    a = ac1(ym, yn)

    # 逐层
    per_stratum = {}
    for st in sorted(set(r['stratum'] for r in meta.values())):
        sp = [p for p in pids if meta[p]['stratum'] == st]
        y1 = [M[p]['label_relation'] for p in sp]
        y2 = [N[p]['label_relation'] for p in sp]
        kk, pp = kappa(y1, y2)
        per_stratum[st] = {
            'n': len(sp), 'agree_rate': round(pp, 4), 'kappa': round(kk, 4),
            'dist_M': dict(collections.Counter(y1)), 'dist_N': dict(collections.Counter(y2)),
        }
    # 逐 S6 子型
    per_sub = {}
    for st in sorted(set(r['subtype'] for r in meta.values() if r['stratum'].startswith('S6'))):
        sp = [p for p in pids if meta[p]['stratum'] == 'S6_same_slot_conflicting' and meta[p]['subtype'] == st]
        y1 = [M[p]['label_relation'] for p in sp]
        per_sub[st] = {'n': len(sp), 'dist_M': dict(collections.Counter(y1))}

    # 分歧清单
    dis = [p for p in pids if M[p]['label_relation'] != N[p]['label_relation']]
    with open(os.path.join(BASE, 'E13_全量v3_分歧清单.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pair_id', 'stratum', 'subtype', 'label_M', 'label_N', 'note_M', 'note_N'])
        for p in dis:
            w.writerow([p, meta[p]['stratum'], meta[p]['subtype'],
                        M[p]['label_relation'], N[p]['label_relation'],
                        M[p].get('label_notes', ''), N[p].get('label_notes', '')])

    out = {
        'n': len(pids),
        'kappa': round(k, 4), 'agree_rate': round(po, 4), 'gwet_ac1': round(a, 4),
        'n_disagree': len(dis),
        'per_stratum': per_stratum,
        'S6_subtypes_dist_channelM': per_sub,
        'label_dist_M': dict(collections.Counter(ym)),
        'label_dist_N': dict(collections.Counter(yn)),
        'files': {'M': 'E13_全量v3_通道M.csv', 'N': 'E13_全量v3_通道N.csv',
                  'disagree': 'E13_全量v3_分歧清单.csv'},
    }
    with open(os.path.join(BASE, 'E13_全量v3_一致性.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('n=%d  agree=%.4f  κ=%.4f  AC1=%.4f  分歧=%d' % (len(pids), po, k, a, len(dis)))
    for st, d in per_stratum.items():
        print('  %-32s n=%3d agree=%.3f κ=%.3f  M:%s' % (st, d['n'], d['agree_rate'], d['kappa'], d['dist_M']))
    print('  S6 子型（M）：', {kk: vv for kk, vv in per_sub.items()})
    print('  分歧按层：', dict(collections.Counter(meta[p]['stratum'] for p in dis)))
    print('  分歧对型（M↔N）TOP：', collections.Counter((M[p]['label_relation'], N[p]['label_relation']) for p in dis).most_common(6))


if __name__ == '__main__':
    main()
