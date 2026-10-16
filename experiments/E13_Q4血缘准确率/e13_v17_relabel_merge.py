# -*- coding: utf-8 -*-
"""
E13 — v1.7 重标合并（轮 7）
--------------------------------------------------------
输入：
  重标盲版 v17_重标/通道A_shard{1..4}.csv   （通道 A，全量 238）
           v17_重标/通道B_shard{1..4}.csv   （通道 B，全量 238）
           v17_重标/通道O_仲裁.csv          （通道 O，仅 17 条分歧）
  冻结 v1.6 真值 E13_全量真值_v3.csv
规则（对齐协议 §6）：
  A==B            ⇒ 共识，how='v1.7:A=B'
  A!=B            ⇒ 取仲裁通道 O，how='v1.7:arb(O)'
  A!=B 且 O 也不等于 A/B（三方全异） ⇒ 标 'TIE'（留待第 4 通道）
未进入重标集的 517 条：**沿用 v1.6 冻结标签**，how='keep:v1.6'
输出：
  E13_全量真值_v4.csv            （755 行，新增 relabeled 标记）
  E13_v17_重标_汇总.json          （κ / 变化清单 / 新旧分布）
"""
import os, csv, json, collections

BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, 'v17_重标')
CATS = ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED']


def load(p):
    d = {}
    for r in csv.DictReader(open(p, encoding='utf-8-sig')):
        pid = (r.get('pair_id') or '').strip()
        d[pid] = {(k or '').lstrip('\ufeff'): v for k, v in r.items()}
    return d


def lab(r):
    return (r.get('label_relation') or '').strip().upper()


def kappa(a, b):
    n = len(a)
    po = sum(1 for p in a if a[p] == b.get(p)) / n
    ca, cb = collections.Counter(a.values()), collections.Counter(b.values())
    pe = sum((ca[c] / n) * (cb[c] / n) for c in CATS)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0, po


def main():
    A, B = {}, {}
    for i in (1, 2, 3, 4):
        A.update(load(os.path.join(R, '通道A_shard%d.csv' % i)))
        B.update(load(os.path.join(R, '通道B_shard%d.csv' % i)))
    O = load(os.path.join(R, '通道O_仲裁.csv'))
    v16 = load(os.path.join(BASE, 'E13_全量真值_v3.csv'))

    LA = {p: lab(r) for p, r in A.items()}
    LB = {p: lab(r) for p, r in B.items()}
    LO = {p: lab(r) for p, r in O.items()}

    rel = sorted(set(LA) | set(LB))
    assert all(p in LA and p in LB for p in rel), '通道缺行'
    k, po = kappa(LA, LB)
    dis = [p for p in rel if LA[p] != LB[p]]

    merged, how, ties = {}, {}, []
    for p in rel:
        a, b = LA[p], LB[p]
        if a == b:
            merged[p], how[p] = a, 'v1.7:A=B'
        else:
            o = LO.get(p, '')
            merged[p], how[p] = o, 'v1.7:arb(O)'
            if o not in (a, b):
                ties.append(p)
                how[p] = 'v1.7:TIE'

    # 组装 755
    rows, changed = [], []
    for pid, r in v16.items():
        from collections import OrderedDict
        row = OrderedDict((c, r.get(c, '')) for c in
                          ['pair_id', 'stratum', 'subtype', 'event', 'p_tweet_id', 'q_tweet_id',
                           'label_M', 'label_N', 'label_O', 'truth', 'how',
                           'note_M', 'note_N', 'note_O', 'note_P'])
        if pid in merged:
            row['label_M'] = LA[pid]
            row['label_N'] = LB[pid]
            row['label_O'] = LO.get(pid, '') if pid in LO else ''
            row['truth'] = merged[pid]
            row['how'] = how[pid]
            row['relabeled'] = 1
            if merged[pid] != r.get('truth'):
                changed.append(pid)
        else:
            row['relabeled'] = 0
        rows.append(row)

    with open(os.path.join(BASE, 'E13_全量真值_v4.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    d16 = collections.Counter(r['truth'] for r in rows if r['relabeled'] == 0)
    d17 = collections.Counter(r['truth'] for r in rows)
    out = {
        'n': len(rows), 'relabeled': len(rel), 'unchanged': len(rows) - len(rel),
        'channel_A_B_kappa': round(k, 4), 'channel_po': round(po, 4),
        'n_disagree': len(dis), 'disagree': dis,
        'n_tie': len(ties), 'ties': ties,
        'n_label_changed': len(changed), 'changed': sorted(changed),
        'dist_full': dict(d17), 'dist_relabel_set': dict(collections.Counter(merged.values())),
        'changed_matrix': dict(collections.Counter(
            '%s->%s' % (v16[p]['truth'], merged[p]) for p in changed)),
    }
    json.dump(out, open(os.path.join(BASE, 'E13_v17_重标_汇总.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    print('重标集 %d | 未动 %d | 通道A/B κ=%.4f (po=%.4f) | 分歧 %d | 三方全异 %d'
          % (len(rel), out['unchanged'], k, po, len(dis), len(ties)))
    print('标签实际变化：%d 条' % len(changed))
    print('变化矩阵：', out['changed_matrix'])
    print('v1.7 真值分布：', out['dist_full'])
    if ties:
        print('三方全异（需第 4 通道）：', ties)


if __name__ == '__main__':
    main()
