# -*- coding: utf-8 -*-
"""
E13 — 全量 v3 真值合并（通道 M / N 全量 + 通道 O 仅分歧项）
规则：M==N ⇒ 共识；否则三通道多数表决；三方全异 ⇒ 标 'tie'（交由第 4 通道）。
输出：E13_全量真值_v3.csv + E13_全量v3_仲裁汇总.json
"""
import os
import csv
import json
import collections

BASE = os.path.dirname(os.path.abspath(__file__))


def load(p):
    return [{(k or '').lstrip('\ufeff'): v for k, v in r.items()}
            for r in csv.DictReader(open(p, encoding='utf-8-sig'))]


def main():
    meta = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_标注集_v3.csv'))}
    M = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_全量v3_通道M.csv'))}
    N = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_全量v3_通道N.csv'))}
    O = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_全量v3_通道O.csv'))}
    pids = sorted(meta)
    rows, stats, ties = [], collections.Counter(), []
    for p in pids:
        a, b = M[p]['label_relation'], N[p]['label_relation']
        if a == b:
            truth, how = a, 'M=N'
        else:
            c = O[p]['label_relation']
            votes = collections.Counter([a, b, c]).most_common()
            if votes[0][1] >= 2:
                truth, how = votes[0][0], 'majority(M,N,O)'
            else:
                truth, how = 'TIE', 'all_distinct'
                ties.append(p)
        stats[how] += 1
        rows.append({'pair_id': p, 'stratum': meta[p]['stratum'], 'subtype': meta[p]['subtype'],
                     'event': meta[p]['event'], 'p_tweet_id': meta[p]['p_tweet_id'],
                     'q_tweet_id': meta[p]['q_tweet_id'],
                     'label_M': a, 'label_N': b, 'label_O': O[p]['label_relation'] if p in O else '',
                     'truth': truth, 'how': how,
                     'note_M': M[p].get('label_notes', ''), 'note_N': N[p].get('label_notes', ''),
                     'note_O': O[p].get('label_notes', '') if p in O else ''})
    with open(os.path.join(BASE, 'E13_全量真值_v3.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    dist = collections.Counter(r['truth'] for r in rows if r['truth'] != 'TIE')
    per = {}
    for st in sorted(set(r['stratum'] for r in rows)):
        sp = [r for r in rows if r['stratum'] == st and r['truth'] != 'TIE']
        per[st] = {'n': len(sp), 'dist': dict(collections.Counter(r['truth'] for r in sp))}
    out = {'n': len(pids), 'resolution': dict(stats), 'n_tie': len(ties), 'ties': ties,
           'truth_dist': dict(dist), 'per_stratum_truth': per}
    with open(os.path.join(BASE, 'E13_全量v3_仲裁汇总.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('n=%d  %s' % (len(pids), dict(stats)))
    print('真值分布：', dict(dist))
    for st, d in per.items():
        print('  %-32s n=%3d %s' % (st, d['n'], d['dist']))
    if ties:
        print('三方全异（需第 4 通道）：%s' % ties)


if __name__ == '__main__':
    main()
