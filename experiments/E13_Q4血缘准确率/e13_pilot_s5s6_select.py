# -*- coding: utf-8 -*-
"""
E13 — $Q_4$：**S5 / S6 先导抽样**（测"候选生成器"的产量）

目的
----
S5/S6 是**候选生成器**，其 DERIVE / CONFLICT **产量（yield）必须实测**，不能假设。
本脚本从 6 层标注集里抽 S5/S6 子集，产出**盲版**（不含 stratum / subtype / 量槽 /
系统预测），供独立通道按**冻结判据 v1.3** 标注。

抽样
----
  S5：20 条（随机）
  S6：40 条，按子型配额 —— strict 20 / polarity 12 / vague 8
  （vague 为**负对照**：按 T5「量的兼容性」，模糊量词与具体数值**可兼容**，预期不是 CONFLICT）
合计 60，顺序打散（`seed=20261011`）。

产物
----
  E13_先导S5S6_盲版.csv   交给标注通道
  E13_先导S5S6_索引.json  pair_id → (stratum, subtype, 系统预测)（**不给标注员**）
"""
import os
import csv
import json
import random
import collections

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = random.Random(20261011)
N_S5 = 20
QUOTA_S6 = {'strict': 20, 'polarity': 12, 'vague': 8}

BLIND_COLS = ['pair_id', 'event', 'p_text', 'q_text',
              'label_relation', 'label_confidence', 'label_notes']


def main():
    rows = list(csv.DictReader(open(os.path.join(BASE, 'E13_标注集_v2.csv'), encoding='utf-8-sig')))
    pred = {r['pair_id']: r for r in csv.DictReader(
        open(os.path.join(BASE, 'E13_系统预测_v2.csv'), encoding='utf-8-sig'))}

    s5 = [r for r in rows if r['stratum'] == 'S5_cross_thread_verbatim']
    s6 = [r for r in rows if r['stratum'] == 'S6_same_slot_conflicting']
    RNG.shuffle(s5)
    picked = s5[:N_S5]

    for kind, q in QUOTA_S6.items():
        pool = [r for r in s6 if r['subtype'] == kind]
        RNG.shuffle(pool)
        picked.extend(pool[:q])
        print('  S6/%-9s 池 %2d → 取 %2d' % (kind, len(pool), min(q, len(pool))))

    RNG.shuffle(picked)          # 打散顺序，避免同层/同子型扎堆

    blind = [{k: r.get(k, '') for k in BLIND_COLS} for r in picked]
    with open(os.path.join(BASE, 'E13_先导S5S6_盲版.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=BLIND_COLS)
        w.writeheader()
        w.writerows(blind)
    print('[save] E13_先导S5S6_盲版.csv（%d 行）' % len(blind))

    idx = {
        'seed': 20261011, 'n': len(picked),
        'design': {'S5_random': N_S5, 'S6_quota': QUOTA_S6},
        'map': {r['pair_id']: {'stratum': r['stratum'], 'subtype': r['subtype'],
                               'slot': r['system_prior'],
                               'system_pred': pred.get(r['pair_id'], {}).get('system_pred', '')}
                for r in picked},
        'leakage_note': ('盲版只含 pair_id/event/p_text/q_text + 待填列；'
                         'stratum（列名自带答案）、subtype、量槽、system_pred 全部剔除'),
        'warning': '本索引含 stratum/subtype，**不得**交给标注员',
    }
    with open(os.path.join(BASE, 'E13_先导S5S6_索引.json'), 'w', encoding='utf-8') as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    print('[save] E13_先导S5S6_索引.json')
    print('分层：%s' % dict(collections.Counter(v['stratum'] for v in idx['map'].values())))
    print('子型：%s' % dict(collections.Counter(v['subtype'] for v in idx['map'].values() if v['subtype'])))


if __name__ == '__main__':
    main()
