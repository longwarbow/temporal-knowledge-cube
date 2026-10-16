# -*- coding: utf-8 -*-
"""
E13 — $Q_4$：**v1.5 复审集抽样**（全部分歧项 + 等量对照）

依据：技能 §2.5.3 纪律 2「修订必须带'对照项'」——
复审集 = **全部分歧项** + 从**已一致项**随机抽的**等量对照**。
只测靶向项会得到"修订完美"的假象。

产物：E13_先导v15_盲版.csv / E13_先导v15_索引.json
"""
import os
import csv
import json
import random

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = random.Random(20261012)

BLIND_COLS = ['pair_id', 'event', 'p_text', 'q_text',
              'label_relation', 'label_confidence', 'label_notes']


def load(path):
    out = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            out.append({(k or '').lstrip('\ufeff'): v for k, v in r.items()})
    return out


def main():
    blind = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_先导v14_盲版.csv'))}
    v14 = load(os.path.join(BASE, 'E13_先导v14_最终真值.csv'))

    dis = [r['pair_id'] for r in v14 if r['agree'] == '0']
    agreed = [r['pair_id'] for r in v14 if r['agree'] == '1']
    RNG.shuffle(agreed)
    ctrl = agreed[:len(dis)]

    picked = dis + ctrl
    RNG.shuffle(picked)
    print('复审集 = 分歧 %d 条 %s' % (len(dis), dis))
    print('        + 对照 %d 条 %s' % (len(ctrl), ctrl))

    with open(os.path.join(BASE, 'E13_先导v15_盲版.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=BLIND_COLS)
        w.writeheader()
        w.writerows([{k: blind[p].get(k, '') for k in BLIND_COLS} for p in picked])
    print('[save] E13_先导v15_盲版.csv（%d 行）' % len(picked))

    v14map = {r['pair_id']: r for r in v14}
    idx = {
        'seed': 20261012, 'n': len(picked),
        'design': {'R1_prev_disagree': dis, 'R2_control': ctrl,
                   'note': '对照 = 从 v1.4 下两通道已一致的 70 条中随机抽等量'},
        'map': {p: {'role': ('R1_disagree' if p in dis else 'R2_control'),
                    'v14_I': v14map[p]['v14_I'], 'v14_J': v14map[p]['v14_J'],
                    'prev_consensus': v14map[p]['prev_consensus']}
                for p in picked},
        'warning': '含 v1.4 真值，**不得**交给标注员',
    }
    with open(os.path.join(BASE, 'E13_先导v15_索引.json'), 'w', encoding='utf-8') as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    print('[save] E13_先导v15_索引.json')


if __name__ == '__main__':
    main()
