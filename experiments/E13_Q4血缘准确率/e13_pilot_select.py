# -*- coding: utf-8 -*-
"""
E13 — $Q_4$ 血缘准确率：**盲版标注表 + 先导 50 条抽样**

修协议漏洞（v1 → v1.1 的动因之一）
----------------------------------
`E13_标注集.csv` 直接发给标注员会**污染盲标**：
  · `stratum` 列名自带答案（`S3_cross_thread_opposite_veracity` ⇒ 暗示 CONFLICT；
    `S4_..._random` ⇒ 暗示 UNRELATED）；
  · `system_prior` 列（"structural: q replies to p" / "cos=0.99" / "veracity: p=true,q=false"）
    直接给出采样逻辑；
  · `p_veracity` / `q_veracity` 是 PHEME 的**金标**，veracity 相反对在 S3 内**就是**冲突的先验。
⇒ 本脚本产出 **盲版标注表**：仅保留判定所必需的信息
  （`pair_id` / `event` / `p_text` / `q_text` + 待填标签列）。

先导子集
--------
分层抽 **50** 条（S1 13 / S2 12 / S3 12 / S4 13），固定种子。
样本量依据：`E12`/D8-rev1 的**「比例·粗」档**（$n_{\\min}\\approx0.96/\\varepsilon^2$，
$\\varepsilon=0.15$ ⇒ $n\\approx43$）——先导用粗档校准判据，全量再用「精档」97/层。

产物
----
  · E13_盲版标注表.csv        —— 388 条盲版（去掉一切泄露列）
  · E13_先导50条_盲版.csv      —— 先导 50 条盲版（交给标注员）
  · E13_先导50条_索引.json     —— pair_id → stratum 映射（**不给标注员**，仅分析用）
"""
import os
import csv
import json
import random
import collections

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = random.Random(20261011)
PILOT_N = {'S1_same_thread_reply': 13,
           'S2_cross_thread_high_sim': 12,
           'S3_cross_thread_opposite_veracity': 12,
           'S4_cross_thread_random': 13}      # 合计 50

BLIND_COLS = ['pair_id', 'event', 'p_text', 'q_text',
              'label_relation', 'label_confidence', 'label_notes']


def main():
    src = os.path.join(BASE, 'E13_标注集.csv')
    rows = list(csv.DictReader(open(src, encoding='utf-8-sig')))
    print('读入 %d 行' % len(rows))

    # ---------- 1) 盲版全表 ----------
    blind = [{k: r.get(k, '') for k in BLIND_COLS} for r in rows]
    p_blind = os.path.join(BASE, 'E13_盲版标注表.csv')
    with open(p_blind, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=BLIND_COLS)
        w.writeheader()
        w.writerows(blind)
    print('[save] E13_盲版标注表.csv（%d 行 × %d 列，已剥离 stratum/system_prior/veracity）'
          % (len(blind), len(BLIND_COLS)))

    # ---------- 2) 先导子集（分层 + 固定种子） ----------
    by_s = collections.defaultdict(list)
    for r in rows:
        by_s[r['stratum']].append(r)
    picked = []
    for s, n in PILOT_N.items():
        pool = list(by_s[s])
        RNG.shuffle(pool)
        picked.extend(pool[:n])
        print('  %-38s 池 %3d → 取 %2d' % (s, len(pool), n))
    # 按 pair_id 排序，稳定输出
    picked.sort(key=lambda r: r['pair_id'])

    p_pilot = os.path.join(BASE, 'E13_先导50条_盲版.csv')
    with open(p_pilot, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=BLIND_COLS)
        w.writeheader()
        for r in picked:
            w.writerow({k: r.get(k, '') for k in BLIND_COLS})
    print('[save] E13_先导50条_盲版.csv（%d 行）' % len(picked))

    idx = {
        'seed': 20261011,
        'pilot_n': PILOT_N,
        'total_pilot': len(picked),
        'sample_size_rationale': ('先导取 E12/D8-rev1「比例·粗」档 n≈0.96/eps^2 '
                                  '(eps=0.15) ⇒ n≈43；全量用「精档」eps=0.10 ⇒ 97/层'),
        'map': {r['pair_id']: r['stratum'] for r in picked},
        'event_distribution': dict(sorted(
            collections.Counter(r['event'] for r in picked).items(), key=lambda kv: -kv[1])),
        'leakage_note': ('盲版已剔除 stratum / system_prior / p_veracity / q_veracity / '
                         'system_pred —— 这些列会给出采样逻辑与金标先验'),
        'warning': '本索引含 stratum，**不得**交给标注员',
    }
    with open(os.path.join(BASE, 'E13_先导50条_索引.json'), 'w', encoding='utf-8') as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    print('[save] E13_先导50条_索引.json')
    print('事件分布：%s' % idx['event_distribution'])


if __name__ == '__main__':
    main()
