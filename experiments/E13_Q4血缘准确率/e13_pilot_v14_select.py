# -*- coding: utf-8 -*-
"""
E13 — $Q_4$：**v1.4 验证集抽样**（靶向 + 对照 + 样本外刷新）

设计（承技能 §2.5.3「修订必须带对照项」）
----------------------------------------
上一轮（判据 v1.3）的 60 条先导暴露了 **6 条分歧**，全部集中在**同两类边界**上
（量的口径、谓词同一性），另有 3 条落在 DERIVE/SUPPORT·T1 例外的边界。本轮把它们
作为**靶向集**，并配上**等量的对照集**（上一轮本已一致的样本）——只测靶向项会得到
"修订完美"的假象；必须同时检验**修订有没有打散本来一致的东西**。

三块：
  A 靶向集 = 上一轮 60 条中「触及上述边界」的全部样本（机械关键词抽取 + 分歧项强制纳入）
  B 对照集 = 上一轮 60 条中**其余**项，随机抽 |A| 条（seed 固定）
  C 样本外刷新集 = 《标注集_v2》里 S6 候选，**排除上一轮 60 条**，按子型配额
                  strict 20 / polarity 12 / vague 8 = 40 条
  ⇒ 用于在 **A3/A4 生效后**重测 S6 产量（保持"产量样本外"这一性质）。

产物：
  E13_先导v14_盲版.csv   交给通道（只含 pair_id/event/p_text/q_text + 待填列）
  E13_先导v14_索引.json  pair_id → 块/分层/子型/上一轮真值（**不得给标注员**）
"""
import os
import csv
import json
import random
import collections

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = random.Random(20261012)

N_FRESH = {'strict': 20, 'polarity': 12, 'vague': 8}

# 边界关键词（覆盖上一轮 6 条分歧 + 两条缝的全部机制）
KW_SEAM = ['子集', '⊂', '范围不同', '单位不同', '总数', '部分',
           '谓词不同', '情形同一', '同一情形', '人质数', '中枪', '逃离', '逃脱',
           '归因', '照抄', 'T1例外', '非命题']
# 上一轮两通道分歧项（agree=0）——无论关键词是否命中，**强制**纳入靶向集
FORCE_TARGET = ['E13-0558', 'E13-0491', 'E13-0503', 'E13-0536', 'E13-0472', 'E13-0482']

BLIND_COLS = ['pair_id', 'event', 'p_text', 'q_text',
              'label_relation', 'label_confidence', 'label_notes']


def main():
    rows = list(csv.DictReader(open(os.path.join(BASE, 'E13_标注集_v2.csv'), encoding='utf-8-sig')))
    truth = list(csv.DictReader(open(os.path.join(BASE, 'E13_先导S5S6_最终真值.csv'),
                                    encoding='utf-8-sig')))
    prev_idx = json.load(open(os.path.join(BASE, 'E13_先导S5S6_索引.json'), encoding='utf-8'))['map']
    pred = {r['pair_id']: r.get('system_pred', '') for r in csv.DictReader(
        open(os.path.join(BASE, 'E13_系统预测_v2.csv'), encoding='utf-8-sig'))}
    byid = {r['pair_id']: r for r in rows}

    prev_ids = list(prev_idx.keys())

    # ---------------- A 靶向集 ----------------
    A = set(FORCE_TARGET)
    hit_reason = collections.defaultdict(list)
    for t in truth:
        note = ((t.get('p_note_G') or '') + ' ' + (t.get('p_note_H') or ''))
        for kw in KW_SEAM:
            if kw in note:
                A.add(t['pair_id'])
                hit_reason[t['pair_id']].append(kw)
                break
    A = [p for p in prev_ids if p in A]
    RNG.shuffle(A)

    # ---------------- B 对照集（等量，来自本已一致项） ----------------
    rest = [p for p in prev_ids if p not in set(A)]
    RNG.shuffle(rest)
    B = rest[:len(A)]

    # ---------------- C 样本外刷新集 ----------------
    s6_all = [r for r in rows if r['stratum'] == 'S6_same_slot_conflicting']
    s6_fresh_pool = [r for r in s6_all if r['pair_id'] not in set(prev_ids)]
    C = []
    for kind, quota in N_FRESH.items():
        pool = [r for r in s6_fresh_pool if r['subtype'] == kind]
        RNG.shuffle(pool)
        take = pool[:quota]
        C.extend(r['pair_id'] for r in take)
        print('  [C] S6/%-9s 剩余池 %3d → 取 %2d' % (kind, len(pool), len(take)))
    print('  [C] 全量 S6 池 %d，其中未用于上一轮者 %d ⇒ 说明样本外刷新确有新样本'
          % (len(s6_all), len(s6_fresh_pool)))

    picked = A + B + C
    assert len(set(picked)) == len(picked), '三块有重叠！'

    # ---------------- 盲版 ----------------
    blind = []
    for pid in picked:
        r = byid[pid]
        blind.append({k: r.get(k, '') for k in BLIND_COLS})
    with open(os.path.join(BASE, 'E13_先导v14_盲版.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=BLIND_COLS)
        w.writeheader()
        w.writerows(blind)
    print('[save] E13_先导v14_盲版.csv（%d 行；A=%d B=%d C=%d）' % (len(blind), len(A), len(B), len(C)))

    # ---------------- 索引 ----------------
    tmap = {t['pair_id']: t for t in truth}
    blocks = {}
    for pid in A:
        blocks[pid] = 'A_target'
    for pid in B:
        blocks[pid] = 'B_control'
    for pid in C:
        blocks[pid] = 'C_fresh'
    idx = {
        'seed': 20261012,
        'n': len(picked),
        'design': {
            'A_target': '上一轮 60 条中触及边界者（含 6 条分歧：%s）' % ','.join(FORCE_TARGET),
            'B_control': '上一轮 60 条中本已一致者，随机抽等量 |A|',
            'C_fresh': 'S6 候选池排除上一轮 60 条后按子型配额再抽 40（用于样本外重测产量）',
            'quota_fresh': N_FRESH,
        },
        'map': {pid: {
            'block': blocks[pid],
            'stratum': byid[pid]['stratum'],
            'subtype': byid[pid]['subtype'],
            'system_pred_v2': pred.get(pid, ''),
            'prev_G': tmap.get(pid, {}).get('G', ''),
            'prev_H': tmap.get(pid, {}).get('H', ''),
            'prev_consensus': tmap.get(pid, {}).get('consensus_label', ''),
            'prev_agree': tmap.get(pid, {}).get('agree', ''),
            'hit_keywords': hit_reason.get(pid, []),
        } for pid in picked},
        'leakage_note': ('盲版只含 pair_id/event/p_text/q_text + 待填列；'
                         'block/stratum/subtype/量槽/系统预测/上一轮真值 全部剔除'),
        'warning': '本索引含分层与上一轮真值，**不得**交给标注员',
    }
    with open(os.path.join(BASE, 'E13_先导v14_索引.json'), 'w', encoding='utf-8') as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    print('[save] E13_先导v14_索引.json')
    print('块分布：%s' % dict(collections.Counter(blocks.values())))
    print('A 靶向命中关键词：%s' % dict(collections.Counter(
        kw for v in idx['map'].values() if v['block'] == 'A_target' for kw in v['hit_keywords'])))
    print('A 中上一轮分歧项：%s' % [p for p in A if idx['map'][p]['prev_agree'] == '0'])


if __name__ == '__main__':
    main()
