# -*- coding: utf-8 -*-
"""E13 真值修正（轮 9 · 通道过判反向核验）

背景：v1.7 重标（轮 7）中，两条独立通道对 4 条「转发/引用型」对**同时**把
RT 归因 `@提及` / hashtag / 情绪尾巴当成「实质新增内容」⇒ 误判 SUPPORT
（v1.6 与系统 v4/v5 均判 DERIVE）。反向复核通道（收紧 A5-② 后）判 8/8 DERIVE。

本脚本：E13_全量真值_v4.csv  →  E13_全量真值_v5.csv
  仅改 4 条：E13-0427/0446/0464/0482  SUPPORT → DERIVE
  新增列 `rv_note` 记录反向核验依据。

产出：E13_全量真值_v5.csv / E13_rv_修正清单.json
"""
import os
import csv
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.abspath(__file__))

CORRECT = {
    'E13-0427': ('SUPPORT', 'DERIVE', 'A5-②收录 RT 归因 @TelegraphNews + "see this" ⇒ 过判；实质新增=0'),
    'E13-0446': ('SUPPORT', 'DERIVE', 'A5-②收录情绪尾巴 "so important to read" ⇒ 过判；实质新增=0'),
    'E13-0464': ('SUPPORT', 'DERIVE', 'A5-②收录 hashtag #tolerance + "anytime" ⇒ 过判；实质新增=0'),
    'E13-0482': ('SUPPORT', 'DERIVE', 'A5-②收录 RT 归因 @Copwatch ⇒ 过判；实质新增=0'),
}

rows = list(csv.DictReader(open(os.path.join(BASE, 'E13_全量真值_v4.csv'), encoding='utf-8-sig')))
cols = list(rows[0].keys()) + ['rv_note']

changed = []
for r in rows:
    pid = r['pair_id']
    r['rv_note'] = ''
    if pid in CORRECT:
        old, new, note = CORRECT[pid]
        assert r['truth'] == old, (pid, r['truth'], old)
        r['truth'] = new
        r['rv_note'] = note
        changed.append({'pair_id': pid, 'old': old, 'new': new, 'note': note})

with open(os.path.join(BASE, 'E13_全量真值_v5.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

import collections
dist = collections.Counter(r['truth'] for r in rows)
print('E13_全量真值_v5.csv 写出，755 行')
print('修正 %d 条：' % len(changed))
for c in changed:
    print('  %s  %s -> %s  | %s' % (c['pair_id'], c['old'], c['new'], c['note']))
print('新分布:', dict(dist))

json.dump({'n': len(rows), 'changed': changed, 'dist': dict(dist)},
          open(os.path.join(BASE, 'E13_rv_修正清单.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('已写 E13_rv_修正清单.json')
