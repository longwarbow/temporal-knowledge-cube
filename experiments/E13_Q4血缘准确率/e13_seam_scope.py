# -*- coding: utf-8 -*-
"""E13 三条残余缝体量侦察（轮 9）

对 755 条全量，量化：
  A4-seam1  上位词缝：一侧 victim/casualty/affected（存活未定）数值 vs 另一侧 DEAD/INJURED 数值
  A4-seam2  身份以事实罗列缝：PERPETRATOR_IDENTITY 以事实列表表达
  R1        评论对缝：两侧皆主观评论/无事实命题，系统仍因词汇重叠误判 SUPPORT
  R2        无锚独立复述缝：真值 SUPPORT 但系统判 UNRELATED，且无共同数值锚

产出：E13_seam_scope.json
"""
import os
import re
import csv
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e13_claim_graph_v5 as V5
import e13_slot_extract as SLOT

BASE = os.path.dirname(os.path.abspath(__file__))

AMBIG_WORDS = re.compile(r'\b(victims?|casualt(?:y|ies)|affected)\b', re.I)
DEAD_WORDS = re.compile(r'\b(dead|died|killed|killing|fatalit(?:y|ies)|toll|bodies?|corpse|slain|murdered)\b', re.I)
INJ_WORDS = re.compile(r'\b(injured|wounded|hurt|hospitalized|hospitalised|critical)\b', re.I)

# 身份/背景事实罗列（PERPETRATOR_IDENTITY 以事实表达）
IDENT_FACT = re.compile(
    r'\b(fled|fleeing|refugee|convert|converted|immigrant|asylum|'
    r'had been arrested|prior|previous|record|history of|'
    r'married|ex-wife|divorce|born in|grew up|studied|graduated|'
    r'was charged|convicted|on bail|probation|social media|'
    r'pledge|pledged|allegiance|swore|loyalty|inspired by)\b', re.I)

# 评论/意见标记（无事实命题的迹象）
OPINION = re.compile(
    r"\b(wtf|ridiculous|misleading|seriously|kidding|ur |you're |u r |"
    r"i'm not buying|not buying|so much here|wrong|nonsense|shame|"
    r"should|shouldn't|why |how can|isn't it|no\??$|"
    r"thank|thanks|please|prayers|rip|thoughts)\b", re.I)

NUM_ANCHOR = re.compile(r'\b\d[\d,]*\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|'
                        r'eleven|twelve|dozen|hundred|thousand)\b', re.I)


def load(fn, key='pair_id'):
    return list(csv.DictReader(open(os.path.join(BASE, fn), encoding='utf-8-sig')))


rows = load('E13_标注集_v3.csv')
t16 = {r['pair_id']: r['truth'] for r in load('E13_全量真值_v3.csv')}
t17 = {r['pair_id']: r['truth'] for r in load('E13_全量真值_v4.csv')}
v4 = {r['pair_id']: r['system_pred'] for r in load('E13_系统预测_v4_全量.csv')}
v5 = {r['pair_id']: r['system_pred'] for r in load('E13_系统预测_v5_全量.csv')}

out = {'a4_seam1': [], 'a4_seam2': [], 'r1': [], 'r2': []}
by_stratum = {}

for r in rows:
    pid = r['pair_id']
    p, q = r['p_text'] or '', r['q_text'] or ''
    st = r['stratum']
    t = t17.get(pid)
    f = V5.features(p, q)
    struct = (r.get('system_prior') or '').startswith('structural')
    lab, rule = V5.predict(f, struct)

    # --- A4 seam1: 上位词 ---
    if AMBIG_WORDS.search(p) and (DEAD_WORDS.search(q) or INJ_WORDS.search(q)):
        out['a4_seam1'].append((pid, 'p=AMBIG', t, v5.get(pid), p[:70], q[:70]))
    elif AMBIG_WORDS.search(q) and (DEAD_WORDS.search(p) or INJ_WORDS.search(p)):
        out['a4_seam1'].append((pid, 'q=AMBIG', t, v5.get(pid), p[:70], q[:70]))

    # --- A4 seam2: 身份以事实罗列 ---
    if IDENT_FACT.search(p) or IDENT_FACT.search(q):
        out['a4_seam2'].append((pid, t, v5.get(pid), (IDENT_FACT.search(p) or IDENT_FACT.search(q)).group(0),
                                p[:60], q[:60]))

    # --- R1: 评论对（两侧均无事实命题迹象 + 有意见标记） ---
    pn, qn = bool(NUM_ANCHOR.search(p)), bool(NUM_ANCHOR.search(q))
    if (OPINION.search(p) and OPINION.search(q)) and not (pn and qn):
        out['r1'].append((pid, st, t, v5.get(pid), rule, p[:70], q[:70]))

    # --- R2: 真值 SUPPORT 但系统 UNRELATED，无共同数值锚 ---
    if t == 'SUPPORT' and v5.get(pid) in ('UNRELATED',) and not (pn and qn):
        out['r2'].append((pid, st, t, v4.get(pid), v5.get(pid), f['contain'], p[:70], q[:70]))

for k in out:
    by_stratum[k] = len(out[k])

print('=== 侦察结果（n=755）===')
print('A4-seam1 上位词候选     :', len(out['a4_seam1']))
print('A4-seam2 身份事实罗列   :', len(out['a4_seam2']))
print('R1 评论对候选           :', len(out['r1']))
print('R2 无锚独立复述候选     :', len(out['r2']))

print('\n--- R2 明细（真值SUPPORT / 系统UNRELATED / 无共同数值锚）---')
for x in out['r2']:
    print('  %-11s %-22s v4=%s v5=%s contain=%.3f' % (x[0], x[1], x[3], x[4], x[5]))
    print('      P:', x[6])
    print('      Q:', x[7])

print('\n--- R1 明细（两侧皆评论迹象 & 无双侧数值锚）---')
for x in out['r1']:
    print('  %-11s %-22s truth=%s v5=%s rule=%s' % (x[0], x[1], x[2], x[3], x[4]))
    print('      P:', x[5])
    print('      Q:', x[6])

print('\n--- A4-seam1 明细 ---')
for x in out['a4_seam1']:
    print('  %-11s %-8s truth=%s v5=%s' % (x[0], x[1], x[2], x[3]))
    print('      P:', x[4])
    print('      Q:', x[5])

print('\n--- A4-seam2 明细 ---')
for x in out['a4_seam2']:
    print('  %-11s truth=%s v5=%s hit=%s' % (x[0], x[1], x[2], x[3]))
    print('      P:', x[4])
    print('      Q:', x[5])

json.dump({'counts': by_stratum, 'detail': out},
          open(os.path.join(BASE, 'E13_seam_scope.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('\n已写 E13_seam_scope.json')
