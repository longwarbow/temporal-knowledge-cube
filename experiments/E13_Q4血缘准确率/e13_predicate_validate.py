# -*- coding: utf-8 -*-
"""
E13 — A4 封闭谓词表：**靶向验证**（谓词门 vs v1.5 共识真值）

A4 门的职责是二值闸，且**只在"两侧都带数值族"时才适用**（第三态 not_applicable）：

    pass            两侧数值绑到同一族 ⇒ 放行到 ③c T5（可能 SUPPORT/CONFLICT）
    block           两侧都绑到族、但族不同 ⇒ **UNRELATED**（不同属性 / 不同动作或阶段）
    not_applicable  任一侧无数值族 ⇒ A4 **不适用**，交回 ④ §2.1 / ⑤ 兜底

于是本验证回答三件事：
  ① A4 门的**适用范围**（applicable 占比）与在该范围内的**判别力**；
  ② **UNRELATED 的构成分解**：多少由 A4 机械决定，多少必须靠 T1/T2/T3/A3-ⓠ 其它条款
     —— 这是"哪一条是约束条款"的量化（同 E13-S5 敏感带、E12 阈值口径的做法）；
  ③ 判据 v1.5 §2.5 反例清单里**点名**的边界族是否被复现。

产物：E13_谓词表验证.json、E13_谓词表验证_明细.csv
"""
import os
import csv
import json
import collections

import e13_predicate_match as PM

BASE = os.path.dirname(os.path.abspath(__file__))


def load(p):
    with open(os.path.join(BASE, p), encoding='utf-8-sig', newline='') as fh:
        return [{(k or '').lstrip('\ufeff'): v for k, v in r.items()} for r in csv.DictReader(fh)]


# 判据 v1.5 §2.5 反例清单里**点名**的边界族（须被复现为 block）
NAMED_FAMILIES = [
    ('E13-0488', '被扣押 ↔ 逃离'),
    ('E13-0506', '被扣押 ↔ 逃离'),
    ('E13-0511', '被扣押 ↔ 逃离'),
    ('E13-0540', '被扣押 ↔ 离开'),
    ('E13-0541', '被扣押 ↔ 逃离'),
    ('E13-0542', '被扣押 ↔ 逃离'),
    ('E13-0491', '存在嫌犯 ↔ 嫌犯身份/背景'),
    ('E13-0543', '死亡 ↔ 医院收治'),
    ('E13-0534', '被扣押 ↔ 枪击/逃离'),
]

GATE = {'same_family': 'pass', 'diff_family': 'block', 'not_applicable': 'not_applicable'}

# ------------------------------------------------------------------ 人工归因表
# 对「A4 适用但判 pass」的 UNRELATED，逐条指定**真正的约束条款**。
# ⚠️ **单编（single-coder）、可复核**：不是机械计算，是读原文后的归类，逐条 pair_id 可回溯。
# 目的：量化"A4 到底覆盖了多少 UNRELATED"，回答"封不封闭"这个投入产出问题。
CLAUSE_ATTRIB = {
    # A3-ⓠ 计量对象不同（起数/处数 vs 人数）
    'E13-0486': 'A3-ⓠ 计量对象', 'E13-0500': 'A3-ⓠ 计量对象', 'E13-0503': 'A3-ⓠ 计量对象',
    # A3-① 部分标记未指向同一总体 ⇒ 不同总体
    'E13-0487': 'A3-① 总体限定',
    # T3 主体/情形不同（不同事件；含"统计 vs 个案"）
    'E13-0498': 'T3 情形不同', 'E13-0523': 'T3 情形不同', 'E13-0525': 'T3 情形不同',
    'E13-0526': 'T3 情形不同', 'E13-0531': 'T3 情形不同', 'E13-0544': 'T3 情形不同',
    # T2 隐含（祝愿 / 预期性表述不构成断言）
    'E13-0547': 'T2 隐含',
    # T1 非命题（提问 / 祈使 / 观点 / 揣测 / 反问）——**最大的一类**
    'E13-0493': 'T1 非命题', 'E13-0496': 'T1 非命题', 'E13-0510': 'T1 非命题',
    'E13-0516': 'T1 非命题', 'E13-0519': 'T1 非命题', 'E13-0530': 'T1 非命题',
    'E13-0551': 'T1 非命题', 'E13-0553': 'T1 非命题', 'E13-0556': 'T1 非命题',
    'E13-0557': 'T1 非命题', 'E13-0561': 'T1 非命题', 'E13-0562': 'T1 非命题',
    'E13-0565': 'T1 非命题', 'E13-0566': 'T1 非命题', 'E13-0567': 'T1 非命题',
    'E13-0572': 'T1 非命题', 'E13-0573': 'T1 非命题', 'E13-0574': 'T1 非命题',
    'E13-0576': 'T1 非命题', 'E13-0580': 'T1 非命题', 'E13-0581': 'T1 非命题',
}


def main():
    rows = load('E13_标注集_v2.csv')
    truth = {r['pair_id']: r['truth'] for r in load('E13_先导v15_合并真值.csv')}

    detail = []
    for r in rows:
        if not r['stratum'].startswith(('S5_', 'S6_')):
            continue
        rel, d = PM.predicate_relation(r['p_text'], r['q_text'])
        detail.append({
            'pair_id': r['pair_id'], 'stratum': r['stratum'], 'subtype': r['subtype'],
            'truth': truth.get(r['pair_id'], ''),
            'pred_rel': rel, 'gate': GATE[rel], 'pred_reason': d.get('reason', ''),
            'fam_p': '+'.join(sorted(d['sp'])) or '-',
            'fam_q': '+'.join(sorted(d['sq'])) or '-',
            'inter': '+'.join(d['inter']) or '-',
        })

    lab = [d for d in detail if d['truth']]
    n_lab = len(lab)
    pos = [d for d in lab if d['truth'] in ('SUPPORT', 'CONFLICT')]   # 人工判"同一槽位"
    neg = [d for d in lab if d['truth'] == 'UNRELATED']

    # ---------------- ① 适用范围与判别力 ----------------
    def split(rows_):
        return ([d for d in rows_ if d['gate'] != 'not_applicable'],
                [d for d in rows_ if d['gate'] == 'not_applicable'])

    pos_app, pos_na = split(pos)
    neg_app, neg_na = split(neg)
    fn = [d for d in pos_app if d['gate'] == 'block']      # 该过的被拦（假阴性）
    fp = [d for d in neg_app if d['gate'] == 'pass']       # 该拦的放行（假阳性）

    # ---------------- ② UNRELATED 的构成分解 ----------------
    decomp = collections.Counter()
    clause = {}
    for d in neg:
        if d['gate'] == 'block':
            key = 'A4_block（谓词不可归并，机械可决）'
        elif d['gate'] == 'not_applicable':
            key = 'A4_not_applicable（无数值族）'
        else:
            key = '需其它条款：%s' % CLAUSE_ATTRIB.get(d['pair_id'], '未归因')
        decomp[key] += 1
        clause[d['pair_id']] = key
    n_attrib = sum(1 for d in neg if d['gate'] == 'pass')
    n_missing = sum(1 for d in neg if d['gate'] == 'pass' and d['pair_id'] not in CLAUSE_ATTRIB)

    # ---------------- ③ 点名边界族 ----------------
    by_id = {d['pair_id']: d for d in detail}
    named = []
    for pid, desc in NAMED_FAMILIES:
        d = by_id.get(pid)
        named.append({'pair_id': pid, 'case': desc, 'ok': (d or {}).get('gate') == 'block',
                      'gate': (d or {}).get('gate', 'MISSING'),
                      'pred_rel': (d or {}).get('pred_rel', ''),
                      'truth': (d or {}).get('truth', ''),
                      'fam_p': (d or {}).get('fam_p', ''), 'fam_q': (d or {}).get('fam_q', '')})

    # ---------------- ④ 覆盖度 / 分层 ----------------
    cov = collections.Counter()
    for d in detail:
        cov['both_sides_have_family' if (d['fam_p'] != '-' and d['fam_q'] != '-')
            else 'one_side_no_family'] += 1
    by_stratum = {}
    for st in ('S5_cross_thread_verbatim', 'S6_same_slot_conflicting'):
        sub = [d for d in detail if d['stratum'] == st]
        by_stratum[st] = {'n': len(sub),
                          'pred_rel': dict(collections.Counter(d['pred_rel'] for d in sub))}

    # ---------------- ⑤ 门施加于 v3 ----------------
    v3 = {r['pair_id']: r['system_pred'] for r in load('E13_系统预测_v3.csv')}
    gate_changed = []
    for d in lab:
        p3, p4 = v3.get(d['pair_id']), v3.get(d['pair_id'])
        if d['gate'] == 'block' and p3 in ('CONFLICT', 'SUPPORT'):
            p4 = 'UNRELATED'      # A4 门：谓词不可归并 ⇒ 同一槽位不成立
        if p4 != p3:
            gate_changed.append({'pair_id': d['pair_id'], 'truth': d['truth'],
                                 'v3': p3, 'after_gate': p4, 'pred_rel': d['pred_rel']})

    res = {
        'n_pairs_scanned': len(detail), 'n_labelled': n_lab,
        'applicability': {'labelled_total': n_lab,
                          'applicable': len(pos_app) + len(neg_app),
                          'not_applicable': len(pos_na) + len(neg_na)},
        'dir1_false_negative': {
            'case': '真值∈{SUPPORT,CONFLICT}（人工判同一槽位）',
            'n_applicable': len(pos_app), 'n_passed': len(pos_app) - len(fn),
            'recall': (len(pos_app) - len(fn)) / max(1, len(pos_app)), 'items': fn},
        'dir2_false_positive': {
            'case': '真值=UNRELATED 且 A4 适用', 'n_applicable': len(neg_app),
            'n_blocked': len(neg_app) - len(fp),
            'precision': (len(neg_app) - len(fp)) / max(1, len(neg_app)), 'items': fp},
        'unrelated_decomposition': dict(decomp),
        'unrelated_decomposition_note': (
            '人工归因（单编、可复核，逐条 pair_id 见 CLAUSE_ATTRIB）；未归因 %d 条' % n_missing),
        'named_boundary_families': named,
        'coverage': dict(cov), 'by_stratum': by_stratum,
        'gate_on_v3': {'n_changed': len(gate_changed), 'changes': gate_changed},
        'note': ('真值取自 v1.5 合并真值（其中 6 条为 v1.5 复审裁定 ⇒ in-sample）；'
                 'A4 门只在"两侧都带数值族"时适用'),
    }
    with open(os.path.join(BASE, 'E13_谓词表验证.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(BASE, 'E13_谓词表验证_明细.csv'), 'w',
              encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['pair_id', 'stratum', 'subtype', 'truth',
                                           'pred_rel', 'gate', 'pred_reason',
                                           'fam_p', 'fam_q', 'inter'])
        w.writeheader()
        w.writerows(detail)

    # ---------------- 控制台 ----------------
    print('扫描 %d 对（S5+S6），已标注 %d 对' % (len(detail), n_lab))
    print('\n[① 适用范围] A4 可用 %d / %d（not_applicable %d）= %.1f%%'
          % (len(pos_app) + len(neg_app), n_lab, len(pos_na) + len(neg_na),
             100.0 * (len(pos_app) + len(neg_app)) / max(1, n_lab)))
    print('[① 判别力] 正类（SUPPORT/CONFLICT）n=%d → pass %d (%.1f%%)'
          % (len(pos_app), len(pos_app) - len(fn),
             100.0 * (len(pos_app) - len(fn)) / max(1, len(pos_app))))
    for d in fn:
        print('    ✗FN %s truth=%-9s pred=%s  P=%s Q=%s' % (d['pair_id'], d['truth'],
                                                           d['pred_rel'], d['fam_p'], d['fam_q']))
    print('           负类（UNRELATED，A4 适用）n=%d → block %d (%.1f%%)'
          % (len(neg_app), len(neg_app) - len(fp),
             100.0 * (len(neg_app) - len(fp)) / max(1, len(neg_app))))
    for d in fp:
        print('    ✗FP %s pred=%s  P=%s Q=%s' % (d['pair_id'], d['pred_rel'], d['fam_p'], d['fam_q']))

    print('\n[② UNRELATED 构成分解] n=%d（人工归因，未归因 %d 条）' % (len(neg), n_missing))
    for k, v in sorted(decomp.items(), key=lambda x: -x[1]):
        print('    %-42s %3d  (%.1f%%)' % (k, v, 100.0 * v / max(1, len(neg))))

    print('\n[③ 点名边界族]')
    for n in named:
        print('    %s %s %-24s gate=%-14s truth=%-9s P=%s Q=%s' % (
            '✅' if n['ok'] else '❌', n['pair_id'], n['case'], n['gate'],
            n['truth'], n['fam_p'], n['fam_q']))

    print('\n[④ 覆盖度] %s' % dict(cov))
    for st, v in by_stratum.items():
        print('    %-32s n=%-3d pred_rel=%s' % (st, v['n'], v['pred_rel']))

    print('\n[⑤ 门施加于 v3] 改变 %d 对' % len(gate_changed))
    for c in gate_changed:
        print('    %s truth=%-9s v3=%-9s → %-9s (%s)' % (c['pair_id'], c['truth'],
                                                         c['v3'], c['after_gate'], c['pred_rel']))
    return res


if __name__ == '__main__':
    main()
