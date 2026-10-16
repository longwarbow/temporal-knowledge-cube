# -*- coding: utf-8 -*-
"""
E13 — $Q_4$：**v1.5 复审分析**（靶向收敛 + 对照完整性 + 合并口径）

复审集 = v1.4 全部分歧项（6） + 等量对照（6，从 v1.4 已一致项随机抽）。
产出：E13_先导v15_分析.json / E13_先导v15_最终真值.csv
"""
import os
import csv
import json
import collections

import e13_pilot_s5s6_analyze as A

BASE = os.path.dirname(os.path.abspath(__file__))
LAB = A.LAB


def load(path):
    out = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            out.append({(k or '').lstrip('\ufeff'): v for k, v in r.items()})
    return out


EXPECT = {   # v1.5 各条款对该条的期望判法（判例核对）
    'E13-0503': ('A3-ⓠ 计量对象不同（人数 vs 起数）', 'UNRELATED'),
    'E13-0539': ('A4-反例清单 被扣押↔枪击/逃离', 'UNRELATED'),
    'E13-0487': ('A3-ⓐ限定 / 不同事件（枪击死者 vs 人质）', 'UNRELATED'),
    'E13-0458': ('A5 via @X 是归因 ⇒ SUPPORT', 'SUPPORT'),
    'E13-0543': ('A4-反例清单 不同谓词（枪手被毙/中枪↔收治伤者）', 'UNRELATED'),
    'E13-0534': ('A4-反例清单 被扣押↔逃离', 'UNRELATED'),
}


def main():
    blind = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_先导v15_盲版.csv'))}
    idx = json.load(open(os.path.join(BASE, 'E13_先导v15_索引.json'), encoding='utf-8'))['map']
    K = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_先导v15_通道K.csv'))}
    L = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_先导v15_通道L.csv'))}
    v14 = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_先导v14_最终真值.csv'))}

    pids = [p for p in blind if p in K and p in L]
    assert len(pids) == len(blind) == len(idx) == 12
    yk = [K[p]['label_relation'] for p in pids]
    yl = [L[p]['label_relation'] for p in pids]
    po, pe, k = A.kappa(yk, yl, LAB)
    dis = [p for p in pids if K[p]['label_relation'] != L[p]['label_relation']]
    print('v1.5 复审 n=%d  p_o=%.3f  κ=%.3f  AC1=%.3f  分歧 %d'
          % (len(pids), po, k, A.gwet_ac1(yk, yl, LAB), len(dis)))

    def role(p):
        return idx[p]['role']

    R1 = [p for p in pids if role(p) == 'R1_disagree']
    R2 = [p for p in pids if role(p) == 'R2_control']

    print('\n[R1 靶向：v1.4 的 6 条分歧]')
    conv = 0
    r1rows = []
    for p in R1:
        agree = int(K[p]['label_relation'] == L[p]['label_relation'])
        conv += agree
        exp = EXPECT.get(p, ('', ''))
        ok = int(agree and K[p]['label_relation'] == exp[1])
        r1rows.append({'pair_id': p, 'rule': exp[0], 'expected': exp[1],
                       'K': K[p]['label_relation'], 'L': L[p]['label_relation'],
                       'converged': agree, 'matches_ruling': ok,
                       'note_K': K[p].get('label_notes', ''), 'note_L': L[p].get('label_notes', '')})
        print('  %s  v1.4: I=%-9s J=%-9s → v1.5: K=%-9s L=%-9s  %s %s'
              % (p, v14[p]['v14_I'], v14[p]['v14_J'], K[p]['label_relation'], L[p]['label_relation'],
                 '✓收敛' if agree else '✗仍分歧', '·符合规则' if ok else '·与规则不符'))
    print('  ⇒ 6 条分歧：收敛 %d 条；其中与 v1.5 条款期望一致 %d 条'
          % (conv, sum(r['matches_ruling'] for r in r1rows)))

    print('\n[R2 对照：v1.4 下本已一致的 6 条]')
    keep = 0
    r2rows = []
    for p in R2:
        agree = int(K[p]['label_relation'] == L[p]['label_relation'])
        same = int(agree and K[p]['label_relation'] == v14[p]['v14_I'])
        keep += same
        r2rows.append({'pair_id': p, 'v14': v14[p]['v14_I'], 'K': K[p]['label_relation'],
                       'L': L[p]['label_relation'], 'still_agree': agree, 'same_label': same})
        print('  %s  v1.4: %-9s → v1.5: K=%-9s L=%-9s  %s %s'
              % (p, v14[p]['v14_I'], K[p]['label_relation'], L[p]['label_relation'],
                 '一致' if agree else '★分歧', '·标签未漂移' if same else '·★标签漂移'))
    print('  ⇒ 对照 6 条：仍一致 %d 条；标签未漂移 %d 条' % (keep, keep))

    # ---------------- 合并口径（全 76 条） ----------------
    v15 = {p: v14[p] for p in v14}
    for p in R1:
        v15[p] = dict(v14[p])
        v15[p]['v14_I'] = v15[p]['v14_J'] = (K[p]['label_relation']
                                             if K[p]['label_relation'] == L[p]['label_relation'] else '★')
    n_dis_76 = sum(1 for p in v15 if v15[p]['v14_I'] != v15[p]['v14_J'])
    print('\n[合并口径] 76 条中两通道仍分歧：v1.4 = 6 条 → v1.5 = %d 条' % n_dis_76)
    print('  ⚠️ v1.5 的这一步是在**被它瞄准的那 6 条**上做的 ⇒ **in-sample**，不算独立证据。')

    # ---------------- 产量影响核查 ----------------
    yld_impact = []
    for p in R1 + R2:
        if p in ('E13-0543', 'E13-0534') or p in ('E13-0503', 'E13-0487', 'E13-0539'):
            yld_impact.append({'pair_id': p, 'v14': v14[p]['v14_I'],
                               'v15_K': K[p]['label_relation'], 'v15_L': L[p]['label_relation'],
                               'is_CONFLICT_any': int(any(x == 'CONFLICT' for x in
                                                          (K[p]['label_relation'], L[p]['label_relation'])))})
    print('\n[产量敏感性] 复审条目里有无 CONFLICT：%s'
          % {r['pair_id']: r['is_CONFLICT_any'] for r in yld_impact})
    print('  ⇒ 若全部为 0，则 C 块（样本外）的 S6 产量 **不受 v1.5 影响**，产量仍可视为样本外。')

    with open(os.path.join(BASE, 'E13_先导v15_最终真值.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pair_id', 'role', 'v14_I', 'v14_J', 'v15_K', 'v15_L', 'v15_agree',
                    'expected', 'matches_ruling', 'note_K', 'note_L'])
        for p in pids:
            exp = EXPECT.get(p, ('', ''))
            agree = int(K[p]['label_relation'] == L[p]['label_relation'])
            w.writerow([p, role(p), v14[p]['v14_I'], v14[p]['v14_J'],
                        K[p]['label_relation'], L[p]['label_relation'], agree,
                        exp[1],
                        int(agree == 1 and K[p]['label_relation'] == exp[1]),
                        K[p].get('label_notes', ''), L[p].get('label_notes', '')])
    print('[save] E13_先导v15_最终真值.csv')

    res = {
        'n': len(pids),
        'overall': {'p_o': round(po, 4), 'kappa': round(k, 4),
                    'gwet_ac1': round(A.gwet_ac1(yk, yl, LAB), 4), 'n_disagree': len(dis),
                    'disagree_ids': dis},
        'R1_target': {'n': len(R1), 'n_converged': conv,
                      'n_matches_ruling': sum(r['matches_ruling'] for r in r1rows),
                      'detail': r1rows},
        'R2_control': {'n': len(R2), 'n_still_agree': keep, 'n_same_label': keep, 'detail': r2rows},
        'merged_76_disagreements': {'v14': 6, 'v15': n_dis_76},
        'yield_sensitivity_rerun_items': yld_impact,
        'caveat': ('复审集 = v1.4 全部分歧项 + 等量对照 ⇒ v1.5 的收敛是 **in-sample** 的靶向结果；'
                   '对照项通过只能说明"没打散"，不能证明判据好。独立证据 = 全量标注的人类 κ。'),
    }
    with open(os.path.join(BASE, 'E13_先导v15_分析.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('[save] E13_先导v15_分析.json')
    return res


if __name__ == '__main__':
    main()
