# -*- coding: utf-8 -*-
"""
E13 — $Q_4$：**v1.4 验证分析**（靶向收敛 / 对照完整性 / 样本外产量 / 判例核对）

判据为 **v1.4（候选冻结版）**。三块样本各司其职：
  A_target  —— 上一轮触及边界的 20 条（含 6 条分歧）：只测"修订收敛"会得到假象
  B_control —— 上一轮本已一致的 20 条：必须**不被打散**，否则修订失败
  C_fresh   —— 新的 36 条 S6：在 A3/A4 生效下重测产量（**样本外**）

产出：E13_先导v14_分析.json / E13_先导v14_最终真值.csv
"""
import os
import csv
import json
import collections

import e13_pilot_s5s6_analyze as A

BASE = os.path.dirname(os.path.abspath(__file__))
LAB = A.LAB


def load(path):
    """读 CSV 并把表头里的**多重 BOM** 归一化（通道可能写出 \\ufeff\\ufeffpair_id）。"""
    rows = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append({(k or '').lstrip('\ufeff'): v for k, v in r.items()})
    return rows


def main():
    blind = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_先导v14_盲版.csv'))}
    idx = json.load(open(os.path.join(BASE, 'E13_先导v14_索引.json'), encoding='utf-8'))['map']
    I = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_先导v14_通道I.csv'))}
    J = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_先导v14_通道J.csv'))}

    pids = [p for p in blind if p in I and p in J]
    assert len(pids) == len(blind) == len(idx), (len(pids), len(blind), len(idx))
    print('可比较 %d 条（盲版 %d / 索引 %d）' % (len(pids), len(blind), len(idx)))

    yI = [I[p]['label_relation'] for p in pids]
    yJ = [J[p]['label_relation'] for p in pids]

    def agree_stats(sel):
        yi = [I[p]['label_relation'] for p in sel]
        yj = [J[p]['label_relation'] for p in sel]
        if not sel:
            return None
        po, pe, k = A.kappa(yi, yj, LAB)
        dis = [p for p in sel if I[p]['label_relation'] != J[p]['label_relation']]
        return {'n': len(sel), 'p_o': round(po, 4), 'kappa': round(k, 4),
                'gwet_ac1': round(A.gwet_ac1(yi, yj, LAB), 4),
                'n_disagree': len(dis), 'disagree_ids': dis}

    allst = agree_stats(pids)
    lo, hi = A.boot_ci(yI, yJ, LAB)
    print('\n[总一致性] κ=%.3f (95%%CI %.3f–%.3f)  AC1=%.3f  分歧 %d/%d'
          % (allst['kappa'], lo, hi, allst['gwet_ac1'], allst['n_disagree'], allst['n']))

    blocks = {b: [p for p in pids if idx[p]['block'] == b]
              for b in ('A_target', 'B_control', 'C_fresh')}
    per = {b: agree_stats(v) for b, v in blocks.items()}
    for b, v in per.items():
        print('  %-10s n=%-3d κ=%-7s 分歧 %d/%d' % (b, v['n'], v['kappa'], v['n_disagree'], v['n']))

    consensus = {p: (I[p]['label_relation'] if I[p]['label_relation'] == J[p]['label_relation'] else '')
                 for p in pids}

    # ---------------- ① 靶向：上一轮 6 条分歧是否收敛 ----------------
    prev_dis = [p for p in pids if idx[p]['prev_agree'] == '0']
    conv = []
    for p in prev_dis:
        row = {'pair_id': p, 'prev_G': idx[p]['prev_G'], 'prev_H': idx[p]['prev_H'],
               'v14_I': I[p]['label_relation'], 'v14_J': J[p]['label_relation'],
               'converged': int(I[p]['label_relation'] == J[p]['label_relation'])}
        conv.append(row)
        print('  [收敛] %s  v1.3: G=%-9s H=%-9s → v1.4: I=%-9s J=%-9s  %s'
              % (p, row['prev_G'], row['prev_H'], row['v14_I'], row['v14_J'],
                 '✓收敛' if row['converged'] else '✗仍分歧'))
    n_conv = sum(r['converged'] for r in conv)
    print('  ⇒ 上轮 %d 条分歧，v1.4 下收敛 %d 条' % (len(conv), n_conv))

    # ---------------- ② 对照：本已一致项是否被打散 ----------------
    fam = []
    for p in blocks['B_control']:
        newc, oldc = consensus[p], idx[p]['prev_consensus']
        fam.append({'pair_id': p, 'prev_consensus': oldc, 'v14_I': I[p]['label_relation'],
                    'v14_J': J[p]['label_relation'], 'v14_consensus': newc,
                    'still_agree': int(newc != ''), 'same_label': int(newc == oldc)})
    n_sa = sum(r['still_agree'] for r in fam)
    n_sl = sum(r['same_label'] for r in fam)
    print('\n[对照 B] 30 条基准：仍然两通道一致 %d/%d；且标签与上一轮相同 %d/%d'
          % (n_sa, len(fam), n_sl, len(fam)))
    broke = [r for r in fam if r['still_agree'] and not r['same_label']]
    if broke:
        print('  ⚠️ 被"打散"（仍一致但改了标签）：')
        for r in broke:
            print('     %s  %s → %s' % (r['pair_id'], r['prev_consensus'], r['v14_consensus']))

    # A 块里非分歧项的标签漂移（修订是否越界改动）
    driftA = []
    for p in blocks['A_target']:
        if not consensus[p]:
            continue
        if idx[p]['prev_consensus'] and consensus[p] != idx[p]['prev_consensus']:
            driftA.append({'pair_id': p, 'prev_consensus': idx[p]['prev_consensus'],
                           'v14_consensus': consensus[p],
                           'prev_agree': idx[p]['prev_agree']})
    print('  [A 块标签漂移] %d 条：%s' % (len(driftA), [d['pair_id'] for d in driftA]))

    # ---------------- ③ 样本外产量（C 块，v1.4 口径） ----------------
    def yld(sel, label='CONFLICT'):
        sub = [p for p in sel if consensus[p]]
        hit = sum(1 for p in sub if consensus[p] == label)
        l, h = A.wilson(hit, len(sub))
        return {'n_agree': len(sub), 'hit': hit,
                'yield': round(hit / len(sub), 4) if sub else None,
                'ci95_wilson': [l, h],
                'dist': dict(collections.Counter(consensus[p] for p in sub))}

    C = blocks['C_fresh']
    y = {
        'S6_all_fresh': yld(C),
        'S6_strict_fresh': yld([p for p in C if idx[p]['subtype'] == 'strict']),
        'S6_polarity_fresh': yld([p for p in C if idx[p]['subtype'] == 'polarity']),
        'S6_vague_fresh_NEG': yld([p for p in C if idx[p]['subtype'] == 'vague']),
    }
    print('\n[★ 样本外产量 v1.4（C 块 36 条）]')
    for kk, vv in y.items():
        print('  %-22s n=%-3s 命中=%-3s yield=%-7s CI95=%-16s 分布=%s'
              % (kk, vv['n_agree'], vv['hit'], vv['yield'], str(vv['ci95_wilson']), vv['dist']))
    print('  对比 v1.3（上一轮 S6 40 条）：all 11.1% / strict 17.6% / polarity 9.1% / vague 0%')

    # 全 76 条上的 S6 产量（含增量样本；仅作参考，A/B 块非随机）
    y76 = yld([p for p in pids if idx[p]['stratum'].startswith('S6')])
    print('  [参考] 全部 S6 条目（含 A/B 块中的 S6）n=%d 命中=%d yield=%s'
          % (y76['n_agree'], y76['hit'], y76['yield']))

    # ---------------- ④ A3 / A4 判例核对 ----------------
    cases = ['E13-0536', 'E13-0503', 'E13-0491', 'E13-0521', 'E13-0522', 'E13-0542']
    ruling = {'E13-0536': ('A3-① 显式部分标记 "these two"', 'SUPPORT'),
              'E13-0503': ('A3-③ 量纲不同（人数 vs 事件数）', 'UNRELATED'),
              'E13-0491': ('A4-不可归并（存在 ↔ 身份背景）', 'UNRELATED'),
              'E13-0521': ('A3-② 下界"至少11死" + 具体子项', 'SUPPORT'),
              'E13-0522': ('A3-② 至少2死 vs 1+1=2', 'SUPPORT'),
              'E13-0542': ('A4-不可归并（被扣押 ↔ 逃脱）', 'UNRELATED')}
    cas = []
    for p in cases:
        if p not in consensus:
            continue
        r = {'pair_id': p, 'ruling': ruling[p][0], 'expected': ruling[p][1],
             'I': I[p]['label_relation'], 'J': J[p]['label_relation'],
             'consensus': consensus[p] or '(分歧)',
             'match': int(consensus[p] == ruling[p][1])}
        cas.append(r)
        print('  [判例] %s 规则=%s 期望=%-9s 实测 I=%-9s J=%-9s  %s'
              % (p, ruling[p][0], ruling[p][1], r['I'], r['J'],
                 '✓符合' if r['match'] else ('（通道分歧）' if not r['consensus'] else '✗不符')))

    # ---------------- 系统 v2 对 v1.4 共识 ----------------
    sysp = {r['pair_id']: r['system_pred'] for r in csv.DictReader(
        open(os.path.join(BASE, 'E13_系统预测_v2.csv'), encoding='utf-8-sig'))}
    sc = [p for p in pids if consensus[p]]
    m = A.metrics([consensus[p] for p in sc], [sysp.get(p, 'UNRELATED') for p in sc])
    m['n'] = len(sc)
    print('\n[系统 v2 对 v1.4 共识 n=%d] macro-F1=%.4f' % (len(sc), m['macro_F1']))
    for L in LAB:
        print('  %-10s n_true=%-3d n_pred=%-3d P=%.3f R=%.3f F1=%.3f'
              % (L, m[L]['n_true'], m[L]['n_pred'], m[L]['P'], m[L]['R'], m[L]['F1']))

    # ---------------- 落盘 ----------------
    with open(os.path.join(BASE, 'E13_先导v14_最终真值.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pair_id', 'block', 'stratum', 'subtype', 'prev_consensus', 'prev_agree',
                    'v14_I', 'v14_J', 'v14_consensus', 'agree', 'note_I', 'note_J'])
        for p in pids:
            w.writerow([p, idx[p]['block'], idx[p]['stratum'], idx[p]['subtype'],
                        idx[p]['prev_consensus'], idx[p]['prev_agree'],
                        I[p]['label_relation'], J[p]['label_relation'], consensus[p],
                        int(I[p]['label_relation'] == J[p]['label_relation']),
                        (I[p].get('label_notes') or '')[:150],
                        (J[p].get('label_notes') or '')[:150]])
    print('[save] E13_先导v14_最终真值.csv')

    res = {
        'n': len(pids),
        'guide': 'v1.4-candidate-frozen',
        'overall': dict(allst, kappa_ci95=[lo, hi]),
        'per_block': per,
        'target_convergence': {'n_prev_disagree': len(conv), 'n_converged': n_conv, 'detail': conv},
        'control_integrity': {'n': len(fam), 'still_agree': n_sa, 'same_label': n_sl,
                              'broken': broke, 'A_block_label_drift': driftA},
        'yield_fresh_v14': y,
        'yield_fresh_v13_reference': {'S6_all': 0.111, 'strict': 0.176, 'polarity': 0.091, 'vague': 0.0},
        'yield_all_s6_in_76_reference': y76,
        'ruling_cases': cas,
        'system_v2_vs_consensus_v14': m,
        'caveat': ('通道同源（同一模型）⇒ 误差相关 ⇒ κ 偏乐观；合法用法是否决性检验。'
                   'A/B 块为 in-sample（上一轮判据在同一批样本上迭代过），'
                   'C 块为样本外（本轮首次使用）⇒ 产量只以 C 块为准。'),
    }
    with open(os.path.join(BASE, 'E13_先导v14_分析.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('[save] E13_先导v14_分析.json')
    return res


if __name__ == '__main__':
    main()
