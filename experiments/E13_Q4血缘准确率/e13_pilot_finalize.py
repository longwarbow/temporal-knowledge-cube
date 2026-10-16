# -*- coding: utf-8 -*-
"""
E13 — $Q_4$ 先导标注：**冻结与最终化**

把两轮标注放在一起做「判据修订迭代」对照：
  轮 1（判据 v1.0，指南 `E13_标注指南_标注员版.md`）      → 标注员A / 标注员B
  轮 2（判据 v1.3 冻结版，指南 `..._v1.3.md`）            → 冻结A / 冻结B
两轮用的是**同一批 50 条**（in-sample 校准，须在报告中如实声明）。

产出：
  · E13_先导50条_仲裁.csv     —— 轮 2 分歧项（待第三通道裁定）
  · E13_先导50条_最终真值.csv  —— 先导真值（共识标签 + 来源 + 置信度）
  · E13_先导分析_冻结.json     —— κ / AC1 / 逐类一致 / 系统评分 / 诊断
  · E13_判据修订迭代.json      —— 两轮对照表
"""
import os
import csv
import json
import math
import collections

BASE = os.path.dirname(os.path.abspath(__file__))
LABELS = ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED']


def rd(name):
    p = os.path.join(BASE, name)
    if not os.path.exists(p):
        return {}
    return {r['pair_id']: r for r in csv.DictReader(open(p, encoding='utf-8-sig'))}


def kappa(a, b, labels):
    n = len(a)
    idx = {l: i for i, l in enumerate(labels)}
    q = len(labels)
    M = [[0] * q for _ in range(q)]
    for x, y in zip(a, b):
        M[idx[x]][idx[y]] += 1
    po = sum(M[i][i] for i in range(q)) / n
    ra = [sum(M[i]) / n for i in range(q)]
    cb = [sum(M[i][j] for i in range(q)) / n for j in range(q)]
    pe = sum(ra[i] * cb[i] for i in range(q))
    kap = (po - pe) / (1 - pe) if pe < 1 else 1.0
    se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2)) if 0 < pe < 1 else 0.0
    pi = [(sum(M[i]) + sum(M[j][i] for j in range(q))) / (2.0 * n) for i in range(q)]
    pe_g = sum(p * (1 - p) for p in pi) / (q - 1)
    ac1 = (po - pe_g) / (1 - pe_g) if pe_g < 1 else 1.0
    return {'po': round(po, 4), 'pe': round(pe, 4), 'kappa': round(kap, 4),
            'se': round(se, 4), 'ci95': [round(kap - 1.96 * se, 4), round(kap + 1.96 * se, 4)],
            'gwet_ac1': round(ac1, 4), 'matrix': M, 'labels': labels}


def prf(pred, true, labels):
    out = {}
    for L in labels:
        tp = sum(1 for p, t in zip(pred, true) if p == L and t == L)
        fp = sum(1 for p, t in zip(pred, true) if p == L and t != L)
        fn = sum(1 for p, t in zip(pred, true) if p != L and t == L)
        sup = tp + fn
        P = tp / (tp + fp) if (tp + fp) else None
        R = tp / (tp + fn) if (tp + fn) else None
        F = (2 * P * R / (P + R)) if (P is not None and R is not None and P + R > 0) else (
            0.0 if sup else None)
        out[L] = {'support': sup, 'tp': tp, 'fp': fp, 'fn': fn,
                  'precision': round(P, 4) if P is not None else None,
                  'recall': round(R, 4) if R is not None else None,
                  'f1': round(F, 4) if F is not None else None}
    acc = sum(1 for p, t in zip(pred, true) if p == t) / len(pred) if pred else None
    fs = [v['f1'] for v in out.values() if v['f1'] is not None]
    return {'per_class': out, 'accuracy_NOT_a_result': round(acc, 4) if acc is not None else None,
            'macro_f1': round(sum(fs) / len(fs), 4) if fs else None, 'n': len(pred)}


def main():
    idx = json.load(open(os.path.join(BASE, 'E13_先导50条_索引.json'), encoding='utf-8'))
    smap = idx['map']
    blind = rd('E13_先导50条_盲版.csv')
    pred = rd('E13_系统预测.csv')
    ids = sorted(blind)

    A, B = rd('E13_先导50条_标注员A.csv'), rd('E13_先导50条_标注员B.csv')
    FA, FB = rd('E13_先导50条_冻结A.csv'), rd('E13_先导50条_冻结B.csv')

    def pair(a, b):
        la = [a[i]['label_relation'].strip() for i in ids]
        lb = [b[i]['label_relation'].strip() for i in ids]
        return la, lb

    la, lb = pair(A, B)
    fa, fb = pair(FA, FB)
    k1 = kappa(la, lb, LABELS)
    k2 = kappa(fa, fb, LABELS)

    d1 = [i for i, x, y in zip(ids, la, lb) if x != y]
    d2 = [i for i, x, y in zip(ids, fa, fb) if x != y]

    print('=== 判据修订迭代（同一批 50 条，in-sample）===')
    print('  轮 1  v1.0  po=%.3f  κ=%.3f (%.3f–%.3f)  AC1=%.3f  分歧 %d'
          % (k1['po'], k1['kappa'], k1['ci95'][0], k1['ci95'][1], k1['gwet_ac1'], len(d1)))
    print('  轮 2  v1.3  po=%.3f  κ=%.3f (%.3f–%.3f)  AC1=%.3f  分歧 %d'
          % (k2['po'], k2['kappa'], k2['ci95'][0], k2['ci95'][1], k2['gwet_ac1'], len(d2)))
    print('  轮 1 分歧项：%s' % d1)
    print('  轮 2 分歧项：%s' % d2)

    # ---------- 仲裁：轮 2 分歧项 ----------
    arb_path = os.path.join(BASE, 'E13_先导50条_仲裁.csv')
    ARB = rd('E13_先导50条_仲裁.csv')
    if d2 and not ARB:
        with open(arb_path, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['pair_id', 'label_relation',
                                              'label_confidence', 'label_notes'])
            w.writeheader()
            for i in d2:
                w.writerow({'pair_id': i, 'label_relation': '', 'label_confidence': '',
                            'label_notes': ''})
        print('[save] E13_先导50条_仲裁.csv（待第三通道填 %d 条）' % len(d2))

    consensus, src, conf = {}, {}, {}
    for i, x, y in zip(ids, fa, fb):
        if x == y:
            consensus[i] = x
            src[i] = 'agree_v1.3'
            conf[i] = min(int(FA[i].get('label_confidence') or 2),
                          int(FB[i].get('label_confidence') or 2))
        else:
            z = (ARB.get(i, {}).get('label_relation') or '').strip()
            if z in LABELS:
                consensus[i] = z
                src[i] = 'arbitration'
                conf[i] = int(ARB[i].get('label_confidence') or 2)
            else:
                consensus[i] = None
                src[i] = 'unresolved'
                conf[i] = None

    cons = [consensus[i] for i in ids if consensus[i]]
    print('\n共识：%d/50 完成 ｜ %s' % (len(cons), dict(collections.Counter(src.values()))))
    print('标签分布：%s' % dict(collections.Counter(cons)))

    # ---------- 系统评分（对轮 2 共识） ----------
    ok = [i for i in ids if consensus[i]]
    true = [consensus[i] for i in ok]
    sys_scores = {}
    for nm, col in [('S_plus_structural', 'system_pred'),
                    ('S_minus_content_only', 'system_pred_content_only')]:
        sys_scores[nm] = prf([pred[i][col] for i in ok], true, LABELS)
    sys_scores['null_all_UNRELATED'] = prf(['UNRELATED'] * len(ok), true, LABELS)
    print('\n系统评分（对轮 2 共识 n=%d）：' % len(ok))
    for k in ['S_plus_structural', 'S_minus_content_only', 'null_all_UNRELATED']:
        print('  %-24s macro-F1=%s' % (k, sys_scores[k]['macro_f1']))
        for L in LABELS:
            v = sys_scores[k]['per_class'][L]
            print('      %-10s P=%s R=%s F1=%s (sup=%d)' % (L, v['precision'], v['recall'],
                                                             v['f1'], v['support']))

    # ---------- 诊断 ----------
    diag = {}
    def rate(sub, lab):
        return round(sum(1 for i in sub if consensus[i] == lab) / len(sub), 4) if sub else None
    for s in sorted(set(smap.values())):
        sub = [i for i in ids if smap[i] == s and consensus[i]]
        diag[s] = {'n': len(sub), 'dist': dict(collections.Counter(consensus[i] for i in sub)),
                   'P_UNRELATED': rate(sub, 'UNRELATED'), 'P_SUPPORT': rate(sub, 'SUPPORT'),
                   'P_CONFLICT': rate(sub, 'CONFLICT'), 'P_DERIVE': rate(sub, 'DERIVE')}
    diag_note = {
        'P_CONFLICT_given_veracity_opposite': diag.get('S3_cross_thread_opposite_veracity', {}).get('P_CONFLICT'),
        'P_UNRELATED_given_veracity_opposite': diag.get('S3_cross_thread_opposite_veracity', {}).get('P_UNRELATED'),
        'P_UNRELATED_given_random': diag.get('S4_cross_thread_random', {}).get('P_UNRELATED'),
        'P_SUPPORT_given_high_sim': diag.get('S2_cross_thread_high_sim', {}).get('P_SUPPORT'),
        'P_DERIVE_given_high_sim': diag.get('S2_cross_thread_high_sim', {}).get('P_DERIVE'),
        'P_UNRELATED_given_same_thread_reply': diag.get('S1_same_thread_reply', {}).get('P_UNRELATED'),
    }

    # ---------- 落盘 ----------
    cols = ['pair_id', 'stratum', 'event', 'consensus_label', 'label_source',
            'confidence_min', 'frozenA', 'frozenB', 'round1A', 'round1B',
            'system_pred', 'system_pred_content_only', 'p_text', 'q_text']
    with open(os.path.join(BASE, 'E13_先导50条_最终真值.csv'), 'w',
              encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in ids:
            w.writerow({
                'pair_id': i, 'stratum': smap[i], 'event': blind[i]['event'],
                'consensus_label': consensus[i] or '', 'label_source': src[i],
                'confidence_min': conf[i] if conf[i] is not None else '',
                'frozenA': FA[i]['label_relation'].strip(),
                'frozenB': FB[i]['label_relation'].strip(),
                'round1A': A[i]['label_relation'].strip(),
                'round1B': B[i]['label_relation'].strip(),
                'system_pred': pred[i]['system_pred'],
                'system_pred_content_only': pred[i]['system_pred_content_only'],
                'p_text': blind[i]['p_text'], 'q_text': blind[i]['q_text']})
    print('[save] E13_先导50条_最终真值.csv')

    out = {
        'n_pilot': 50,
        'caveat': ('两轮用的是同一批 50 条 ⇒ 这是 **in-sample 判据校准**，'
                   '轮 2 的 κ 提升含"对样本过拟合"成分，不能当作泛化性能；'
                   '冻结判据的人类 κ 必须在**独立样本**上测。'),
        'round1_guide_v1.0': {'kappa': k1, 'n_disagree': len(d1), 'disagree_ids': d1},
        'round2_guide_v1.3_frozen': {'kappa': k2, 'n_disagree': len(d2), 'disagree_ids': d2},
        'gate': {'target': 0.70, 'round1': k1['kappa'], 'round2': k2['kappa'],
                 'pass_round2': bool(k2['kappa'] >= 0.70)},
        'consensus': {'n_resolved': len(cons), 'source': dict(collections.Counter(src.values())),
                      'distribution': dict(collections.Counter(cons))},
        'system_scores': sys_scores,
        'stratum_diagnostics': diag,
        'design_priors_vs_observed': diag_note,
        'conclusion_headline': ('结构回复边 ≠ 语义关系（S1 同线程回复 %s 判 UNRELATED）；'
                                'veracity 相反 ≠ CONFLICT（S3 的 CONFLICT 率 %s）；'
                                '高相似对 SUPPORT 率 %s、DERIVE 率 %s（同命题层，但多为独立同命题）。'
                                % (diag_note['P_UNRELATED_given_same_thread_reply'],
                                   diag_note['P_CONFLICT_given_veracity_opposite'],
                                   diag_note['P_SUPPORT_given_high_sim'],
                                   diag_note['P_DERIVE_given_high_sim'])),
    }
    with open(os.path.join(BASE, 'E13_先导分析_冻结.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(BASE, 'E13_判据修订迭代.json'), 'w', encoding='utf-8') as f:
        json.dump({'rounds': [
            {'guide': 'v1.0（首版）', 'kappa': k1['kappa'], 'po': k1['po'],
             'ac1': k1['gwet_ac1'], 'n_disagree': len(d1), 'disagree_ids': d1,
             'what_changed': '——'},
            {'guide': 'v1.3（冻结）', 'kappa': k2['kappa'], 'po': k2['po'],
             'ac1': k2['gwet_ac1'], 'n_disagree': len(d2), 'disagree_ids': d2,
             'what_changed': 'T1 命题性（含混合语力）、T2 不推断隐含、T3 同槽位、'
                             'T4 情形同一性、T5 量的兼容性、§2.1-补 细节优先、'
                             '**判定顺序**（消解规则互相覆盖）'}],
            'intermediate_rounds': [
                {'guide': 'v1.1（4 条分歧 + 6 条对照，10 条）',
                 'result': '4/4 靶向分歧收敛到 UNRELATED；5/6 对照稳定；'
                           '**1 条对照（E13-0181）被打散** ⇒ 促成 T4'},
                {'guide': 'v1.2（同 10 条）',
                 'result': '10/10 一致；但两名复审员各自独立指出 '
                           '**T3 与 T4 对"量是否属槽位键"自相矛盾** ⇒ 促成 v1.3'}],
            'caveat': '同一批 50 条 in-sample 校准；提升含过拟合成分。'},
        f, ensure_ascii=False, indent=2)
    print('[save] E13_先导分析_冻结.json / E13_判据修订迭代.json')


if __name__ == '__main__':
    main()
