# -*- coding: utf-8 -*-
"""
E13 — $Q_4$ 血缘准确率：**先导标注分析器**

读入：
  E13_先导50条_标注员A.csv / _标注员B.csv            （两通道独立标注）
  E13_先导50条_仲裁.csv                              （可选；分歧项第三通道）
  E13_先导50条_索引.json / E13_先导50条_盲版.csv
  E13_系统预测.csv

输出：E13_先导分析.json（并打印摘要）
  · 观察一致性 po、Cohen's κ（含 95% CI）、**Gwet's AC1**（抗"κ 悖论"）
  · A×B 混淆矩阵、逐类一致率、分歧清单（含原文与特征）
  · 共识标签下的标签分布
  · system_pred（S+ 结构版 / S- 纯内容版 / 全 UNRELATED 空模型）的 per-class P/R/F1
  · 诊断：veracity 相反是否预测 CONFLICT（用共识真值反查）
"""
import os
import csv
import json
import math
import collections

BASE = os.path.dirname(os.path.abspath(__file__))
LABELS = ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED']


def read_csv(p):
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p, encoding='utf-8-sig')))


def kappa(a, b, labels):
    n = len(a)
    if n == 0:
        return None
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    M = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        M[idx[x]][idx[y]] += 1
    po = sum(M[i][i] for i in range(k)) / n
    ra = [sum(M[i]) / n for i in range(k)]
    cb = [sum(M[i][j] for i in range(k)) / n for j in range(k)]
    pe = sum(ra[i] * cb[i] for i in range(k))
    kap = (po - pe) / (1 - pe) if pe < 1 else 0.0
    # 简单近似标准误（Fleiss 形式）
    se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2)) if (1 - pe) > 0 else 0.0
    return {'po': round(po, 4), 'pe': round(pe, 4), 'kappa': round(kap, 4),
            'se': round(se, 4),
            'ci95': [round(kap - 1.96 * se, 4), round(kap + 1.96 * se, 4)],
            'matrix': M}


def gwet_ac1(a, b, labels):
    """Gwet's AC1：类别严重偏斜时比 κ 稳健（κ 悖论）。"""
    n = len(a)
    if n == 0:
        return None
    q = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    M = [[0] * q for _ in range(q)]
    for x, y in zip(a, b):
        M[idx[x]][idx[y]] += 1
    po = sum(M[i][i] for i in range(q)) / n
    pi = [(sum(M[i]) + sum(M[j][i] for j in range(q))) / (2.0 * n) for i in range(q)]
    pe = sum(p * (1 - p) for p in pi) / (q - 1)
    ac1 = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return {'po': round(po, 4), 'pe': round(pe, 4), 'ac1': round(ac1, 4)}


def prf(pred, true, labels):
    out = {}
    for L in labels:
        tp = sum(1 for p, t in zip(pred, true) if p == L and t == L)
        fp = sum(1 for p, t in zip(pred, true) if p == L and t != L)
        fn = sum(1 for p, t in zip(pred, true) if p != L and t == L)
        sup = tp + fn
        P = tp / (tp + fp) if (tp + fp) else None
        R = tp / (tp + fn) if (tp + fn) else None
        F = (2 * P * R / (P + R)) if (P is not None and R is not None and (P + R) > 0) else (
            0.0 if (sup > 0) else None)
        out[L] = {'support': sup, 'tp': tp, 'fp': fp, 'fn': fn,
                  'precision': round(P, 4) if P is not None else None,
                  'recall': round(R, 4) if R is not None else None,
                  'f1': round(F, 4) if F is not None else None}
    acc = sum(1 for p, t in zip(pred, true) if p == t) / len(pred) if pred else None
    fs = [v['f1'] for v in out.values() if v['f1'] is not None]
    return {'per_class': out,
            'accuracy_NOT_a_result': round(acc, 4) if acc is not None else None,
            'macro_f1': round(sum(fs) / len(fs), 4) if fs else None,
            'n': len(pred)}


def main():
    idx = json.load(open(os.path.join(BASE, 'E13_先导50条_索引.json'), encoding='utf-8'))
    smap = idx['map']
    blind = {r['pair_id']: r for r in read_csv(os.path.join(BASE, 'E13_先导50条_盲版.csv'))}
    A = {r['pair_id']: r for r in read_csv(os.path.join(BASE, 'E13_先导50条_标注员A.csv'))}
    B = {r['pair_id']: r for r in read_csv(os.path.join(BASE, 'E13_先导50条_标注员B.csv'))}
    ARB = {r['pair_id']: r for r in read_csv(os.path.join(BASE, 'E13_先导50条_仲裁.csv'))}
    pred = {r['pair_id']: r for r in read_csv(os.path.join(BASE, 'E13_系统预测.csv'))}

    ids = sorted(blind)
    print('标注员 A %d 行 ｜ B %d 行 ｜ 仲裁 %d 行 ｜ 待标注 %d 行'
          % (len(A), len(B), len(ARB), len(ids)))

    la = [A[i]['label_relation'].strip() for i in ids]
    lb = [B[i]['label_relation'].strip() for i in ids]
    bad = [i for i, x, y in zip(ids, la, lb) if x not in LABELS or y not in LABELS]
    if bad:
        print('!! 非法标签：%s' % bad)

    kap = kappa(la, lb, LABELS)
    ac1 = gwet_ac1(la, lb, LABELS)
    print('\n观察一致 po=%.3f  ｜ Cohen κ=%.3f (95%%CI %.3f–%.3f)  ｜ Gwet AC1=%.3f'
          % (kap['po'], kap['kappa'], kap['ci95'][0], kap['ci95'][1], ac1['ac1']))
    print('  A×B 混淆矩阵（行=A，列=B，序 %s）：' % LABELS)
    for i, L in enumerate(LABELS):
        print('   %-10s %s' % (L, kap['matrix'][i]))

    # ---------- 共识标签 ----------
    dis = [i for i, x, y in zip(ids, la, lb) if x != y]
    src = {}
    consensus = {}
    for i, x, y in zip(ids, la, lb):
        if x == y:
            consensus[i] = x
            src[i] = 'agree'
        else:
            z = (ARB.get(i, {}).get('label_relation') or '').strip()
            if z in LABELS:
                consensus[i] = z
                src[i] = 'arbitration'
            else:
                consensus[i] = None
                src[i] = 'unresolved'

    print('\n分歧 %d/%d (%.1f%%)  ｜ 已仲裁 %d ｜ 未解决 %d'
          % (len(dis), len(ids), 100.0 * len(dis) / len(ids),
             sum(1 for i in dis if src[i] == 'arbitration'),
             sum(1 for i in dis if src[i] == 'unresolved')))

    # 逐类一致（A 的类 → B 的类）
    per_class_agree = {}
    for L in LABELS:
        sub = [(x, y) for x, y in zip(la, lb) if x == L]
        per_class_agree[L] = {'A_n': len(sub),
                              'agree': sum(1 for x, y in sub if y == L),
                              'agree_rate': round(sum(1 for x, y in sub if y == L) / len(sub), 4)
                              if sub else None}

    # ---------- 分歧清单 ----------
    dis_rows = []
    for i in dis:
        r = blind[i]
        f = pred.get(i, {})
        dis_rows.append({
            'pair_id': i, 'stratum': smap[i], 'event': r['event'],
            'A': A[i]['label_relation'].strip(), 'B': B[i]['label_relation'].strip(),
            'arbitration': (ARB.get(i, {}).get('label_relation') or '').strip() or None,
            'A_conf': A[i].get('label_confidence'), 'B_conf': B[i].get('label_confidence'),
            'A_note': (A[i].get('label_notes') or '').strip(),
            'B_note': (B[i].get('label_notes') or '').strip(),
            'contain': f.get('contain'), 'j_tok': f.get('j_tok'),
            'pol_opposite': f.get('pol_opposite'), 'char3_j': f.get('char3_j'),
            'system_pred': f.get('system_pred'),
            'p_text': r['p_text'], 'q_text': r['q_text'],
        })

    # ---------- 标签分布 ----------
    dist = {'A': dict(collections.Counter(la)), 'B': dict(collections.Counter(lb))}
    cons = [consensus[i] for i in ids if consensus[i]]
    dist['consensus'] = dict(collections.Counter(cons))
    dist['consensus_by_stratum'] = {}
    for s in sorted(set(smap.values())):
        sub = [consensus[i] for i in ids if smap[i] == s and consensus[i]]
        dist['consensus_by_stratum'][s] = dict(collections.Counter(sub))

    # ---------- 系统评分（对共识真值） ----------
    scored_ids = [i for i in ids if consensus[i]]
    true = [consensus[i] for i in scored_ids]
    sysres = {}
    for name, col in [('S_plus_structural', 'system_pred'),
                      ('S_minus_content_only', 'system_pred_content_only')]:
        p = [pred[i][col] for i in scored_ids]
        sysres[name] = prf(p, true, LABELS)
    allun = ['UNRELATED'] * len(scored_ids)
    sysres['null_all_UNRELATED'] = prf(allun, true, LABELS)

    print('\n系统评分（对共识标签 n=%d）：' % len(scored_ids))
    for k in ['S_plus_structural', 'S_minus_content_only', 'null_all_UNRELATED']:
        d = sysres[k]
        print('  %-24s macro-F1=%s  Acc(仅参考)=%s'
              % (k, d['macro_f1'], d['accuracy_NOT_a_result']))
        for L in LABELS:
            v = d['per_class'][L]
            print('      %-10s P=%s R=%s F1=%s (sup=%d)' %
                  (L, v['precision'], v['recall'], v['f1'], v['support']))

    # ---------- 诊断 ----------
    diag = {}
    # 1) veracity 相反 是否预测 CONFLICT
    s3 = [i for i in ids if smap[i] == 'S3_cross_thread_opposite_veracity' and consensus[i]]
    s4 = [i for i in ids if smap[i] == 'S4_cross_thread_random' and consensus[i]]
    def c_rate(sub):
        return round(sum(1 for i in sub if consensus[i] == 'CONFLICT') / len(sub), 4) if sub else None
    diag['P_CONFLICT_given_veracity_opposite'] = {'n': len(s3), 'rate': c_rate(s3)}
    diag['P_CONFLICT_given_random_same_event'] = {'n': len(s4), 'rate': c_rate(s4)}
    # 2) 各层共识标签分布
    diag['consensus_by_stratum'] = dist['consensus_by_stratum']
    # 3) 系统在 S+/S- 上是否给出不同预测
    n_diff = sum(1 for i in ids if pred[i]['system_pred'] != pred[i]['system_pred_content_only'])
    diag['S_plus_vs_S_minus_diff_on_pilot'] = n_diff
    # 4) 高置信子集上的 κ（只取两人都 ≥2 的行）
    sub = [(i, x, y) for i, x, y in zip(ids, la, lb)
           if (A[i].get('label_confidence') or '') in ('2', '3')
           and (B[i].get('label_confidence') or '') in ('2', '3')]
    if sub:
        k2 = kappa([x for _, x, _ in sub], [y for _, _, y in sub], LABELS)
        diag['kappa_high_conf_subset'] = {'n': len(sub), 'kappa': k2['kappa'], 'po': k2['po']}

    res = {
        'n_pilot': len(ids),
        'annotators': {'A': 'E13_先导50条_标注员A.csv', 'B': 'E13_先导50条_标注员B.csv',
                       'arbitration': 'E13_先导50条_仲裁.csv'},
        'agreement': {'cohen_kappa': kap, 'gwet_ac1': ac1,
                      'n_disagree': len(dis), 'disagree_pct': round(100.0 * len(dis) / len(ids), 2),
                      'per_class_agreement': per_class_agree},
        'disagreements': dis_rows,
        'label_distribution': dist,
        'consensus_source': dict(collections.Counter(src.values())),
        'system_scores_on_consensus': sysres,
        'diagnostics': diag,
        'gate': {'target_kappa': 0.70, 'kappa': kap['kappa'],
                 'pass': bool(kap['kappa'] >= 0.70),
                 'note': ('AI 双通道探针的 κ 只用于**否决性检验**（连 AI 都分不开 ⇒ 判据肯定坏了），'
                          '不能替代人类 κ；两者误差相关，探针 κ 偏乐观')},
    }
    with open(os.path.join(BASE, 'E13_先导分析.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('\n[save] E13_先导分析.json')
    return res


if __name__ == '__main__':
    main()
