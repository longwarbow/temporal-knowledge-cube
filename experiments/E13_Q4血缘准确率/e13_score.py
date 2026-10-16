# -*- coding: utf-8 -*-
"""
E13 — $Q_4$ 血缘准确率：**标注结果评分器**

用法：
    python e13_score.py                 # 读 E13_标注集.csv，报告标注进度与可得指标
    python e13_score.py <filled.csv>    # 指定填好的标注文件

计算（按可得性自动降级）：
  1) 标注进度：各层已填 / 总数
  2) 标签分布：按层 × 标签
  3) 标注员间一致性：Cohen's kappa（用 label_relation vs label_relation_2 的重叠行）
  4) 系统准确率：**仅在 `system_pred` 已填时**计算 per-class P/R/F1 + 混淆矩阵
输出：E13_评分.json（并打印人类可读摘要）
"""
import os
import sys
import csv
import json
import collections

BASE = os.path.dirname(os.path.abspath(__file__))
LABELS = ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED']


def cohen_kappa(a, b, labels):
    """a, b = 两个标注序列（同一批项目）。返回 kappa 与混淆矩阵。"""
    n = len(a)
    if n == 0:
        return None, None
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    M = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        if x in idx and y in idx:
            M[idx[x]][idx[y]] += 1
    tot = sum(sum(r) for r in M)
    if tot == 0:
        return None, M
    po = sum(M[i][i] for i in range(k)) / tot
    ra = [sum(M[i]) / tot for i in range(k)]
    cb = [sum(M[i][j] for i in range(k)) / tot for j in range(k)]
    pe = sum(ra[i] * cb[i] for i in range(k))
    kap = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return kap, M


def prf(pred, true, labels):
    out = {}
    for L in labels:
        tp = sum(1 for p, t in zip(pred, true) if p == L and t == L)
        fp = sum(1 for p, t in zip(pred, true) if p == L and t != L)
        fn = sum(1 for p, t in zip(pred, true) if p != L and t == L)
        P = tp / (tp + fp) if (tp + fp) else None
        R = tp / (tp + fn) if (tp + fn) else None
        F = (2 * P * R / (P + R)) if (P and R) else None
        out[L] = {'tp': tp, 'fp': fp, 'fn': fn,
                  'precision': round(P, 4) if P is not None else None,
                  'recall': round(R, 4) if R is not None else None,
                  'f1': round(F, 4) if F is not None else None}
    acc = sum(1 for p, t in zip(pred, true) if p == t) / len(pred) if pred else None
    macro_f1 = None
    fs = [out[L]['f1'] for L in labels if out[L]['f1'] is not None]
    if fs:
        macro_f1 = round(sum(fs) / len(fs), 4)
    return {'per_class': out, 'accuracy': round(acc, 4) if acc is not None else None,
            'macro_f1': macro_f1}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'E13_标注集.csv')
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    print('读入 %d 行：%s' % (len(rows), path))

    R = {'file': os.path.basename(path), 'n_rows': len(rows)}

    # 1) 进度
    filled = [r for r in rows if (r.get('label_relation') or '').strip()]
    R['progress'] = {
        'total': len(rows), 'labelled': len(filled),
        'pct': round(100.0 * len(filled) / len(rows), 1) if rows else 0.0,
        'by_stratum': dict(sorted(collections.Counter(
            r['stratum'] for r in filled).items())),
    }
    print('标注进度：%d/%d (%.1f%%)' % (len(filled), len(rows), R['progress']['pct']))

    # 2) 标签分布
    dist = collections.defaultdict(collections.Counter)
    for r in filled:
        dist[r['stratum']][r['label_relation'].strip()] += 1
    R['label_distribution'] = {k: dict(v) for k, v in dist.items()}
    for k, v in R['label_distribution'].items():
        print('  %-34s %s' % (k, v))

    # 3) kappa（重叠行：两个标注员都填了）
    ov = [(r['label_relation'].strip(), r['label_relation_2'].strip())
          for r in rows if (r.get('label_relation') or '').strip()
          and (r.get('label_relation_2') or '').strip()]
    if ov:
        kap, M = cohen_kappa([a for a, _ in ov], [b for _, b in ov], LABELS)
        R['inter_annotator'] = {'n_overlap': len(ov),
                                'cohen_kappa': round(kap, 4) if kap is not None else None,
                                'matrix': M, 'labels': LABELS}
        print('标注员间一致性：n=%d  Cohen kappa=%.3f' % (len(ov), kap or 0.0))
    else:
        R['inter_annotator'] = {'n_overlap': 0, 'cohen_kappa': None,
                                'note': '尚无重叠标注（需 label_relation_2）'}

    # 4) 系统准确率（仅当 system_pred 已填）
    sp = [r for r in filled if (r.get('system_pred') or '').strip()]
    if sp:
        pred = [r['system_pred'].strip() for r in sp]
        true = [r['label_relation'].strip() for r in sp]
        R['system_scores'] = prf(pred, true, LABELS)
        R['system_scores']['n'] = len(sp)
        print('系统评分（n=%d）：Acc=%.3f  macro-F1=%s' %
              (len(sp), R['system_scores']['accuracy'] or 0.0,
               R['system_scores']['macro_f1']))
        for L in LABELS:
            d = R['system_scores']['per_class'][L]
            print('  %-10s P=%s R=%s F1=%s (tp=%d fp=%d fn=%d)' %
                  (L, d['precision'], d['recall'], d['f1'], d['tp'], d['fp'], d['fn']))
    else:
        R['system_scores'] = {'note': 'system_pred 尚未填写——系统产出后重跑本脚本'}

    with open(os.path.join(BASE, 'E13_评分.json'), 'w', encoding='utf-8') as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    print('[save] E13_评分.json')


if __name__ == '__main__':
    main()
