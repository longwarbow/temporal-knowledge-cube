# -*- coding: utf-8 -*-
"""
E13 — $Q_4$ 血缘准确率：**声明图构造器 v2**（加入量槽规则）

与 v1（`e13_claim_graph.py`）的差别
-----------------------------------
v1 的规则 5「数值互斥」用的是**全文数字集合**的交集空 —— 实测精度差
（"by 9 ET" / "In 2009" 之类会被当成计数）。v2 改用 **量槽**（`e13_slot_extract.py`），
即 (量纲 head-class, 取值) 的结构化匹配，并新增一条**量槽+极性相反**规则。

判定梯 v2（自上而下，首个命中即出）
-----------------------------------
| # | 条件 | 预测 |
|---|---|---|
| 1 | RT/引用痕迹 且 词包含度 ≥ 0.50 | DERIVE |
| 2 | 字符 3-gram Jaccard ≥ 0.85 | DERIVE |
| 3 | 词包含度 ≥ 0.85 且 极性同向 | DERIVE |
| 4 | 极性相反 且 词包含度 ≥ 0.25（S+ 结构边可用 → 0.15） | CONFLICT |
| 5 | **量槽互斥（同量纲、双侧纯数值、值互斥）且 词包含度 ≥ 0.20** | CONFLICT |
| 6 | ~~量槽互斥（含模糊量词）~~ ⇒ **不做 CONFLICT**（见下注） | —（落至 8/9/10） |
| 7 | **同量槽 且 极性相反** | CONFLICT |
| 8 | 词包含度 ≥ 0.60 且 极性同向 | SUPPORT |
| 9 | 词包含度 ≥ 0.40 且 词 Jaccard ≥ 0.25 且 极性同向（S+ → 0.30） | SUPPORT |
| 10 | 以上皆不中 | UNRELATED |

> **极性与 RT 沿用 v1 定义**（`e13_claim_graph.py`），使 v1↔v2 的差异**只有"量槽"一个变量**。

产出
----
  E13_系统预测_v2.csv   —— pair_id / stratum / 特征 / 量槽 / system_pred(_content_only)
  e13_claim_graph_v2.json —— 规则表、命中分布、S+/S− 差异、v1↔v2 对照
"""
import os
import re
import csv
import json
import html
import collections

import e13_slot_extract as SLOT
import e13_claim_graph as V1          # 复用 v1 的极性/RT 定义 ⇒ 隔离"只换量槽规则"这一个变量

BASE = os.path.dirname(os.path.abspath(__file__))

P = {
    'rt_containment': 0.50,
    'char3_jaccard': 0.85,
    'contain_derive': 0.85,
    'contain_conflict': 0.25,
    'contain_conflict_struct': 0.15,
    'contain_slot_conflict': 0.20,     # 规则 5（量槽互斥）
    'contain_support_hi': 0.60,
    'contain_support_lo': 0.40,
    'jaccard_support_lo': 0.25,
    'contain_support_lo_struct': 0.30,
}

URL_RE = re.compile(r'http\S+|www\.\S+')
MENT_RE = re.compile(r'@[A-Za-z0-9_]+')
RT_RE = re.compile(r'^\s*rt\b[:\s]', re.I)
NONWORD_RE = re.compile(r'[^a-z0-9\s]')

STOP = set("""a an the and or but if then than so because as of to in on at by for with from into over under
this that these those it its it's is are was were be been being am do does did done doing have has had having
i you he she we they me him her us them my your his their our mine yours theirs
will would shall should can could may might must
there here what which who whom whose when where why how
all any both each few more most other some such no nor not only own same too very
just now also get got go going went make makes made say says said rt via amp
u ur im dont cant wont isnt arent wasnt didnt doesnt
""".split())


def norm_tokens(t):
    t = html.unescape(t or '')
    t = URL_RE.sub(' ', t)
    t = MENT_RE.sub(' ', t)
    t = RT_RE.sub(' ', t)
    t = t.replace('#', ' ')
    t = NONWORD_RE.sub(' ', t.lower())
    return [w for w in t.split() if w and w not in STOP and len(w) > 1]


def char3(t):
    t = html.unescape(t or '')
    t = URL_RE.sub('', t)
    t = MENT_RE.sub('', t)
    t = re.sub(r'[^a-z0-9]', '', t.lower())
    return set(t[i:i + 3] for i in range(max(0, len(t) - 2)))


def jacc(a, b):
    if not a or not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def containment(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def is_rt(raw):
    t = html.unescape(raw or '')
    return bool(re.search(r'\brt\b\s*@', t, re.I)) or bool(re.search(r'\bvia\s+@', t, re.I))


SLOT_CLASSES = ['DEAD', 'INJURED', 'HOSTAGE', 'SUSPECT', 'AIRPLANE', 'BUILDING']


def features(p, q):
    tp, tq = norm_tokens(p), norm_tokens(q)
    sp, sq = set(tp), set(tq)
    # ★ 极性与 RT 沿用 v1 的定义，**只让"量槽"这一个变量变化**（干净对照）
    pol_p, neg_p = V1.polarity(p)
    pol_q, neg_q = V1.polarity(q)
    qc = SLOT.quantity_conflict(p, q)          # (cls, vals_p, vals_q, strict) | None
    # 量槽冲突限制在"实体计数"类（与 S6 同口径）
    if qc and qc[0] not in SLOT_CLASSES:
        qc = None
    slot_cls = slot_kind = ''
    if qc:
        slot_cls = qc[0]
        slot_kind = 'num_exclusive' if qc[3] else 'vague_exclusive'
    elif pol_p != pol_q:
        # 规则 7：同量槽 + 极性相反（用 v1 的极性定义判定"相反"）
        ms = [m for m in SLOT.match_slots(p, q) if m.cls in SLOT_CLASSES]
        if ms:
            m = ms[0]
            slot_cls, slot_kind = m.cls, 'polarity'
            qc = (m.cls, sorted(m.vals_p), sorted(m.vals_q), False)
    return {
        'j_tok': round(jacc(sp, sq), 4),
        'contain': round(containment(sp, sq), 4),
        'char3_j': round(jacc(char3(p), char3(q)), 4),
        'pol_p': pol_p, 'pol_q': pol_q, 'neg_p': neg_p, 'neg_q': neg_q,
        'pol_opposite': int(pol_p != pol_q),
        'rt_p': int(is_rt(p)), 'rt_q': int(is_rt(q)),
        'len_p': len(tp), 'len_q': len(tq),
        'slot_cls': slot_cls,
        'slot_vals_p': ','.join(qc[1]) if qc else '',
        'slot_vals_q': ','.join(qc[2]) if qc else '',
        'slot_kind': slot_kind,
    }


def predict(f, struct_edge):
    rt = f['rt_p'] or f['rt_q']
    if rt and f['contain'] >= P['rt_containment']:
        return 'DERIVE', 1
    if f['char3_j'] >= P['char3_jaccard']:
        return 'DERIVE', 2
    if f['contain'] >= P['contain_derive'] and not f['pol_opposite']:
        return 'DERIVE', 3
    thr4 = P['contain_conflict_struct'] if struct_edge else P['contain_conflict']
    if f['pol_opposite'] and f['contain'] >= thr4:
        return 'CONFLICT', 4
    thr5 = P['contain_slot_conflict']
    if f['slot_kind'] == 'num_exclusive' and f['contain'] >= thr5:
        return 'CONFLICT', 5
    # ★ **模糊量槽不做 CONFLICT**（v2 的一处实测修正）：
    #   "multiple suspects" vs "3 suspects" 在冻结判据 v1.3 的 **T5「量的兼容性」** 下
    #   判为 **SUPPORT**（"多名"是"3 名"的粗化，可同时为真）。先导标的 E13-0105 正是
    #   这条，也正是当初让判据从 v1.2 振荡到 v1.3 的那条边界案例。
    #   ⇒ 模糊量槽只作为 S6 的**负对照子型**保留，不产生 CONFLICT 预测。
    if f['slot_kind'] == 'polarity':
        return 'CONFLICT', 7
    if f['contain'] >= P['contain_support_hi'] and not f['pol_opposite']:
        return 'SUPPORT', 8
    thr9 = P['contain_support_lo_struct'] if struct_edge else P['contain_support_lo']
    if (f['contain'] >= thr9 and f['j_tok'] >= P['jaccard_support_lo']
            and not f['pol_opposite']):
        return 'SUPPORT', 9
    return 'UNRELATED', 10


LAB = ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED']


def _metrics(y_true, y_pred):
    """per-class P/R/F1（不报 Accuracy，承 E6 教训）。"""
    res = {}
    for L in LAB:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == L and p == L)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != L and p == L)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == L and p != L)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        res[L] = {'n_true': sum(1 for t in y_true if t == L),
                  'P': round(prec, 4), 'R': round(rec, 4), 'F1': round(f1, 4)}
    res['macro_F1'] = round(sum(res[L]['F1'] for L in LAB) / 4, 4)
    cm = {t: collections.Counter() for t in LAB}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    res['confusion'] = {k: dict(v) for k, v in cm.items()}
    return res


def main():
    src = os.path.join(BASE, 'E13_标注集_v2.csv')
    if not os.path.exists(src):
        src = os.path.join(BASE, 'E13_标注集.csv')
    rows = list(csv.DictReader(open(src, encoding='utf-8-sig')))
    print('读入 %d 对（%s）' % (len(rows), os.path.basename(src)))

    out = []
    for r in rows:
        struct_edge = (r.get('system_prior') or '').startswith('structural')
        f = features(r['p_text'], r['q_text'])
        pred_s, rule_s = predict(f, struct_edge)
        pred_c, rule_c = predict(f, False)
        out.append(dict(r, **f, struct_edge=int(struct_edge),
                        system_pred=pred_s, system_pred_rule=rule_s,
                        system_pred_content_only=pred_c,
                        system_pred_content_only_rule=rule_c))

    cols = ['pair_id', 'stratum', 'subtype', 'event', 'struct_edge',
            'j_tok', 'contain', 'char3_j', 'pol_p', 'pol_q', 'pol_opposite',
            'rt_p', 'rt_q', 'len_p', 'len_q',
            'slot_cls', 'slot_vals_p', 'slot_vals_q', 'slot_kind',
            'system_pred', 'system_pred_rule',
            'system_pred_content_only', 'system_pred_content_only_rule',
            'p_text', 'q_text']
    with open(os.path.join(BASE, 'E13_系统预测_v2.csv'), 'w', encoding='utf-8-sig', newline='') as f_:
        w = csv.DictWriter(f_, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(out)
    print('[save] E13_系统预测_v2.csv')

    # ---------------- 统计 ----------------
    by_stratum = collections.defaultdict(collections.Counter)
    rules = collections.Counter()
    for r in out:
        by_stratum[r['stratum']][r['system_pred']] += 1
        rules['rule%d' % r['system_pred_rule']] += 1
    n_diff = sum(1 for r in out if r['system_pred'] != r['system_pred_content_only'])

    # 仅 S1–S4（原 388 行）上的分布 —— 与 v1 的 388 行结果**严格可比**
    base = [r for r in out if r['stratum'].startswith(('S1_', 'S2_', 'S3_', 'S4_'))]
    dist_388 = dict(collections.Counter(r['system_pred'] for r in base))
    v1_dist_388 = None
    p_v1 = os.path.join(BASE, 'e13_claim_graph.json')
    if os.path.exists(p_v1):
        try:
            raw = json.load(open(p_v1, encoding='utf-8')).get('pred_distribution') or {}
            # v1 的 pred_distribution 是 {标签: {规则: 计数}}，这里压平成标签计数
            v1_dist_388 = {k: (sum(v.values()) if isinstance(v, dict) else int(v))
                           for k, v in raw.items()}
        except Exception:
            pass

    # v1 ↔ v2 对照（若存在 v1 预测）
    v1 = {}
    p_v1 = os.path.join(BASE, 'E13_系统预测.csv')
    if os.path.exists(p_v1):
        v1 = {r['pair_id']: r['system_pred'] for r in csv.DictReader(open(p_v1, encoding='utf-8-sig'))}
    changed = [r['pair_id'] for r in out if r['pair_id'] in v1 and v1[r['pair_id']] != r['system_pred']]

    # 对先导真值（S1–S4 的 50 条）评 v2，看是否回归
    truth = {}
    p_t = os.path.join(BASE, 'E13_先导50条_最终真值.csv')
    if os.path.exists(p_t):
        truth = {r['pair_id']: r['consensus_label']
                 for r in csv.DictReader(open(p_t, encoding='utf-8-sig'))}

    def eval_on(truth_map, pred_key='system_pred'):
        ids = [r['pair_id'] for r in out if r['pair_id'] in truth_map]
        if not ids:
            return None
        by = {r['pair_id']: r for r in out}
        yt = [truth_map[i] for i in ids]
        yp = [by[i][pred_key] for i in ids]
        m = _metrics(yt, yp)
        m['n'] = len(ids)
        return m

    res = {
        'n_pairs': len(out),
        'source': os.path.basename(src),
        'params': P, 'slot_classes': SLOT_CLASSES,
        'rule_hits': dict(rules),
        'by_stratum': {k: dict(v) for k, v in by_stratum.items()},
        'S_plus_vs_S_minus': {'n_differ': n_diff,
                              'pct_differ': round(100.0 * n_diff / len(out), 2)},
        'v1_vs_v2': {'n_changed': len(changed), 'changed_pair_ids': changed[:40],
                     'v2_distribution_on_388': dist_388,
                     'v1_distribution_on_388': v1_dist_388,
                     'note': ('两者都在原 S1–S4 的 388 行上计算 ⇒ 严格可比；'
                              '唯一变量是"全文数字交集 → 量槽"（极性/RT 定义已复用 v1）')},
        'pilot_truth_eval_v2': eval_on(truth),
        'pilot_truth_eval_S_only': eval_on(truth, 'system_pred_content_only'),
        'pred_distribution': dict(collections.Counter(r['system_pred'] for r in out)),
        'note': ('v2 在 v1 判定梯上把"全文数字交集"换成**量槽**，并新增「同量槽+极性相反」。'
                 'S+ 仍只放宽规则 4/8 的阈值。'),
    }
    with open(os.path.join(BASE, 'e13_claim_graph_v2.json'), 'w', encoding='utf-8') as f_:
        json.dump(res, f_, ensure_ascii=False, indent=2)
    print('[save] e13_claim_graph_v2.json')

    print('\n规则命中：%s' % dict(rules))
    for s in sorted(by_stratum):
        print('  %-34s %s' % (s, dict(by_stratum[s])))
    print('S+ vs S− 不同：%d 对（%.2f%%）' % (n_diff, 100.0 * n_diff / len(out)))
    print('v1↔v2 预测变化：%d 对' % len(changed))
    print('[仅原 388 行] v2 = %s' % dist_388)
    print('[仅原 388 行] v1 = %s' % v1_dist_388)
    m = res['pilot_truth_eval_v2']
    if m:
        print('\n[对先导真值 n=%d] macro-F1=%.4f' % (m['n'], m['macro_F1']))
        for L in LAB:
            print('  %-10s n_true=%-3d P=%.3f R=%.3f F1=%.3f'
                  % (L, m[L]['n_true'], m[L]['P'], m[L]['R'], m[L]['F1']))
    return res


if __name__ == '__main__':
    main()
