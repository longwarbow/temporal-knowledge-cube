# -*- coding: utf-8 -*-
"""
E13 — $Q_4$ 血缘准确率：**声明图构造器 v3**（同步判据 v1.5 的 A3 / A5）

与 v2 的差别（**只加三个可机械化守卫，不动阈值、不动极性与 RT 之外的定义**）
------------------------------------------------------------------
| 守卫 | 依据 | 作用 |
|---|---|---|
| **A3-ⓠ 计量对象门** | 判据 v1.5 §2.6 ⓠ | 一侧出现 "N situations / attacks / incidents / locations" 时，该数是**起数/处数**，与人数类量槽**不可比** ⇒ **不作 CONFLICT** |
| **A3-ⓐ 子集放行** | 判据 v1.5 §2.6 ⓐ | 量值互斥但满足"总数 vs 部分"（① 部分标记且与**同一量纲**同类；② 下界标记落在**较大值**一侧；③ 较小值一侧是**具指子项** `the N`）⇒ **不作 CONFLICT**（落至 SUPPORT/UNRELATED 分支） |
| **A5 归因守卫** | 判据 v1.5 §2.7 A5 | `via @X` / `source: @X` / `据 @X` 是**归因**（新增事实细节），且 **q 有而 p 无** ⇒ **不得判 DERIVE**（降为 SUPPORT）；同时 `is_rt` 不再把 `via @` 当作引用痕迹 |

产物：E13_系统预测_v3.csv / e13_claim_graph_v3.json / E13_先导v15_合并真值.csv
"""
import os
import re
import csv
import json
import collections

import e13_claim_graph_v2 as V2
import e13_slot_extract as SLOT

BASE = os.path.dirname(os.path.abspath(__file__))
LAB = V2.LAB
P = dict(V2.P)

# ---------------- 三个守卫的词表 ----------------
EVENT_UNIT_RE = re.compile(
    r'\b\d[\d,]*\s+(?:\w+\s+){0,2}'
    r'(situations?|attacks?|incidents?|cases?|locations?|scenes?|places?|sieges?|'
    r'events?|blasts?|explosions?|raids?|bombings?)\b', re.I)
LOWER_BOUND_RE = re.compile(
    r'\b(at\s+least|up\s+to|more\s+than|over|nearly|almost|dozens|scores|hundreds|'
    r'multiple|several|many)\b', re.I)
PART_MARK_RE = re.compile(r'\b(these|those|among|of\s+them|of\s+the|其中)\b', re.I)
DEFINITE_N_RE = re.compile(r'\bthe\s+\d', re.I)
SOURCE_RE = re.compile(r'\b(via|source|credit|courtesy|per)\s*:?\s*@|据\s*@', re.I)


def _nums(vals):
    return set(int(v) for v in vals if str(v).isdigit())


def _near_class(text, pos, cls, span=70):
    """pos 附近 span 字符内是否出现该量纲的触发词（用于 A3-ⓐ 的"指同一总体"限定）。"""
    seg = text[max(0, pos - span): pos + span].lower()
    return any(w in seg for w in SLOT.CANON.get(cls, ()))


def a3_subset(p, q, cls, vals_p, vals_q):
    """A3-ⓐ：量值互斥但属于"总数 vs 部分" ⇒ True（放行，不作 CONFLICT）。"""
    npv, nqv = _nums(vals_p), _nums(vals_q)
    if not npv or not nqv:
        return False, ''
    big_is_p = max(npv) > max(nqv)
    big_text, small_text = (p, q) if big_is_p else (q, p)
    why = ''
    # ① 部分标记，且该标记**指向同一量纲**（同 总体）
    for text in (p, q):
        for m in PART_MARK_RE.finditer(text):
            if _near_class(text, m.start(), cls):
                return True, 'part_mark'
    # ② 下界/模糊总量标记落在**较大值**一侧
    if LOWER_BOUND_RE.search(big_text):
        return True, 'lower_bound_on_total'
    # ③ 较小值一侧是**具指子项**（`the N …`）
    if DEFINITE_N_RE.search(small_text):
        return True, 'definite_subitem'
    return False, why


def features(p, q):
    f = V2.features(p, q)
    # --- A5：来源标注（q 有而 p 无 ⇒ 新增归因）
    f['src_q_only'] = int(bool(SOURCE_RE.search(q)) and not bool(SOURCE_RE.search(p)))
    f['rt_v3'] = int(bool(re.search(r'\brt\b\s*@', p, re.I))) or int(bool(re.search(r'\brt\b\s*@', q, re.I)))
    # --- A3-ⓠ：计量对象门
    f['event_unit'] = int(bool(EVENT_UNIT_RE.search(p)) or bool(EVENT_UNIT_RE.search(q)))
    # --- A3-ⓐ：子集放行
    sub, why = False, ''
    if f['slot_kind'] == 'num_exclusive':
        sub, why = a3_subset(p, q, f['slot_cls'], f['slot_vals_p'].split(','), f['slot_vals_q'].split(','))
    f['a3_subset'] = int(sub)
    f['a3_subset_why'] = why
    return f


def predict(f, struct_edge):
    rt = f['rt_v3']
    d = None
    if rt and f['contain'] >= P['rt_containment']:
        d = 1
    elif f['char3_j'] >= P['char3_jaccard']:
        d = 2
    elif f['contain'] >= P['contain_derive'] and not f['pol_opposite']:
        d = 3
    # ★ A5 守卫：q 新增来源归因 ⇒ 不得判 DERIVE
    if d and f['src_q_only']:
        return 'SUPPORT', 30
    if d:
        return 'DERIVE', d
    thr4 = P['contain_conflict_struct'] if struct_edge else P['contain_conflict']
    if f['pol_opposite'] and f['contain'] >= thr4:
        return 'CONFLICT', 4
    # ★ A3-ⓠ：计量对象不同 ⇒ 量槽不作 CONFLICT
    if f['slot_kind'] == 'num_exclusive' and f['event_unit']:
        pass
    # ★ A3-ⓐ：总数 vs 部分 ⇒ 不作 CONFLICT
    elif f['slot_kind'] == 'num_exclusive' and not f['a3_subset'] and f['contain'] >= P['contain_slot_conflict']:
        return 'CONFLICT', 5
    if f['slot_kind'] == 'polarity':
        return 'CONFLICT', 7
    if f['contain'] >= P['contain_support_hi'] and not f['pol_opposite']:
        return 'SUPPORT', 8
    thr9 = P['contain_support_lo_struct'] if struct_edge else P['contain_support_lo']
    if (f['contain'] >= thr9 and f['j_tok'] >= P['jaccard_support_lo']
            and not f['pol_opposite']):
        return 'SUPPORT', 9
    return 'UNRELATED', 10


def build_merged_truth():
    """v1.5 合并真值：v1.4 的 76 条，其中 v1.4 的 6 条分歧用 v1.5 复审结果替换。"""
    def load(path):
        out = []
        with open(path, encoding='utf-8-sig', newline='') as fh:
            for r in csv.DictReader(fh):
                out.append({(k or '').lstrip('\ufeff'): v for k, v in r.items()})
        return out
    v14 = load(os.path.join(BASE, 'E13_先导v14_最终真值.csv'))
    v15 = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_先导v15_最终真值.csv'))}
    rows = []
    for r in v14:
        lab = r['v14_consensus']
        src = 'v14'
        if r['agree'] == '0' and r['pair_id'] in v15:
            v = v15[r['pair_id']]
            lab = v['v15_K'] if v['v15_K'] == v['v15_L'] else ''
            src = 'v15'
        rows.append({'pair_id': r['pair_id'], 'block': r['block'], 'stratum': r['stratum'],
                     'subtype': r['subtype'], 'truth': lab, 'src': src})
    with open(os.path.join(BASE, 'E13_先导v15_合并真值.csv'), 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['pair_id', 'block', 'stratum', 'subtype', 'truth', 'src'])
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    src = os.path.join(BASE, 'E13_标注集_v2.csv')
    rows = list(csv.DictReader(open(src, encoding='utf-8-sig')))
    print('读入 %d 对（%s）' % (len(rows), os.path.basename(src)))

    out = []
    for r in rows:
        struct_edge = (r.get('system_prior') or '').startswith('structural')
        f = features(r['p_text'], r['q_text'])
        ps, rs = predict(f, struct_edge)
        pc, rc = predict(f, False)
        out.append(dict(r, **f, struct_edge=int(struct_edge),
                        system_pred=ps, system_pred_rule=rs,
                        system_pred_content_only=pc, system_pred_content_only_rule=rc))

    cols = ['pair_id', 'stratum', 'subtype', 'event', 'struct_edge',
            'j_tok', 'contain', 'char3_j', 'pol_p', 'pol_q', 'pol_opposite',
            'rt_v3', 'src_q_only', 'event_unit', 'a3_subset', 'a3_subset_why',
            'slot_cls', 'slot_vals_p', 'slot_vals_q', 'slot_kind',
            'system_pred', 'system_pred_rule',
            'system_pred_content_only', 'system_pred_content_only_rule',
            'p_text', 'q_text']
    with open(os.path.join(BASE, 'E13_系统预测_v3.csv'), 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(out)
    print('[save] E13_系统预测_v3.csv')

    # ---------------- v2 ↔ v3 对照（原 388 行严格可比） ----------------
    v2 = {r['pair_id']: r['system_pred'] for r in csv.DictReader(
        open(os.path.join(BASE, 'E13_系统预测_v2.csv'), encoding='utf-8-sig'))}
    base = [r for r in out if r['stratum'].startswith(('S1_', 'S2_', 'S3_', 'S4_'))]
    dist_388 = dict(collections.Counter(r['system_pred'] for r in base))
    v2_dist_388 = {L: sum(1 for r in base if v2.get(r['pair_id']) == L) for L in LAB}
    changed = [r['pair_id'] for r in out if r['pair_id'] in v2 and v2[r['pair_id']] != r['system_pred']]
    print('v2↔v3 预测变化：%d 对 %s' % (len(changed), changed[:20]))
    print('[原 388 行] v3=%s' % dist_388)
    print('[原 388 行] v2=%s' % v2_dist_388)

    # ---------------- 对 v1.5 合并真值评分 ----------------
    truth = {r['pair_id']: r['truth'] for r in build_merged_truth()}
    ids = [r['pair_id'] for r in out if r['pair_id'] in truth and truth[r['pair_id']]]
    by = {r['pair_id']: r for r in out}
    yt = [truth[i] for i in ids]
    m3 = V2._metrics(yt, [by[i]['system_pred'] for i in ids])
    m2 = V2._metrics(yt, [v2.get(i, 'UNRELATED') for i in ids])
    for m in (m3, m2):
        m['n'] = len(ids)
    print('\n[对 v1.5 合并真值 n=%d]  v2 macro-F1=%.4f → v3 macro-F1=%.4f'
          % (len(ids), m2['macro_F1'], m3['macro_F1']))
    for L in LAB:
        print('  %-10s n_true=%-3d  v2: P=%.3f R=%.3f F1=%.3f  |  v3: P=%.3f R=%.3f F1=%.3f'
              % (L, m3[L]['n_true'], m2[L]['P'], m2[L]['R'], m2[L]['F1'],
                 m3[L]['P'], m3[L]['R'], m3[L]['F1']))

    # 三条守卫的命中数
    hits = {
        'A3_Q_event_unit_suppressed': sum(1 for r in out if r['slot_kind'] == 'num_exclusive'
                                          and r['event_unit'] == 1),
        'A3_a_subset_pass': sum(1 for r in out if r['a3_subset'] == 1),
        'A5_src_q_only_downgrade': sum(1 for r in out if r['src_q_only'] == 1
                                       and r['system_pred_rule'] == 30),
    }
    print('\n守卫命中：%s' % hits)

    res = {
        'n_pairs': len(out), 'params': P, 'guards': hits,
        'v2_vs_v3': {'n_changed': len(changed), 'changed_ids': changed[:40],
                     'v3_distribution_on_388': dist_388, 'v2_distribution_on_388': v2_dist_388,
                     'note': '两者都在原 S1–S4 的 388 行上算 ⇒ 严格可比；唯一变量 = 三个守卫'},
        'truth_v15_merged': {'v3': m3, 'v2': m2,
                             'note': ('真值 = v1.4 的 76 条一致项 + v1.5 复审对 6 条分歧的裁定；'
                                      'v1.5 的收敛是 in-sample 的靶向结果')},
        'rule_hits': dict(collections.Counter('rule%d' % r['system_pred_rule'] for r in out)),
        'pred_distribution': dict(collections.Counter(r['system_pred'] for r in out)),
    }
    with open(os.path.join(BASE, 'e13_claim_graph_v3.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print('[save] e13_claim_graph_v3.json')
    return res


if __name__ == '__main__':
    main()
