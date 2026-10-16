# -*- coding: utf-8 -*-
"""
E13 — 全量 v3 收口：通道 P 关闭平局 + 构造器 v4 全量评分（逐层）

输入：E13_标注集_v3.csv（755 行元数据）+ E13_全量真值_v3.csv（M/N/O 合并）
     + E13_全量v3_通道P.csv（5 条三方全异的终裁）
输出：E13_全量真值_v3.csv（就地更新，TIE → 通道 P 标签）
     E13_系统预测_v4_全量.csv、E13_全量v3_系统评分.json
"""
import os
import csv
import json
import collections

import e13_claim_graph_v4 as V4
import e13_claim_graph_v3 as V3
import e13_predicate_match as PM

BASE = os.path.dirname(os.path.abspath(__file__))
LAB = V3.V2.LAB


def load(p):
    return [{(k or '').lstrip('\ufeff'): v for k, v in r.items()}
            for r in csv.DictReader(open(p, encoding='utf-8-sig'))]


def main():
    rows = load(os.path.join(BASE, 'E13_标注集_v3.csv'))
    truth = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_全量真值_v3.csv'))}
    P = {r['pair_id']: r for r in load(os.path.join(BASE, 'E13_全量v3_通道P.csv'))}

    # ---------- ① 关闭平局 ----------
    n_tie = 0
    for pid, t in truth.items():
        if t['truth'] == 'TIE':
            t['truth'] = P[pid]['label_relation']
            t['how'] = 'P_final(4th channel)'
            t['note_P'] = P[pid].get('label_notes', '')
            n_tie += 1
    cols_t = []
    for t in truth.values():
        for k in t:
            if k not in cols_t:
                cols_t.append(k)
    with open(os.path.join(BASE, 'E13_全量真值_v3.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols_t, extrasaction='ignore')
        w.writeheader()
        w.writerows([truth[p] for p in sorted(truth)])
    print('关闭平局 %d 条 → 真值 %d 行' % (n_tie, len(truth)))

    # ---------- ② v4 全量预测 ----------
    out = []
    for r in rows:
        struct_edge = (r.get('system_prior') or '').startswith('structural')
        f = V4.features(r['p_text'], r['q_text'])
        pred, rule = V4.predict(f, struct_edge)
        out.append(dict(r, **f, struct_edge=int(struct_edge),
                        system_pred=pred, system_pred_rule=rule))
    cols = ['pair_id', 'stratum', 'subtype', 'event', 'struct_edge',
            'j_tok', 'contain', 'char3_j', 'pol_opposite', 'rt_v3', 'src_q_only',
            'event_unit', 'a3_subset', 'slot_cls', 'slot_kind',
            'pred_rel', 'pred_fam_p', 'pred_fam_q', 'pred_inter', 'pred_block',
            'system_pred', 'system_pred_rule', 'p_text', 'q_text']
    with open(os.path.join(BASE, 'E13_系统预测_v4_全量.csv'), 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(out)

    by = {r['pair_id']: r for r in out}
    pids = [p for p in sorted(truth) if truth[p]['truth'] in LAB]
    yt = [truth[p]['truth'] for p in pids]
    yp = [by[p]['system_pred'] for p in pids]
    m_all = V3.V2._metrics(yt, yp)
    m_all['n'] = len(pids)
    acc_all = sum(1 for t, p in zip(yt, yp) if t == p) / len(pids)
    # 空模型：恒预测全局多数类
    maj = collections.Counter(yt).most_common(1)[0][0]
    m_naive = V3.V2._metrics(yt, [maj] * len(pids))
    acc_naive = sum(1 for t in yt if t == maj) / len(pids)

    per = {}
    for st in sorted(set(truth[p]['stratum'] for p in pids)):
        sp = [p for p in pids if truth[p]['stratum'] == st]
        per[st] = {'n': len(sp),
                   'truth': dict(collections.Counter(truth[p]['truth'] for p in sp)),
                   'pred': dict(collections.Counter(by[p]['system_pred'] for p in sp))}
        m = V3.V2._metrics([truth[p]['truth'] for p in sp], [by[p]['system_pred'] for p in sp])
        m['n'] = len(sp)
        per[st]['metrics'] = {L: {k: round(m[L][k], 4) for k in ('P', 'R', 'F1', 'n_true')} for L in LAB}
        per[st]['macro_F1'] = round(m['macro_F1'], 4)
        per[st]['macro_F1'] = round(m['macro_F1'], 4)
        per[st]['accuracy'] = round(sum(1 for t, p in zip(
            [truth[x]['truth'] for x in sp], [by[x]['system_pred'] for x in sp]) if t == p) / len(sp), 4)

    print('\n[全量 n=%d] macro-F1 v4 = %.4f ｜ accuracy = %.4f ｜ 空模型(恒%s) macro-F1 = %.4f / acc = %.4f'
          % (len(pids), m_all['macro_F1'], acc_all, maj, m_naive['macro_F1'], acc_naive))
    for L in LAB:
        print('  %-10s n_true=%-3d P=%.3f R=%.3f F1=%.3f' % (L, m_all[L]['n_true'],
              m_all[L]['P'], m_all[L]['R'], m_all[L]['F1']))
    print()
    for st, d in per.items():
        print('  %-32s n=%3d acc=%.3f macroF1=%.3f  CONFLICT F1=%s' %
              (st, d['n'], d['accuracy'], d['macro_F1'], d['metrics']['CONFLICT']['F1']))

    res = {
        'n': len(pids), 'truth_dist': dict(collections.Counter(yt)),
        'v4_overall': {'n': m_all['n'], 'macro_F1': m_all['macro_F1'], 'accuracy': round(acc_all, 4),
                       **{L: m_all[L] for L in LAB}},
        'naive_majority': {'label': maj, 'macro_F1': m_naive['macro_F1'], 'accuracy': round(acc_naive, 4)},
        'per_stratum': per,
        'guards': {'A4_block': sum(1 for r in out if r['pred_block'] == 1),
                   'A4_suppressed': sum(1 for r in out if r['system_pred_rule'] == 51),
                   'A4_not_applicable': sum(1 for r in out if r['pred_rel'] == 'not_applicable')},
        'ties_closed_by_channel_P': n_tie,
        'probe_status': ('探针 κ（多通道）= 见 E13_全量v3_一致性.json；'
                         '人类 κ 仍缺（协议 §7 探针地位声明）'),
    }
    with open(os.path.join(BASE, 'E13_全量v3_系统评分.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print('\n[save] E13_全量v3_系统评分.json / E13_系统预测_v4_全量.csv')

    # ---------- ③ 与先导对照 ----------
    print('\n[先导→全量 对照] 真值分布')
    print('  先导 76 条 ：', {'UNRELATED': 36, 'SUPPORT': 22, 'DERIVE': 12, 'CONFLICT': 6})
    print('  全量 755 条：', dict(collections.Counter(yt)))
    return res


if __name__ == '__main__':
    main()
