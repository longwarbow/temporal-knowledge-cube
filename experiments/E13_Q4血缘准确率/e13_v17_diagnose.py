# -*- coding: utf-8 -*-
"""v1.7 诊断：对 21 条靶向案例（18 人类盲标 + 3 平局方向性）逐条复现 v4 判定路径，
   定位卡点（哪个规则没命中 / 缺什么特征）。产出 E13_v17_诊断.json"""
import os
import csv
import json
import collections

import e13_claim_graph_v3 as V3
import e13_claim_graph_v4 as V4
import e13_predicate_match as PM
import e13_slot_extract as SLOT

BASE = os.path.dirname(os.path.abspath(__file__))

# ---- 21 条靶向（group -> pair_ids）----
TARGETS = {
    'A-CONF->UNREL': ['E13-0735', 'E13-0560', 'E13-0334', 'E13-0270',
                      'E13-0753', 'E13-0102'],
    'A-UNREL误判(对照方向)': ['E13-0086', 'E13-0630'],
    'A-SUPP->UNREL': ['E13-0297', 'E13-0105', 'E13-0140', 'E13-0579', 'E13-0698'],
    'B-SUPP->DERIVE': ['E13-0203', 'E13-0399', 'E13-0415'],
    'C-对照(v4已对)': ['E13-0281', 'E13-0019'],
    'D-平局方向性': ['E13-0596', 'E13-0601', 'E13-0687', 'E13-0139'],
}
QMAP = {
    'E13-0735': 'Q05', 'E13-0086': 'Q08', 'E13-0297': 'Q09', 'E13-0560': 'Q23',
    'E13-0203': 'Q29', 'E13-0334': 'Q32', 'E13-0270': 'Q36', 'E13-0105': 'Q37',
    'E13-0019': 'Q38', 'E13-0399': 'Q39', 'E13-0140': 'Q44', 'E13-0415': 'Q48',
    'E13-0698': 'Q03', 'E13-0753': 'Q11', 'E13-0102': 'Q12', 'E13-0579': 'Q19',
    'E13-0630': 'Q20', 'E13-0281': 'Q06', 'E13-0596': '-', 'E13-0601': '-',
    'E13-0687': '-', 'E13-0139': '-',
}


def main():
    v4rows = {r['pair_id']: r for r in csv.DictReader(
        open(os.path.join(BASE, 'E13_系统预测_v4_全量.csv'), encoding='utf-8-sig'))}
    truth = {r['pair_id']: r['truth'] for r in csv.DictReader(
        open(os.path.join(BASE, 'E13_全量真值_v3.csv'), encoding='utf-8-sig'))}

    diag = []
    for grp, pids in TARGETS.items():
        for pid in pids:
            r = v4rows[pid]
            p, q = r['p_text'], r['q_text']
            f = V4.features(p, q)
            ms = SLOT.match_slots(p, q)
            slots = [m.as_dict() for m in ms] if ms else []
            rel, d = PM.predicate_relation(p, q)
            # 手工复现判定梯，记录每步的布尔值
            steps = {
                'rt': f['rt_v3'], 'contain': round(f['contain'], 3),
                'char3_j': round(f['char3_j'], 3),
                'pol_opposite': f['pol_opposite'],
                'slot_kind': f['slot_kind'], 'slot_cls': f['slot_cls'],
                'slot_vals_p': f['slot_vals_p'], 'slot_vals_q': f['slot_vals_q'],
                'a3_subset': f['a3_subset'], 'a3_subset_why': f['a3_subset_why'],
                'event_unit': f['event_unit'], 'src_q_only': f['src_q_only'],
                'pred_rel': rel, 'pred_fam': '%s | %s' % (f['pred_fam_p'], f['pred_fam_q']),
                'slots_detail': slots,
            }
            diag.append(dict(
                group=grp, pair_id=pid, qid=QMAP.get(pid, '-'),
                truth=truth[pid], v4_pred=r['system_pred'],
                v4_rule=r['system_pred_rule'], steps=steps,
                p=p[:200], q=q[:200]))
    with open(os.path.join(BASE, 'E13_v17_诊断.json'), 'w', encoding='utf-8') as fh:
        json.dump(diag, fh, ensure_ascii=False, indent=1)

    # 控制台摘要
    for grp, pids in TARGETS.items():
        print('#####', grp)
        for pid in pids:
            d = next(x for x in diag if x['pair_id'] == pid)
            s = d['steps']
            print('%s %-9s v4=%-9s(rule%-3s) slot=%s/%s vals=%s|%s sub=%d contain=%.2f j=%.2f pred=%s fam=%s' % (
                pid, d['truth'], d['v4_pred'], d['v4_rule'], s['slot_cls'][:12],
                s['slot_kind'][:14], s['slot_vals_p'][:12], s['slot_vals_q'][:12],
                s['a3_subset'], s['contain'], s['char3_j'], s['pred_rel'][:14],
                s['pred_fam'][:34]))
        print()


if __name__ == '__main__':
    main()
