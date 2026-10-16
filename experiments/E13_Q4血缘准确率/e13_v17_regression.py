# -*- coding: utf-8 -*-
"""v1.7 回归：v5 vs v4 在全量 755 上的对比审计。
   产出 E13_v17_回归.json + 控制台报告。
   判收标准（事先写死）：
   - 靶向：21 条中机械可达 16 条收敛（自检已证 13 mini + 3 条真实变体）
   - 对照：正典 8 条（0503/0487/0458/0482/0491/0534/0539/0543）+ 0281/0019/0139 零漂移
   - 全局：macro-F1 不降；S5 DERIVE 真值翻转逐条列出（预期 ≤3，超出须逐条归因）
"""
import os
import csv
import json
import collections

import e13_claim_graph_v5 as V5
import e13_claim_graph_v3 as V3

BASE = os.path.dirname(os.path.abspath(__file__))

CONTROLS = ['E13-0281', 'E13-0019', 'E13-0139', 'E13-0503', 'E13-0487',
            'E13-0458', 'E13-0482', 'E13-0491', 'E13-0534', 'E13-0539', 'E13-0543']
HUMAN_TARGET = {
    'E13-0735': 'CONFLICT', 'E13-0560': 'CONFLICT', 'E13-0334': 'CONFLICT',
    'E13-0270': 'CONFLICT', 'E13-0753': 'CONFLICT', 'E13-0102': 'CONFLICT',
    'E13-0086': 'UNRELATED', 'E13-0630': 'UNRELATED',
    'E13-0297': 'SUPPORT', 'E13-0105': 'SUPPORT', 'E13-0140': 'SUPPORT',
    'E13-0579': 'SUPPORT', 'E13-0698': 'SUPPORT',
    'E13-0203': 'SUPPORT', 'E13-0399': 'SUPPORT', 'E13-0415': 'SUPPORT',
    'E13-0596': 'CONFLICT', 'E13-0601': 'CONFLICT', 'E13-0687': 'CONFLICT',
}


def main():
    rows = list(csv.DictReader(open(os.path.join(BASE, 'E13_标注集_v3.csv'),
                                    encoding='utf-8-sig')))
    truth = {r['pair_id']: r['truth'] for r in csv.DictReader(
        open(os.path.join(BASE, 'E13_全量真值_v3.csv'), encoding='utf-8-sig'))}
    strata = {r['pair_id']: r['stratum'] for r in
              csv.DictReader(open(os.path.join(BASE, 'E13_全量真值_v3.csv'),
                                  encoding='utf-8-sig'))}
    v4 = {r['pair_id']: (r['system_pred'], r['system_pred_rule']) for r in
          csv.DictReader(open(os.path.join(BASE, 'E13_系统预测_v4_全量.csv'),
                              encoding='utf-8-sig'))}

    out = []
    for r in rows:
        struct_edge = (r.get('system_prior') or '').startswith('structural')
        f = V5.features(r['p_text'], r['q_text'])
        ps, rs = V5.predict(f, struct_edge)
        out.append(dict(r, **f, system_pred=ps, system_pred_rule=rs))

    cols = ['pair_id', 'stratum', 'subtype', 'event', 'struct_edge',
            'j_tok', 'contain', 'char3_j', 'pol_opposite', 'rt_v3', 'src_q_only',
            'event_unit', 'a3_subset', 'slot_cls', 'slot_kind',
            'pred_rel', 'pred_fam_p', 'pred_fam_q', 'pred_inter', 'pred_block',
            'handle_trace', 'head_attrib_p', 'report_verb',
            'v5_num_verdict', 'v5_num_rule', 'v5_num_why',
            'system_pred', 'system_pred_rule', 'p_text', 'q_text']
    with open(os.path.join(BASE, 'E13_系统预测_v5_全量.csv'), 'w',
              encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(out)
    print('[save] E13_系统预测_v5_全量.csv (n=%d)' % len(out))

    by = {r['pair_id']: r for r in out}
    ids = [r['pair_id'] for r in out if r['pair_id'] in truth and truth[r['pair_id']]]
    yt = [truth[i] for i in ids]
    m4 = V3.V2._metrics(yt, [v4[i][0] if i in v4 else 'UNRELATED' for i in ids])
    m5 = V3.V2._metrics(yt, [by[i]['system_pred'] for i in ids])
    for m in (m4, m5):
        m['n'] = len(ids)

    print('\n===== 全量指标（n=%d）=====' % len(ids))
    print('macro-F1:  v4=%.4f  →  v5=%.4f  (%+.4f)' %
          (m4['macro_F1'], m5['macro_F1'], m5['macro_F1'] - m4['macro_F1']))
    for L in V3.V2.LAB:
        print('  %-10s n_true=%-3d  v4: P=%.3f R=%.3f F1=%.3f | v5: P=%.3f R=%.3f F1=%.3f' % (
            L, m5[L]['n_true'], m4[L]['P'], m4[L]['R'], m4[L]['F1'],
            m5[L]['P'], m5[L]['R'], m5[L]['F1']))

    # ---- 翻转审计 ----
    flips = []
    for r in out:
        pid = r['pair_id']
        if pid in v4 and v4[pid][0] != r['system_pred']:
            flips.append(dict(
                pair_id=pid, stratum=strata.get(pid, ''), truth=truth.get(pid, ''),
                v4=v4[pid][0], v4_rule=v4[pid][1], v5=r['system_pred'],
                v5_rule=r['system_pred_rule'], v5_why=r['v5_num_why'] or 'M1/handle',
                p=r['p_text'][:110], q=r['q_text'][:110]))
    print('\n===== v4→v5 翻转：%d 条 =====' % len(flips))
    fc = collections.Counter((f['v4'], f['v5']) for f in flips)
    for (a, b), c in fc.most_common():
        print('  %-9s→%-9s : %d' % (a, b, c))
    print('\n-- 翻转明细 --')
    for f in flips:
        mark = '✓' if f['v5'] == f['truth'] else ('=' if f['v4'] == f['truth'] else '✗')
        print('%s %s [%s] %s→%s (rule%s) truth=%s | %s' % (
            mark, f['pair_id'], f['stratum'][:13], f['v4'], f['v5'],
            f['v5_rule'], f['truth'], f['v5_why'][:30]))

    # ---- 靶向与对照 ----
    print('\n===== 人类靶向 19 条（机械可达 16）=====')
    t_ok = t_bad = 0
    for pid, want in HUMAN_TARGET.items():
        got = by[pid]['system_pred']
        ok = got == want
        t_ok += ok
        t_bad += not ok
        print('%s %s want=%-9s got=%-9s (rule%s)' % (
            '✓' if ok else '✗', pid, want, got, by[pid]['system_pred_rule']))
    print('\n===== 对照 %d 条（不得漂移）=====' % len(CONTROLS))
    c_ok = c_bad = 0
    for pid in CONTROLS:
        if pid not in by or pid not in truth:
            print('  ? %s 不在数据中' % pid)
            continue
        old = v4[pid][0] if pid in v4 else '?'
        new = by[pid]['system_pred']
        drift = old != new
        c_ok += not drift
        c_bad += drift
        print('%s %s v4=%-9s v5=%-9s truth=%-9s' % (
            '✗漂移' if drift else '✓稳定', pid, old, new, truth[pid]))

    res = {
        'metrics': {'v4': m4, 'v5': m5},
        'n_flips': len(flips),
        'flip_matrix': {'%s->%s' % k: v for k, v in fc.items()},
        'flips': flips,
        'human_target': {'ok': t_ok, 'bad': t_bad},
        'controls': {'ok': c_ok, 'bad': c_bad},
        'rule_hits': dict(collections.Counter('rule%d' % r['system_pred_rule'] for r in out)),
    }
    with open(os.path.join(BASE, 'E13_v17_回归.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print('\n[save] E13_v17_回归.json')
    return res


if __name__ == '__main__':
    main()
