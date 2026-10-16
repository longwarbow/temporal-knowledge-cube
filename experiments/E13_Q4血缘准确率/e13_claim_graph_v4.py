# -*- coding: utf-8 -*-
"""
E13 — $Q_4$ 血缘准确率：**声明图构造器 v4**（同步判据 v1.5 的 A4）

与 v3 的差别（**只加一个可机械化守卫，不动阈值、不动其它守卫**）
------------------------------------------------------------------
| 守卫 | 依据 | 作用 |
|---|---|---|
| **A4 谓词门** | 判据 v1.5 §2.5 + `e13_predicate_match.py` | 两侧数值绑到**不同谓词族**（`block`）⇒ 同一槽位不成立 ⇒ **不得判 CONFLICT/SUPPORT**，落 UNRELATED |

★ 为什么 v3 做不到 A4：v3 的量纲是 `HOSTAGE`/`SUSPECT` 这类**粗类**，
`hostages held`（被扣押）与 `hostages escaped`（逃离）被归成同一类。
v4 用 `e13_predicate_match` 的**族表**（`CUSTODY_HELD` vs `CUSTODY_EXITED`）把它们分开。

⚠️ `not_applicable`（任一侧无数值族）**不放行任何阻断**——否则 S5 的纯转述层会被误杀。

产物：E13_系统预测_v4.csv / e13_claim_graph_v4.json
"""
import os
import csv
import json
import collections

import e13_claim_graph_v3 as V3
import e13_predicate_match as PM

BASE = os.path.dirname(os.path.abspath(__file__))
LAB = V3.V2.LAB


def features(p, q):
    f = V3.features(p, q)
    rel, d = PM.predicate_relation(p, q)
    f['pred_rel'] = rel
    f['pred_fam_p'] = '+'.join(sorted(d['sp'])) or '-'
    f['pred_fam_q'] = '+'.join(sorted(d['sq'])) or '-'
    f['pred_inter'] = '+'.join(d['inter']) or '-'
    f['pred_block'] = int(rel == 'diff_family')
    return f


def predict(f, struct_edge):
    lab, rule = V3.predict(f, struct_edge)
    # ★ A4 门：谓词不可归并 ⇒ 同一槽位不成立
    if f['pred_block'] and lab in ('CONFLICT', 'SUPPORT'):
        return 'UNRELATED', 51
    return lab, rule


def main():
    src = os.path.join(BASE, 'E13_标注集_v2.csv')
    rows = list(csv.DictReader(open(src, encoding='utf-8-sig')))
    print('读入 %d 对（%s）' % (len(rows), os.path.basename(src)))

    out = []
    for r in rows:
        struct_edge = (r.get('system_prior') or '').startswith('structural')
        f = features(r['p_text'], r['q_text'])
        ps, rs = predict(f, struct_edge)
        pc, _ = predict(f, False)
        out.append(dict(r, **f, struct_edge=int(struct_edge),
                        system_pred=ps, system_pred_rule=rs,
                        system_pred_content_only=pc))

    cols = ['pair_id', 'stratum', 'subtype', 'event', 'struct_edge',
            'j_tok', 'contain', 'char3_j', 'pol_opposite', 'rt_v3', 'src_q_only',
            'event_unit', 'a3_subset', 'slot_cls', 'slot_kind',
            'pred_rel', 'pred_fam_p', 'pred_fam_q', 'pred_inter', 'pred_block',
            'system_pred', 'system_pred_rule', 'system_pred_content_only',
            'p_text', 'q_text']
    with open(os.path.join(BASE, 'E13_系统预测_v4.csv'), 'w',
              encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(out)
    print('[save] E13_系统预测_v4.csv')

    # ---------------- v3 ↔ v4 对照（原 388 行严格可比） ----------------
    def load_pred(fn, col='system_pred'):
        return {r['pair_id']: r[col] for r in csv.DictReader(
            open(os.path.join(BASE, fn), encoding='utf-8-sig'))}
    v3 = load_pred('E13_系统预测_v3.csv')
    v2 = load_pred('E13_系统预测_v2.csv')
    base = [r for r in out if r['stratum'].startswith(('S1_', 'S2_', 'S3_', 'S4_'))]
    changed = [r['pair_id'] for r in out if r['pair_id'] in v3 and v3[r['pair_id']] != r['system_pred']]
    dist_v4 = dict(collections.Counter(r['system_pred'] for r in base))
    dist_v3 = {L: sum(1 for r in base if v3.get(r['pair_id']) == L) for L in LAB}
    print('v3↔v4 预测变化：%d 对 %s' % (len(changed), changed[:20]))
    print('[原 388 行] v4=%s' % dist_v4)
    print('[原 388 行] v3=%s' % dist_v3)

    # ---------------- 对 v1.5 合并真值评分 ----------------
    truth = {r['pair_id']: r['truth'] for r in V3.build_merged_truth()}
    ids = [r['pair_id'] for r in out if r['pair_id'] in truth and truth[r['pair_id']]]
    by = {r['pair_id']: r for r in out}
    yt = [truth[i] for i in ids]
    mv4 = V3.V2._metrics(yt, [by[i]['system_pred'] for i in ids])
    mv3 = V3.V2._metrics(yt, [v3.get(i, 'UNRELATED') for i in ids])
    mv2 = V3.V2._metrics(yt, [v2.get(i, 'UNRELATED') for i in ids])
    for m in (mv2, mv3, mv4):
        m['n'] = len(ids)
    print('\n[对 v1.5 合并真值 n=%d]  v2=%.4f  v3=%.4f  →  v4=%.4f'
          % (len(ids), mv2['macro_F1'], mv3['macro_F1'], mv4['macro_F1']))
    for L in LAB:
        print('  %-10s n_true=%-3d  v3: P=%.3f R=%.3f F1=%.3f  |  v4: P=%.3f R=%.3f F1=%.3f'
              % (L, mv4[L]['n_true'], mv3[L]['P'], mv3[L]['R'], mv3[L]['F1'],
                 mv4[L]['P'], mv4[L]['R'], mv4[L]['F1']))

    hits = {
        'A4_block_pairs': sum(1 for r in out if r['pred_block'] == 1),
        'A4_suppressed': sum(1 for r in out if r['system_pred_rule'] == 51),
        'A4_not_applicable': sum(1 for r in out if r['pred_rel'] == 'not_applicable'),
    }
    print('\n守卫命中：%s' % hits)

    res = {
        'n_pairs': len(out), 'guards': hits,
        'v3_vs_v4': {'n_changed': len(changed), 'changed_ids': changed[:40],
                     'v4_distribution_on_388': dist_v4, 'v3_distribution_on_388': dist_v3,
                     'note': '唯一变量 = A4 谓词门（其余全部 import v3）'},
        'truth_v15_merged': {'v4': mv4, 'v3': mv3, 'v2': mv2,
                             'note': '真值 = v1.4 的 76 条一致项 + v1.5 复审对 6 条分歧的裁定（in-sample）'},
        'rule_hits': dict(collections.Counter('rule%d' % r['system_pred_rule'] for r in out)),
        'pred_distribution': dict(collections.Counter(r['system_pred'] for r in out)),
    }
    with open(os.path.join(BASE, 'e13_claim_graph_v4.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print('[save] e13_claim_graph_v4.json')
    return res


if __name__ == '__main__':
    main()
