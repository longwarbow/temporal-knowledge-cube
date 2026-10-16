# -*- coding: utf-8 -*-
"""E13 系统 v6 = v5 + R1 评论对守卫 + R2 语义复述守卫（轮 9）

== 本版新增（两条残余缝的机械化尝试）==
| # | 修订 | 规则号 | 依据 |
|---|---|---|---|
| **R1** | **评论对守卫（低召回）**：lab==SUPPORT 且**两侧皆无谓词族锚**、且至少一侧含**强意见/修辞标记** ⇒ **UNRELATED** | rule 53 | 判据 §2.2（`E13-0086`/`E13-0093`） |
| **R2** | **语义复述守卫**：lab==UNRELATED 且为**兜底**（rule 10）且 q/p 语义余弦 **≥ 0.70** 且无数值冲突 ⇒ **SUPPORT** | rule 39 | 判据 A5-③（无锚独立复述） |

== R2 的语义模型 ==
PHEME 数据集**自带** 384 维句向量（`text_embedding`），与标注**完全独立**（标注从未使用）
⇒ 本轮的 R2 增益**不是纯循环**：它是一个独立信号对系统漏检的召回。
embedding 缓存：`E13_v6_emb_cache.pkl`（1221 个节点，覆盖 755/755 对）。

== 已声明代价 ==
R2(τ=0.70) 在 755 上触发 24 条 = **18 对 / 6 错**；6 条错为"同事件不同面"
（0143/0251/0281/0288/0336/0542，其中 0281 是 v1.7 轮的对照）⇒ 见于报告 §4。

产出：E13_系统预测_v6_全量.csv
"""
import os
import re
import csv
import math
import pickle
import sys

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import e13_claim_graph_v5 as V5
import e13_slot_extract as SLOT

TAU_SEM = 0.70            # R2 语义阈值（冻结）

_EMB_PATH = os.path.join(BASE, 'E13_v6_emb_cache.pkl')
_EMB = pickle.load(open(_EMB_PATH, 'rb')) if os.path.exists(_EMB_PATH) else {}

# 强意见 / 修辞标记（区别于裸 "?"）：只在"两侧都无谓词族锚"时作为补充证据
_STRONG_OP = re.compile(
    r"\b(wtf|ridiculous|misleading|seriously|kidding|not buying|"
    r"how does that make you feel|make you feel|at all|isn't much)\b", re.I)


def cos(a, b):
    if not a or not b:
        return 0.0
    n = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return n / (na * nb) if na and nb else 0.0


def has_family(text):
    return any(w in SLOT.TRIG2CLASS for w in re.findall(r'[a-z]+', (text or '').lower()))


def is_comment_pair(p, q):
    """R1 触发条件：两侧皆无谓词族锚 + 至少一侧含强意见/修辞标记。"""
    if has_family(p) or has_family(q):
        return 0
    if _STRONG_OP.search(p or '') or _STRONG_OP.search(q or ''):
        return 1
    return 0


def sem_sim(p_id, q_id):
    return cos(_EMB.get(p_id), _EMB.get(q_id))


def predict_v6(f, struct_edge, pid_p=None, pid_q=None, p=None, q=None):
    lab, rule = V5.predict(f, struct_edge)
    # R1 评论对守卫
    if lab == 'SUPPORT' and is_comment_pair(p, q):
        return 'UNRELATED', 53
    # R2 语义复述守卫：仅在**兜底**（rule 10，无任何槽位/数值信号）且无数值冲突时启用
    if lab == 'UNRELATED' and rule == 10 and f.get('v5_num_verdict') != 'CONFLICT':
        if sem_sim(pid_p, pid_q) >= TAU_SEM:
            return 'SUPPORT', 39
    return lab, rule


def main():
    src = os.path.join(BASE, 'E13_标注集_v3.csv')
    rows = list(csv.DictReader(open(src, encoding='utf-8-sig')))
    print('读入 %d 对' % len(rows))
    out = []
    for r in rows:
        p, q = r['p_text'] or '', r['q_text'] or ''
        struct_edge = (r.get('system_prior') or '').startswith('structural')
        f = V5.features(p, q)
        s = sem_sim(r['p_tweet_id'], r['q_tweet_id'])
        f['sem_cos'] = round(s, 4)
        f['comment_pair'] = is_comment_pair(p, q)
        ps, rs = V5.predict(f, struct_edge)
        v6s, v6r = predict_v6(f, struct_edge, r['p_tweet_id'], r['q_tweet_id'], p, q)
        out.append(dict(r, **f, struct_edge=int(struct_edge),
                        v5_pred=ps, v5_rule=rs,
                        system_pred=v6s, system_pred_rule=v6r))
    cols = ['pair_id', 'stratum', 'subtype', 'event', 'struct_edge',
            'contain', 'sem_cos', 'comment_pair', 'event_unit', 'pred_rel',
            'pred_block', 'num_salad', 'v5_num_verdict', 'v5_num_rule',
            'new_content_ratio', 'v5_pred', 'v5_rule', 'system_pred', 'system_pred_rule',
            'p_text', 'q_text']
    with open(os.path.join(BASE, 'E13_系统预测_v6_全量.csv'), 'w',
              encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(out)
    print('[save] E13_系统预测_v6_全量.csv')
    import collections
    print('规则命中:', dict(collections.Counter('rule%d' % r['system_pred_rule'] for r in out)))
    return out


if __name__ == '__main__':
    main()
