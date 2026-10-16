# -*- coding: utf-8 -*-
"""
E13 — $Q_4$ 血缘准确率：**声明图构造器（system_pred）** —— 协议 §5 接口的 C6a 实现

目标
----
按 `E13_标注协议.md` §5，对同一批 388 对断言 $(p,q)$ 输出系统预测
（DERIVE / SUPPORT / CONFLICT / UNRELATED），供 `e13_score.py` 打分。

两版对照（直接量化 D24「不依赖单一平台 API」的代价）
---------------------------------------------------
  S+ 结构可用版：额外使用真实回复树边（"q replies to p"）——**只降阈值，不直接定类**；
  S- 内容可用版：完全不用平台结构，只用两条文本。

> 与 A2 的关系：A2 已证明 384 维句嵌入对父边 AUC=0.5005（随机）。
> 本构造器用**显式规则**（近似逐字重复 / 极性 / 数值互斥）而非嵌入——
> 若 S+ 与 S- 差异很小，说明"结构"只在极窄的模糊带上起作用；差异大则说明结构不可替代。

判定梯（自上而下，首个命中即出）
--------------------------------
| # | 条件 | 预测 |
|---|---|---|
| 1 | RT/引用痕迹 且 词包含度 ≥ 0.50 | DERIVE |
| 2 | 字符 3-gram Jaccard ≥ 0.85（去 @/URL 后） | DERIVE |
| 3 | 词包含度 ≥ 0.85 且 极性同向 | DERIVE |
| 4 | 极性相反 且 词包含度 ≥ 0.25（S+ 且结构边可用 → ≥0.15） | CONFLICT |
| 5 | 数值互斥（双方均有数字且交集空）且 词包含度 ≥ 0.30 | CONFLICT |
| 6 | 词包含度 ≥ 0.60 且 极性同向 | SUPPORT |
| 7 | 词包含度 ≥ 0.40 且 词 Jaccard ≥ 0.25 且 极性同向（S+ 结构边可用 → ≥0.30） | SUPPORT |
| 8 | 以上皆不中 | UNRELATED |

产出
----
  · E13_系统预测.csv   —— pair_id / stratum / 特征 / system_pred / system_pred_content_only
  · e13_claim_graph.json —— 规则表、命中分布、S+/S- 差异、参数
"""
import os
import re
import csv
import json
import html
import math
import collections

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------- 规则参数（集中在顶部，便于消融） ----------------
P = {
    'rt_containment': 0.50,        # 规则 1
    'char3_jaccard': 0.85,         # 规则 2
    'contain_derive': 0.85,        # 规则 3
    'contain_conflict': 0.25,      # 规则 4
    'contain_conflict_struct': 0.15,   # 规则 4（S+ 结构边可用时放宽）
    'contain_numeric_conflict': 0.30,  # 规则 5
    'contain_support_hi': 0.60,    # 规则 6
    'contain_support_lo': 0.40,    # 规则 7
    'jaccard_support_lo': 0.25,    # 规则 7
    'contain_support_lo_struct': 0.30,  # 规则 7（S+ 结构边可用时放宽）
}

URL_RE = re.compile(r'http\S+|www\.\S+')
MENT_RE = re.compile(r'@[A-Za-z0-9_]+')
RT_RE = re.compile(r'^\s*rt\b[:\s]', re.I)
NONWORD_RE = re.compile(r'[^a-z0-9\s]')
NUM_RE = re.compile(r'\b\d[\d,\.]*\b')
HASH_RE = re.compile(r'#')

STOP = set("""a an the and or but if then than so because as of to in on at by for with from into over under
this that these those it its it's is are was were be been being am do does did done doing have has had having
i you he she we they me him her us them my your his their our mine yours theirs
will would shall should can could may might must
there here what which who whom whose when where why how
all any both each few more most other some such no nor not only own same too very
just now also get got go going went make makes made say says said rt via amp
u ur im dont cant wont isnt arent wasnt didnt doesnt
""".split())

NEG = set("""not n't no never none nobody nothing neither nor without
deny denies denied denial false fake fakes hoax hoaxed falsely debunk debunked debunks
rumour rumor rumours rumors myth myths untrue fabricated fabricate wrong incorrect
disprove disproven refute refuted misinformation misleading conspiracy nonsense
lie lies lied liar bullshit rubbish bollocks staged photoshopped edited doctored
crap stupid idiot idiotic absurd ridiculous""".split())


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


def polarity(t):
    toks = norm_tokens(t)
    c = sum(1 for w in toks if w in NEG or "n't" in w)
    return 1 if c > 0 else 0, c


def jacc(a, b):
    if not a or not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def containment(a, b):
    """包含度 = |A∩B| / min(|A|,|B|)  —— 短文本比 Jaccard 稳健"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def is_rt(raw):
    t = html.unescape(raw or '')
    return bool(re.search(r'\brt\b\s*@', t, re.I)) or bool(re.search(r'\bvia\s+@', t, re.I))


def features(p, q):
    tp, tq = norm_tokens(p), norm_tokens(q)
    sp, sq = set(tp), set(tq)
    pol_p, neg_p = polarity(p)
    pol_q, neg_q = polarity(q)
    np_ = set(NUM_RE.findall(html.unescape(p or '')))
    nq_ = set(NUM_RE.findall(html.unescape(q or '')))
    num_conflict = bool(np_ and nq_ and not (np_ & nq_))
    return {
        'j_tok': round(jacc(sp, sq), 4),
        'contain': round(containment(sp, sq), 4),
        'char3_j': round(jacc(char3(p), char3(q)), 4),
        'pol_p': pol_p, 'pol_q': pol_q, 'neg_p': neg_p, 'neg_q': neg_q,
        'pol_opposite': int(pol_p != pol_q),
        'rt_p': int(is_rt(p)), 'rt_q': int(is_rt(q)),
        'n_nums_p': len(np_), 'n_nums_q': len(nq_),
        'num_conflict': int(num_conflict),
        'len_p': len(tp), 'len_q': len(tq),
    }


def predict(f, struct_edge):
    """返回 (pred, rule_id)。struct_edge = 真实回复树边可用（S+）。"""
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
    if f['num_conflict'] and f['contain'] >= P['contain_numeric_conflict']:
        return 'CONFLICT', 5
    if f['contain'] >= P['contain_support_hi'] and not f['pol_opposite']:
        return 'SUPPORT', 6
    thr7 = P['contain_support_lo_struct'] if struct_edge else P['contain_support_lo']
    if (f['contain'] >= thr7 and f['j_tok'] >= P['jaccard_support_lo']
            and not f['pol_opposite']):
        return 'SUPPORT', 7
    return 'UNRELATED', 8


def main():
    src = os.path.join(BASE, 'E13_标注集.csv')
    rows = list(csv.DictReader(open(src, encoding='utf-8-sig')))
    print('读入 %d 对' % len(rows))

    out = []
    for r in rows:
        struct_edge = (r.get('system_prior') or '').startswith('structural')
        f = features(r['p_text'], r['q_text'])
        pred_s, rule_s = predict(f, struct_edge)
        pred_c, rule_c = predict(f, False)
        out.append(dict(r, **f,
                        struct_edge=int(struct_edge),
                        system_pred=pred_s, system_pred_rule=rule_s,
                        system_pred_content_only=pred_c,
                        system_pred_content_only_rule=rule_c))

    cols = (['pair_id', 'stratum', 'event', 'struct_edge',
             'j_tok', 'contain', 'char3_j', 'pol_p', 'pol_q', 'pol_opposite',
             'rt_p', 'rt_q', 'n_nums_p', 'n_nums_q', 'num_conflict', 'len_p', 'len_q',
             'system_pred', 'system_pred_rule',
             'system_pred_content_only', 'system_pred_content_only_rule',
             'p_text', 'q_text'])
    p_out = os.path.join(BASE, 'E13_系统预测.csv')
    with open(p_out, 'w', encoding='utf-8-sig', newline='') as f_:
        w = csv.DictWriter(f_, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(out)
    print('[save] E13_系统预测.csv')

    # ---------------- 统计 ----------------
    LAB = ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED']
    dist = {L: collections.Counter() for L in LAB}
    by_stratum = collections.defaultdict(collections.Counter)
    rules = collections.Counter()
    for r in out:
        dist[r['system_pred']][r['system_pred_rule']] += 1
        by_stratum[r['stratum']][r['system_pred']] += 1
        rules['rule%d' % r['system_pred_rule']] += 1

    n_diff = sum(1 for r in out if r['system_pred'] != r['system_pred_content_only'])
    diff_stratum = collections.Counter(r['stratum'] for r in out
                                       if r['system_pred'] != r['system_pred_content_only'])

    # 特征画像（按层）
    prof = {}
    for s in sorted(by_stratum):
        sub = [r for r in out if r['stratum'] == s]
        def mean(k):
            return round(sum(r[k] for r in sub) / len(sub), 4)
        prof[s] = {'n': len(sub), 'contain_mean': mean('contain'),
                   'j_tok_mean': mean('j_tok'), 'char3_mean': mean('char3_j'),
                   'pol_opposite_rate': mean('pol_opposite'),
                   'rt_rate': round(sum(1 for r in sub if r['rt_p'] or r['rt_q']) / len(sub), 4)}

    res = {
        'n_pairs': len(out),
        'params': P,
        'rule_hits': dict(rules),
        'pred_distribution': {k: dict(v) for k, v in dist.items()},
        'by_stratum': {k: dict(v) for k, v in by_stratum.items()},
        'feature_profile_by_stratum': prof,
        'S_plus_vs_S_minus': {
            'n_differ': n_diff,
            'pct_differ': round(100.0 * n_diff / len(out), 2),
            'by_stratum': dict(diff_stratum),
        },
        'note': ('规则式声明图构造器。S+ = 使用真实回复树边（仅放宽阈值 4/7），'
                 'S- = 完全不用平台结构。与 E13 协议 §5 接口对应。'),
    }
    with open(os.path.join(BASE, 'e13_claim_graph.json'), 'w', encoding='utf-8') as f_:
        json.dump(res, f_, ensure_ascii=False, indent=2)
    print('[save] e13_claim_graph.json')

    print('\n规则命中：%s' % dict(rules))
    for s in sorted(by_stratum):
        print('  %-38s %s' % (s, dict(by_stratum[s])))
    print('S+ 与 S- 预测不同：%d 对 (%.2f%%)  分层 %s'
          % (n_diff, 100.0 * n_diff / len(out), dict(diff_stratum)))


if __name__ == '__main__':
    main()
