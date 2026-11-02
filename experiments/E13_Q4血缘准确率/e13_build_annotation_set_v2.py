# -*- coding: utf-8 -*-
"""
E13 — $Q_4$ 血缘准确率：**6 层标注集构造 v2**（S5 / S6 落地）

动因（协议 v1.1 §4.2 + 用户 2026-10-11 拍板「选 a」）
----------------------------------------------------
先导标注证明原 4 层**拿不到少数类**：外推到 388 条只剩 ≈15 DERIVE / ≈8 CONFLICT，
逐类 P/R/F1 不可解释。协议 v1.1 因此新增两层，但把实现列为"下一轮"：

  S5 **RT / 引用定向** → DERIVE 候选：跨线程、**字符覆盖率 ≥ 0.90**（近似逐字复制）
      或 $q$ 带 `RT @…` 定向痕迹；
  S6 **同槽位取值互斥** → CONFLICT 候选：**量槽匹配**（量纲相同）+ 取值互斥
      （`e13_slot_extract.py` 实现；用户拍板 (a) 把**量槽抽取**纳入 E13）。

本脚本产出 **6 层标注集**：
  · S1–S4  **原样保留**（pair_id E13-0001…E13-0388 不变，先导真值仍可对齐）；
  · S5 / S6 追加（pair_id 续编 E13-0389…）。

产出
----
  E13_标注集_v2.csv        6 层（含分层元数据，**不得外发**）
  E13_盲版标注表_v2.csv    去泄露盲版（仅 pair_id/event/p_text/q_text + 待填列）
  E13_抽样清单_v2.json     抽样参数、层统计、子型分布（可复现）
"""

# --- path resolution (released artifact) -----------------------------------
# No absolute path is baked in.  KB_ROOT points at the directory holding the
# experiment folders; it defaults to this repository's root and can be
# overridden with an environment variable.
import os as _os
_KB_ROOT = _os.environ.get(
    "KB_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)


def _KB(*parts):
    """Join *parts* onto the knowledge-base root."""
    return _os.path.join(_KB_ROOT, *parts)
# ---------------------------------------------------------------------------
import os
import re
import csv
import json
import html
import random
import pickle
import collections

import e13_slot_extract as S

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = _KB('数据', 'PHEME')

RNG = random.Random(20261011)
N_PER_STRATUM = 97
CAP_PER_EVENT = 30
MAXLEN = 320
TW_SNOW = 1288834974657

# S5 参数
COV_MIN = 0.90          # 字符 3-gram 覆盖率下限
S5_MINLEN = 40          # 参与覆盖率比较的最短归一文本长度（滤掉短样板句）
TEMPLATE_MAX = 3        # 归一文本重复 > 该次数 ⇒ 视为模板/机器人消息，不进候选池
MAX_NODE_REUSE = 2      # 单个节点在单层内最多出现次数（防 hub 退化）
# S6：每 (量纲,事件) 组最多参与配对的节点数（控制组合爆炸）
S6_GROUP_CAP = 320

URL_RE = re.compile(r'http\S+|www\.\S+')
MENT_RE = re.compile(r'@[A-Za-z0-9_]+')
RT_RE = re.compile(r'^\s*rt\b[:\s]', re.I)

BLIND_COLS = ['pair_id', 'event', 'p_text', 'q_text',
              'label_relation', 'label_confidence', 'label_notes']
FULL_COLS = ['pair_id', 'stratum', 'subtype', 'event', 'p_thread', 'q_thread',
             'p_tweet_id', 'q_tweet_id', 'p_time', 'q_time', 'p_veracity', 'q_veracity',
             'p_text', 'q_text', 'system_prior', 'system_pred',
             'label_relation', 'label_relation_2', 'label_confidence', 'label_notes']


def clean(t):
    return ' '.join((t or '').split())[:MAXLEN]


def normtext(t):
    t = html.unescape(t or '')
    t = URL_RE.sub(' ', t)
    t = MENT_RE.sub(' ', t)
    t = RT_RE.sub(' ', t)
    return re.sub(r'[^a-z0-9]', '', t.lower())


def char3(s):
    return set(s[i:i + 3] for i in range(max(0, len(s) - 2)))


def coverage(a, b):
    """短者被长者覆盖的比例（3-gram 集合的包含率）。"""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return len(a & b) / len(a)


def ts_of(nid):
    return (int(nid) >> 22) + TW_SNOW


def main():
    print('[1/4] 装载 PHEME …')
    G = pickle.load(open(os.path.join(DATA, 'pheme_cascades.gpickle'), 'rb'))
    ver_raw = pickle.load(open(os.path.join(DATA, 'veracity_dict.pkl'), 'rb'))
    VER = {str(k): ({'true': 0, 'false': 1, 'unverified': 2}[v] if isinstance(v, str) else int(v))
           for k, v in ver_raw.items()}
    VNAME = {0: 'true', 1: 'false', 2: 'unverified', None: 'na'}

    # 节点表（全局）
    #   key = tweet_id(str) -> dict(text, thread, event, ts, veracity)
    NODES = {}
    for g in G:
        tid = str(g.graph.get('thread_id', ''))
        ev = g.graph.get('event', '?')
        v = VER.get(tid)
        for n, d in g.nodes(data=True):
            s = str(n)
            if not s.isdigit():
                continue
            t = clean(d.get('text'))
            if len(t) < 10:
                continue
            NODES[s] = {'text': t, 'thread': tid, 'event': ev,
                        'ts': ts_of(s), 'ver': VNAME.get(v, 'na')}
    NODES = {k: v for k, v in NODES.items() if len(v['text']) >= 10}
    print('    节点 %d ｜ 线程 %d ｜ 事件 %d'
          % (len(NODES), len(set(v['thread'] for v in NODES.values())),
             len(set(v['event'] for v in NODES.values()))))

    # ★ 模板/机器人文本卫生
    #   实测：`@X, 25 Favs! Your tweet has been favorited by 25 people. http://t.co/…`
    #   这类自动消息在去掉 URL 后**与自身逐字相同**，会在 S5 里刷出成百上千条
    #   coverage=1.000 的假"近似复制"，并在 S6 里以 `PERSON 25` 槽污染配对。
    #   判据：归一文本在语料中重复出现 > TEMPLATE_MAX 次 ⇒ 视为模板，**不进入候选池**。
    TEMPLATE_MAX = 3
    nz_all = collections.Counter(normtext(v['text']) for v in NODES.values())
    templates = set(nz for nz, c in nz_all.items() if c > TEMPLATE_MAX)
    n_before = len(NODES)
    NODES = {k: v for k, v in NODES.items() if normtext(v['text']) not in templates}
    print('    模板过滤：剔除 %d 个节点（归一文本重复 >%d 次）；剩 %d'
          % (n_before - len(NODES), TEMPLATE_MAX, len(NODES)))
    print('    模板样例：%s' % [t[:70] for t in list(templates)[:3]])

    # 量槽倒排可用的量纲：**只保留"实体计数"类**——
    #   DEAD / INJURED / HOSTAGE / SUSPECT / AIRPLANE / BUILDING。
    # 排除理由（已在报告中声明）：
    #   · PERSON —— 句法中心词而非被量化的度量（"one sick stupid person" ⇒ PERSON 1），语义过载；
    #   · POLICE —— 角色词，中心名词易混淆（"two police chiefs" ⇒ POLICE 2）；
    #   · YEAR   —— 多为日期/时长（"25 years"），不是实体计数。
    S6_CLASSES = ['DEAD', 'INJURED', 'HOSTAGE', 'SUSPECT', 'AIRPLANE', 'BUILDING']

    # 已用对（S1–S4）避免重复
    base_rows = list(csv.DictReader(open(os.path.join(BASE, 'E13_标注集.csv'), encoding='utf-8-sig')))
    used_pairs = set()
    for r in base_rows:
        used_pairs.add(tuple(sorted([r['p_tweet_id'], r['q_tweet_id']])))

    rows = []
    for r in base_rows:
        d = {k: r.get(k, '') for k in FULL_COLS}
        d['subtype'] = ''
        if not d.get('p_time'):
            d['p_time'] = str(ts_of(r['p_tweet_id']))
            d['q_time'] = str(ts_of(r['q_tweet_id']))
        rows.append(d)
    print('[2/4] 继承 S1–S4：%d 行（pair_id 不变）' % len(rows))

    # 按事件分组
    by_event = collections.defaultdict(list)
    for k, v in NODES.items():
        by_event[v['event']].append(k)

    # ---------------- S5：跨线程近似逐字复制 / RT 定向 ----------------
    print('[3/4] 生成 S5（覆盖率≥%.2f 的跨线程对）…' % COV_MIN)
    s5 = []
    for ev, ids in by_event.items():
        # 分块：前缀键 + 中点键
        buckets = collections.defaultdict(list)
        for k in ids:
            nz = normtext(NODES[k]['text'])
            if len(nz) < S5_MINLEN:
                continue
            buckets['A:' + nz[:50]].append(k)
            mid = len(nz) // 2
            buckets['B:' + nz[mid:mid + 50]].append(k)
        seen = set()
        for key, group in buckets.items():
            if len(group) < 2:
                continue
            if len(group) > 400:
                group = RNG.sample(group, 400)
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if NODES[a]['thread'] == NODES[b]['thread']:
                        continue          # 必须跨线程（同线程已被 S1 覆盖）
                    pr = tuple(sorted([a, b]))
                    if pr in seen or pr in used_pairs:
                        continue
                    ca, cb = char3(normtext(NODES[a]['text'])), char3(normtext(NODES[b]['text']))
                    cov = coverage(ca, cb)
                    if cov < COV_MIN:
                        continue
                    seen.add(pr)
                    p, q = (a, b) if NODES[a]['ts'] <= NODES[b]['ts'] else (b, a)
                    s5.append((cov, ev, p, q))
    # 覆盖率降序（越像越发 DERIVE）；**先打散再加复用上限**——
    #   若不限制，配对会退化成"一个 hub 节点 vs 所有人"（实测踩坑：同一节点霸占全部对）。
    RNG.shuffle(s5)
    s5.sort(key=lambda x: -x[0])
    used_node = collections.Counter()
    ev_cnt, s5_rows = collections.Counter(), []
    for cov, ev, p, q in s5:
        if len(s5_rows) >= N_PER_STRATUM:
            break
        if ev_cnt[ev] >= CAP_PER_EVENT:
            continue
        if used_node[p] >= MAX_NODE_REUSE or used_node[q] >= MAX_NODE_REUSE:
            continue
        rt = int(bool(RT_RE.match(NODES[q]['text'])) or (' rt ' in (' ' + NODES[q]['text'].lower() + ' ')))
        # ★ 子型：**"仅前导 @提及不同" vs 真·文本差异** —— 先导实测这条缝决定 S5 产量
        #   在 94.4%（判 DERIVE）与 16.7%（判 SUPPORT）之间浮动，必须显式区分。
        nz_p, nz_q = normtext(NODES[p]['text']), normtext(NODES[q]['text'])
        #   normtext 已去 @/URL ⇒ 若两者相等，差异只在 @/URL
        if nz_p == nz_q:
            sub = 'rt' if rt else 'mention_only'
        else:
            sub = 'rt' if rt else 'copy'
        d = _mk(len(rows) + 1, 'S5_cross_thread_verbatim', sub,
                NODES[p], NODES[q], p, q,
                note='coverage=%.3f%s%s' % (cov, ' + RT-prefix' if rt else '',
                                            ' + mention-only' if sub == 'mention_only' else ''))
        s5_rows.append(d)
        rows.append(d)
        used_node[p] += 1
        used_node[q] += 1
        ev_cnt[ev] += 1
    print('    S5 候选 %d → 采 %d' % (len(s5), len(s5_rows)))

    # ---------------- S6：量槽匹配 + 取值互斥 / 极性相反 ----------------
    print('[4/4] 生成 S6（同量槽取值互斥 / 极性相反）…')
    # 倒排：量纲 -> 节点
    inv = collections.defaultdict(list)
    for k, v in NODES.items():
        sl, _ = S.slots_of(v['text'])
        for c, vals in sl.items():
            if c.startswith('RAW:') or c not in S6_CLASSES:
                continue           # 只用正规**度量**量纲，保证量槽语义可解释
            inv[c].append((k, frozenset(vals)))
    print('    量纲倒排：%s' % {c: len(v) for c, v in sorted(inv.items(), key=lambda kv: -len(kv[1]))})

    s6 = []
    for c, items in inv.items():
        # 按事件分组
        byev = collections.defaultdict(list)
        for k, vals in items:
            byev[NODES[k]['event']].append((k, vals))
        for ev, lst in byev.items():
            if len(lst) > S6_GROUP_CAP:
                lst = RNG.sample(lst, S6_GROUP_CAP)
            n = len(lst)
            for i in range(n):
                ki, vi = lst[i]
                for j in range(i + 1, n):
                    kj, vj = lst[j]
                    if ki == kj or NODES[ki]['thread'] == NODES[kj]['thread']:
                        continue      # 跨线程独立断言
                    inter = vi & vj
                    if inter:
                        continue      # 同值 ⇒ 不互斥
                    pr = tuple(sorted([ki, kj]))
                    if pr in used_pairs:
                        continue
                    ni, nj = (vi, vj)
                    num_both = all(x.isdigit() for x in ni) and all(x.isdigit() for x in nj)
                    # 极性相反（同一量槽 statement 的一正一负）
                    pi = S._polarity(NODES[ki]['text'])[0]
                    pj = S._polarity(NODES[kj]['text'])[0]
                    if num_both:
                        kind = 'strict'
                    elif pi != pj:
                        kind = 'polarity'
                    else:
                        kind = 'vague'
                    p, q = (ki, kj) if NODES[ki]['ts'] <= NODES[kj]['ts'] else (kj, ki)
                    vp, vq = (ni, nj) if p == ki else (nj, ni)
                    s6.append((kind, c, ev, p, q, sorted(vp), sorted(vq)))

    # 优先 strict（最可判定），再 polarity，再 vague
    order = {'strict': 0, 'polarity': 1, 'vague': 2}
    s6.sort(key=lambda x: order[x[0]])
    avail = collections.Counter(x[0] for x in s6)
    print('    S6 候选子型可用量：%s' % dict(avail))

    # **按子型配额**采样（否则 strict 会挤掉全部其它子型，先导就测不到它们的产量）
    target = {'strict': 60, 'polarity': 20, 'vague': 17}
    quota = {k: min(t, avail.get(k, 0)) for k, t in target.items()}
    left = N_PER_STRATUM - sum(quota.values())
    if left > 0:
        quota['strict'] = min(avail.get('strict', 0), quota['strict'] + left)
    left = N_PER_STRATUM - sum(quota.values())
    if left > 0:
        quota['polarity'] = min(avail.get('polarity', 0), quota['polarity'] + left)
    left = N_PER_STRATUM - sum(quota.values())
    if left > 0:
        quota['vague'] = min(avail.get('vague', 0), quota['vague'] + left)

    ev_cnt = collections.Counter()
    used_node = collections.Counter()
    sub_cnt = collections.Counter()
    s6_rows = []
    for kind in ('strict', 'polarity', 'vague'):
        pool = [x for x in s6 if x[0] == kind]
        RNG.shuffle(pool)
        got = 0
        for kk, c, ev, p, q, vp, vq in pool:
            if got >= quota.get(kind, 0):
                break
            if ev_cnt[ev] >= CAP_PER_EVENT:
                continue
            if used_node[p] >= MAX_NODE_REUSE or used_node[q] >= MAX_NODE_REUSE:
                continue
            d = _mk(len(rows) + 1, 'S6_same_slot_conflicting', kind,
                    NODES[p], NODES[q], p, q,
                    note='slot=%s %s|%s' % (c, ','.join(vp), ','.join(vq)))
            s6_rows.append(d)
            rows.append(d)
            used_node[p] += 1
            used_node[q] += 1
            ev_cnt[ev] += 1
            sub_cnt[kind] += 1
            got += 1
    print('    S6 采 %d ｜ 子型 %s ｜ 配额 %s' % (len(s6_rows), dict(sub_cnt), quota))

    # ---------------- 落盘 ----------------
    with open(os.path.join(BASE, 'E13_标注集_v2.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FULL_COLS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print('[save] E13_标注集_v2.csv（%d 行）' % len(rows))

    blind = [{k: r.get(k, '') for k in BLIND_COLS} for r in rows]
    with open(os.path.join(BASE, 'E13_盲版标注表_v2.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=BLIND_COLS)
        w.writeheader()
        w.writerows(blind)
    print('[save] E13_盲版标注表_v2.csv（%d 行 × %d 列，已剥离全部泄露列）'
          % (len(blind), len(BLIND_COLS)))

    st = collections.Counter(r['stratum'] for r in rows)
    manifest = {
        'n_per_stratum_target': N_PER_STRATUM,
        'cap_per_event': CAP_PER_EVENT,
        'total': len(rows),
        'strata': dict(st),
        'S6_subtypes': dict(sub_cnt),
        'candidate_pool': {'S5_candidates_coverage_ge_0.90': len(s5),
                           'S6_candidates_slot_exclusive': len(s6),
                           'S6_candidate_subtypes': dict(avail)},
        'S6_quota': quota,
        'S5_params': {'coverage_min': COV_MIN, 'min_norm_len': S5_MINLEN,
                      'blocking': 'prefix-50 / midpoint-50 char-3gram buckets'},
        'S6_params': {'group_cap': S6_GROUP_CAP, 'canonical_only': True,
                      'slot_definition': '量槽 = (量纲 head-class, 取值集合)；'
                                         'T5「量的兼容性」作用层的可机械化投影'},
        'event_distribution': dict(collections.Counter(r['event'] for r in rows).most_common()),
        'label_set': ['DERIVE', 'SUPPORT', 'CONFLICT', 'UNRELATED'],
        'base_rows_preserved': 'S1–S4 的 388 行 pair_id（E13-0001…E13-0388）与 v1 完全一致',
        'hygiene': {
            'template_max_repeat': TEMPLATE_MAX,
            'n_nodes_dropped_as_template': n_before - len(NODES),
            'n_nodes_after_hygiene': len(NODES),
            's6_classes_used': S6_CLASSES,
            's6_person_excluded_reason': ('PERSON 是句法中心词而非被量化的度量，语义过载'
                                          '（"one sick stupid person" ⇒ PERSON 1），会引入大量假候选'),
        },
        'note': 'S5/S6 为**候选生成器**；其 CONFLICT/DERIVE **产量由先导标注实测**，不假设。',
    }
    with open(os.path.join(BASE, 'E13_抽样清单_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print('[save] E13_抽样清单_v2.json')
    print('\n层分布：%s' % dict(st))
    return manifest


def _mk(n, stratum, subtype, P, Q, pid, qid, note=''):
    return {
        'pair_id': 'E13-%04d' % n, 'stratum': stratum, 'subtype': subtype,
        'event': P['event'], 'p_thread': P['thread'], 'q_thread': Q['thread'],
        'p_tweet_id': pid, 'q_tweet_id': qid,
        'p_time': str(P['ts']), 'q_time': str(Q['ts']),
        'p_veracity': P['ver'], 'q_veracity': Q['ver'],
        'p_text': P['text'], 'q_text': Q['text'],
        'system_prior': note, 'system_pred': '',
        'label_relation': '', 'label_relation_2': '', 'label_confidence': '', 'label_notes': '',
    }


if __name__ == '__main__':
    main()
