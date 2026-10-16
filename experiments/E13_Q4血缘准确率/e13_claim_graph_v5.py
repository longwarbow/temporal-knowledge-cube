# -*- coding: utf-8 -*-
"""
E13 — 声明图构造器 **v5**（同步判据 v1.7）

依据：人类盲标双遍结果（κ 0.394→0.496，判据有洞）+ 21 条对照案例诊断
（`E13_v17_诊断.json`）。v4 与人类在 18 条人类靶向中**同错 12 条**
⇒ 洞在判据文本与机械规则两层都真实，v1.7 同时修两层。

v5 = v4 + 8 个修订（对应诊断卡点 + 人类靶向；全部可回归审计）
----------------------------------------------------------------
| # | 修订 | 判据依据 | 修的案例 |
|---|---|---|---|
| M1 | **平行转述守卫**：rule 2/3 的高重叠 ⇒ DERIVE，但 p、q 之间**无 handle 痕迹**且 q 有**实质新增词 ≥3**（hashtag 不计入新增内容）⇒ 判据 v1.7 A5-新：共同来源平行转述 ⇒ **SUPPORT**（rule 31）。收尾收紧后不再误伤 hashtag 化复制 | A5 收口 | 0203/0399/0415/0050/0209/0212/0213 |
| M2 | **模糊量词下界化**：multiple=≥2、several=≥3…；模糊 vs 点值：点值 ≥ 下界 ⇒ 相容（rule 36 SUPPORT）；点值 < 下界 ⇒ **CONFLICT**（rule 37） | A2 修订 | 0560/0105/0579 |
| M3 | **A3-ⓓ 数值方向性**：点值 M vs 下界 N：M≥N ⇒ 相容（rule 32 SUPPORT）；M<N ⇒ **CONFLICT**（rule 33） | A3-ⓓ 新增（裁定 3 条平局） | 0698/0596/0601 |
| M4 | **总量互斥优先于子集释放**：`among N <类>` 显式断言总量 / 单值侧总量，两侧点值总量互斥 ⇒ **CONFLICT**（rule 34），子集结构不再放行 | A3-ⓑ 优先级 | 0735 |
| M5 | **同总体数值互斥去包含度门**：同槽类、双侧纯数值、互斥、无子集释放 ⇒ **CONFLICT**（rule 35），**不再要求词包含度 ≥0.20**（量纲/年份/计数语境守卫已在 SLOT 层完成） | A3-ⓑ | 0753/0687/0334/0270 |
| M6 | **隐含单数**：`a/an + 类名词` ⇒ 该类隐含值 1；`identified/named` + 单数类名词 ⇒ 隐含 1；`more than one` ⇒ 下界 2（非点值 1） | A4 缝④ + A3-ⓓ | 0334/0270/0579 |
| M7 | **序数计数**：`Nth <事件单位名词>` 断言"该单位已报到第 N 起"（计数本身是 as-of 演进的）；与另一侧同一事件单位名词的**点值计数**比较：相等 ⇒ SUPPORT、互斥 ⇒ **CONFLICT**（rule 38）。仅双侧同名词干时触发 | A3-ⓓ 方向性 | 0102 |
| M8 | **乱码数值守卫**：`20, dead 12 injured 57 Hostage` 这类"数值-逗号-关键词"串联 ⇒ 数结构不可信，不得据其断冲突 ⇒ **UNRELATED**（rule 52） | A3-ⓑ 守卫 | 0630 |

**修订 ② 的收紧（子集判定）**：总数 vs 部分 ②（下界侧 + 更小点值）现在要求
**小值侧是"真子项"**——其量纲头名词属于**另一个 CANON 类**（`4 cartoonists`：cartoonists=PERSON ≠ DEAD）；
若头名词与槽同类（`1 hostage`：hostage=HOSTAGE=槽类）⇒ **同一总体** ⇒ 不放行，
走 M3 方向性（1 < ≥5 ⇒ CONFLICT）。

对照项（不得漂移）：0281/0019/0139 + 判据正典 E13-0503/0487/0458/0482/0491/0534/0539/0543。

产物：E13_系统预测_v5_全量.csv / e13_claim_graph_v5.json
"""
import os
import re
import csv
import json
import collections

import e13_claim_graph_v3 as V3
import e13_claim_graph_v4 as V4
import e13_claim_graph_v2 as V2
import e13_slot_extract as SLOT
import e13_predicate_match as PM

BASE = os.path.dirname(os.path.abspath(__file__))
LAB = V3.V2.LAB

# ---------------------------------------------------------------- v1.7 词表
VAGUE_LB = {'multiple': 2, 'several': 3, 'many': 2, 'numerous': 3, 'various': 2,
            'dozens': 24, 'scores': 40, 'hundreds': 200, 'thousands': 2000,
            'couple': 2, 'plenty': 2}
LOWER_MARK_RE = re.compile(r'\b(at\s+least|more\s+than|over|no\s+fewer\s+than)\b', re.I)
UPPER_MARK_RE = re.compile(r'\b(up\s+to|fewer\s+than|less\s+than|under)\b', re.I)
HEDGE_RE = re.compile(r'\b(possible|possibly|apparently|reportedly|allegedly)\b', re.I)
MORE_THAN_ONE_RE = re.compile(r'\bmore\s+than\s+one\b', re.I)
ARTICLE_ONE_RE = re.compile(
    r'\b(?:a|an)\s+((?:\w+\s+){0,2}?)'
    r'(gunman|gunmen|shooter|shooters|suspect|suspects|hostage|hostages|captive|'
    r'attacker|bomber|killer|hostage\s+taker|gun\s?man)\b', re.I)
IDENTITY_ONE_RE = re.compile(r'\b(identified|named)\b', re.I)
TOTAL_ASSERT_RE = re.compile(r'\b(?:among|of)\s+(\d[\d,]*)\s+((?:\w+\s+){0,2}?)', re.I)
# M7（v1.7 收尾）：**序数计数**——`Nth <事件单位名词>` 断言"该单位已报到第 N 起"
# （本项目计数本身就是 as-of 演进的）；若另一侧给出同一事件单位名词的点值计数
# 且数值互斥 ⇒ CONFLICT；相等 ⇒ SUPPORT。仅当**双侧同名词干**时触发（单侧序数
# 不触发，避免把"second hostage situation"这类无关提及当作计数）。
ORDINAL_IDX = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
               'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10}
EVENT_UNIT_NOUNS = ['shootings?', 'incidents?', 'attacks?', 'explosions?', 'bombings?',
                    'blasts?', 'crashes?']
_UNIT2STEM = {'shooting': 'shooting', 'shootings': 'shooting',
              'incident': 'incident', 'incidents': 'incident',
              'attack': 'attack', 'attacks': 'attack',
              'explosion': 'explosion', 'explosions': 'explosion',
              'bombing': 'bombing', 'bombings': 'bombing',
              'blast': 'blast', 'blasts': 'blast',
              'crash': 'crash', 'crashes': 'crash'}
CLASS2FAM = {'DEAD': 'FATALITY', 'INJURED': 'INJURY', 'HOSTAGE': 'CUSTODY',
             'SUSPECT': 'PERPETRATOR_EXISTS', 'AIRPLANE': 'AIRPLANE',
             'BUILDING': 'BUILDING', 'PERSON': 'PERSON', 'POLICE': 'POLICE',
             'YEAR': 'YEAR'}


def _handles(t):
    return set(m.group(0).lower() for m in V2.MENT_RE.finditer(t or ''))


GENERIC_POP = {'people', 'person', 'persons', 'individual', 'individuals'}
EMOTION = {'horrible', 'terrible', 'sad', 'tragic', 'awful', 'ridiculous', 'wtf',
           'shocking', 'disgusting', 'prayers', 'thoughts', 'sick', 'insane',
           'crazy', 'unbelievable', 'horrifying', 'devastating', 'rip', 'omg'}
ROLE_TRIG = {'gunman', 'shooter', 'suspect', 'attacker', 'bomber', 'killer',
             'assailant', 'terrorist', 'hostage', 'captive'}
CONTENT_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
HASHTAG_RE = re.compile(r'#(\w+)', re.UNICODE)
URL_RE2 = re.compile(r'http\S+|www\.\S+')


def _lemma(w):
    """最小词形归一：复数 → 单数；gunmen → gunman。"""
    w = w.lower()
    if w == 'gunmen':
        return 'gunman'
    if w.endswith('men') and w[:-3] + 'man' in ROLE_TRIG:
        return w[:-3] + 'man'
    if w.endswith('ies'):
        return w[:-3] + 'y'
    if w.endswith('s') and w[:-1] in ROLE_TRIG | {'hostages', 'shooters', 'suspects'}:
        return w[:-1]
    if w.endswith('es') and w[:-2] in ROLE_TRIG:
        return w[:-2]
    return w


def _content_set(text):
    """内容词集：去 URL、去收件人 @handle、保留 #hashtag、数词归一、去停用词/情绪词。"""
    t = URL_RE2.sub(' ', text or '')
    t = re.sub(r'@[A-Za-z0-9_]+', ' ', t)          # 收件人/来源 handle 不算内容
    out = set()
    for m in HASHTAG_RE.finditer(t):
        out.add('#' + m.group(1).lower())
    t = re.sub(r'#\w+', ' ', t)
    for w in CONTENT_RE.findall(t):
        low = w.lower()
        if low.isdigit() or low in NUM_WORD_STR:
            out.add('n')          # 数值统一为占位（防 2/two、148/150 误判为"新增内容"）
            continue
        if low in V2.STOP or low in EMOTION or len(low) < 3:
            continue
        out.add(low)
    return out


NUM_WORD_STR = set('one two three four five six seven eight nine ten eleven twelve '
                   'thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty '
                   'dozen hundred thousand million'.split())


def _new_content_ratio(p, q):
    """q 相对 p 的新增内容词（含 hashtag）；手柄/情绪/数值已归一。返回 (n_new, ratio, new_set)"""
    cp, cq = _content_set(p), _content_set(q)
    new = cq - cp
    ratio = len(new) / max(1, len(cq))
    return len(new), ratio, new


def _new_word_count(p, q):
    """q 相对 p 的**实质新增词**（v1.7 收尾 / M1 收紧）：
    - hashtag 一律视为"话题标签"⇒ 不构成新增命题内容：q 的 hashtag 词并入 p 侧、
      q 的实质词只统计非 hashtag 词；
    - @handle、URL、数值、情绪词已剔除。
    用于区分「hashtag 化复制 / 追加标签的复制」（仍属 DERIVE）与「平行转述」（SUPPORT）。"""
    def _plain(t):
        t = URL_RE2.sub(' ', t or '')
        t = re.sub(r'@[A-Za-z0-9_]+', ' ', t)
        t = re.sub(r'#\w+', ' ', t)
        out = set()
        for w in CONTENT_RE.findall(t):
            low = w.lower()
            if low.isdigit() or low in NUM_WORD_STR:
                continue
            if low in V2.STOP or low in EMOTION or len(low) < 3:
                continue
            out.add(low)
        return out

    def _with_hash(t):
        out = _plain(t)
        for m in HASHTAG_RE.finditer(t or ''):
            w = m.group(1).lower()
            if len(w) >= 3 and w not in V2.STOP and w not in EMOTION:
                out.add(w)
        return out

    new = _plain(q) - _with_hash(p)
    return len(new), round(len(new) / max(1, len(_plain(q))), 3), new


def _order_break(p, q):
    """q 的共享内容词序列在 p 中是否**保持原序**（非 ⇒ 断裂）。"""
    cp_seq = []
    seen = set()
    for w in CONTENT_RE.findall(URL_RE2.sub(' ', re.sub(r'@[A-Za-z0-9_]+', ' ', p or '')).lower()):
        if w.isdigit() or w in NUM_WORD_STR:
            w = 'n'
        if w in V2.STOP or w in EMOTION or len(w) < 3:
            continue
        cp_seq.append(_lemma(w))
    cq_seq = []
    for w in CONTENT_RE.findall(URL_RE2.sub(' ', re.sub(r'@[A-Za-z0-9_]+', ' ', q or '')).lower()):
        if w.isdigit() or w in NUM_WORD_STR:
            w = 'n'
        if w in V2.STOP or w in EMOTION or len(w) < 3:
            continue
        cq_seq.append(_lemma(w))
    # 贪心按序匹配
    j = 0
    for w in cq_seq:
        while j < len(cp_seq) and cp_seq[j] != w:
            j += 1
        if j >= len(cp_seq):
            return True
        j += 1
    return False


def _val_matches(tk, i, val):
    """token i 的量化取值（数字或数词）是否等于 val（统一走 SLOT._numval）。"""
    try:
        return SLOT._numval(i, tk) == val
    except Exception:
        return False


def _find_val(tk, val):
    """返回所有取值为 val 的 token 下标。"""
    return [i for i in range(len(tk)) if _val_matches(tk, i, val)]


def _head_noun_after(text, val):
    """数值 val 后第一个非停用词；无 ⇒ ''。"""
    tk = SLOT._toks(text)
    for i in _find_val(tk, val):
        for j in range(i + 1, min(i + 3, len(tk))):
            w = tk[j][0]
            if w in V2.STOP:
                continue
            if SLOT._numval(j, tk) is not None:
                return ''
            return w
    return ''


def _pop_relation(text, cls, val):
    """数值 val 的名词短语与槽类 cls 的总体关系：
    'same'（同一总体）/ 'sub'（真子项）/ ''（无法判定，保守 same）。"""
    w = _head_noun_after(text, val)
    if not w:
        return ''
    if w in SLOT.CANON.get(cls, set()):
        return 'same'          # 头名词就是槽类触发词
    if w in GENERIC_POP:
        return 'same'          # 泛指总体（"148 people feared dead"）
    for c, trig in SLOT.CANON.items():
        if c != cls and w in trig:
            return 'sub'       # 头名词属于另一量纲类（如 police 之于 DEAD）
    if w.endswith('s'):
        return 'sub'           # 复数类别名词（"4 cartoonists"）＝ 总体的真子项
    return ''


def _value_markers(text, val):
    """数值 val 的方向/置信标记（含数词）。"""
    tk = SLOT._toks(text)
    for i in _find_val(tk, val):
        prev = [tk[k][0] for k in range(max(0, i - 3), i)]
        ctx = ' '.join(prev)
        m = {'lower': bool(LOWER_MARK_RE.search(ctx + ' ' + tk[i][0])),
             'upper': bool(UPPER_MARK_RE.search(ctx)),
             'hedge': bool(HEDGE_RE.search(ctx))}
        if MORE_THAN_ONE_RE.search(ctx + ' ' + tk[i][0]):
            m['lower'] = True
        return m
    return {'lower': False, 'upper': False, 'hedge': False}


def augment_side(text, cls, vals):
    """v1.7 M6：隐含单数 / more-than-one 修正。返回 (vals:set, info:dict)"""
    out = set(vals)
    info = {'implied': False, 'lower': set(), 'upper': set(), 'hedge': set()}
    # more than one ⇒ 下界 2（替换点值 1）
    if MORE_THAN_ONE_RE.search(text):
        out.discard('1')
        out.add('2')
        info['lower'].add('2')
    # a/an + 类名词 ⇒ 隐含 1
    for m in ARTICLE_ONE_RE.finditer(text):
        noun = m.group(2).lower().split()[-1]
        if noun in SLOT.CANON.get(cls, set()):
            out.add('1')
            info['implied'] = True
    # identified/named + 单数类名词 ⇒ 隐含 1
    if IDENTITY_ONE_RE.search(text):
        for trig in SLOT.CANON.get(cls, set()):
            if re.search(r'\b%s\b' % re.escape(trig), text, re.I) and not trig.endswith('s') \
                    and not re.search(r'\b%s s\b' % re.escape(trig[:-1]), text):
                out.add('1')
                info['implied'] = True
                break
    # 逐值方向标记
    for v in list(out):
        if v in VAGUE_LB:
            info['lower'].add(v)
            continue
        mk = _value_markers(text, v)
        if mk['lower']:
            info['lower'].add(v)
        if mk['upper']:
            info['upper'].add(v)
        if mk['hedge']:
            info['hedge'].add(v)
    return out, info


def asserted_totals(text, cls, vals):
    """该侧**显式断言的总量点值**：`among N <类>` 模式，或（单值、无方向/子项标记时）该值本身。"""
    trig = '|'.join(re.escape(t) for t in sorted(SLOT.CANON.get(cls, set()), key=len, reverse=True))
    if not trig:
        return set()
    out = set()
    for m in re.finditer(r'\b(?:among|of)\s+(\d[\d,]*)\s+((?:\w+\s+){0,2}?)(' + trig + r')\b',
                         text, re.I):
        out.add(str(int(float(m.group(1).replace(',', '')))))
    if not out and len(vals) == 1:
        only = next(iter(vals))
        if only.isdigit() and not PART_MARK_NEAR(text, only):
            mk = _value_markers(text, only)
            sub, _w = is_true_subitem(text, cls, only)
            if not mk['lower'] and not mk['hedge'] and not sub:
                out.add(only)
    return out


def PART_MARK_NEAR(text, val):
    tk = SLOT._toks(text)
    for i in _find_val(tk, val):
        ctx = ' '.join(tk[k][0] for k in range(max(0, i - 4), min(len(tk), i + 3)))
        return bool(re.search(r'\b(including|among|of\s+them|of\s+the|these|those)\b', ctx, re.I))
    return False


def is_true_subitem(text, cls, val):
    """v1.7 ② 收紧：小值侧是否**真子项**——定指 the N / 显式子集标记 /
       头名词属另一量纲类 / 复数类别名词。"""
    if re.search(r'\bthe\s+%s\b' % re.escape(val), text, re.I):
        return True, 'definite'
    if PART_MARK_NEAR(text, val):
        return True, 'part_mark'
    rel = _pop_relation(text, cls, val)
    if rel == 'sub':
        return True, 'head_noun:%s' % _head_noun_after(text, val)
    return False, ''


def _has_more_marker(text, val):
    """小值侧带 more/another/additional ⇒ 是同一总体的续计，不作互斥。"""
    tk = SLOT._toks(text)
    for i in _find_val(tk, val):
        prev = [tk[k][0] for k in range(max(0, i - 3), i)]
        return bool(set(prev) & {'more', 'another', 'additional', 'other', 'further'})
    return False


def _population_shift(text, cls, val):
    """数值 val 右侧出现 `of/among N <另一类名词>` ⇒ 该数值属于另一个总体（如 `4 dead of six hostages`）。"""
    tk = SLOT._toks(text)
    for i in _find_val(tk, val):
        ctx = [tk[k][0] for k in range(i + 1, min(len(tk), i + 7))]
        for j, w in enumerate(ctx):
            if w in ('of', 'among') and j + 1 < len(ctx):
                tail = ctx[j + 1:j + 5]
                for t in tail:
                    for c, trig in SLOT.CANON.items():
                        if c != cls and t in trig:
                            return True
        return False
    return False


ROLE_EQUIV = [{'gunman', 'shooter'}, {'hostage', 'captive'}]


def _same_role(a, b):
    if not a or not b:
        return False
    if a == b:
        return True
    return any(a in g and b in g for g in ROLE_EQUIV)


def _first_role_trigger(text, cls):
    """文本中出现的第一个该类角色触发词（单数优先）——隐含值的头名词回退。"""
    for trig in SLOT.CANON.get(cls, set()):
        if cls not in ('SUSPECT', 'HOSTAGE'):
            return ''
    cands = []
    for m in re.finditer(r'\b([a-z]+)\b', (text or '').lower()):
        w = m.group(1)
        if w in SLOT.CANON.get(cls, set()):
            cands.append(_lemma(w))
    if not cands:
        return ''
    sing = [c for c in cands if c == _lemma(c) and not c.endswith('s')]
    return (sing or cands)[0]


def _role_lemma_same(p, q, cls, val_p, val_q):
    """双侧数值的头名词是否为**同形角色名词**（shooter|gunman、hostages|hostage）。"""
    if cls not in ('SUSPECT', 'HOSTAGE'):
        return False
    hp = _lemma(_head_noun_after(p, val_p)) if val_p else ''
    hq = _lemma(_head_noun_after(q, val_q)) if val_q else ''
    if not hp:
        hp = _first_role_trigger(p, cls)
    if not hq:
        hq = _first_role_trigger(q, cls)
    return _same_role(hp, hq)


def _number_salad(text):
    """乱码守卫：`20, dead 12 injured 57 Hostage` 这类"数值-逗号-关键词"串联
    ⇒ 数结构不可信，数值层整体不适用（登记为 v1.7 新增守卫，E13-0630）。"""
    return bool(re.search(
        r'\d[\d,]*\s*,\s*(dead|injured|killed|wounded|hostage|hostages|shooters?|gunmen)\b',
        text or '', re.I))


def _ordinal_unit_claims(text):
    """`Nth <事件单位名词>` ⇒ {名词干: N}（E13-0102：`second shooting` ⇒ shooting=2）。"""
    out = {}
    pat = (r'\b(%s)\s+((?:\w+\s+){0,2}?)(%s)\b'
           % ('|'.join(ORDINAL_IDX), '|'.join(EVENT_UNIT_NOUNS)))
    for m in re.finditer(pat, text or '', re.I):
        k = ORDINAL_IDX[m.group(1).lower()]
        stem = _UNIT2STEM.get(m.group(3).lower(), m.group(3).lower())
        out.setdefault(stem, k)
    return out


def _count_unit_claims(text):
    """`<数字> <事件单位名词>` ⇒ {名词干: n}（E13-0102：`3 separate shootings` ⇒ shooting=3）。"""
    out = {}
    pat = r'\b(\d+)\s+((?:\w+\s+){0,2}?)(%s)\b' % '|'.join(EVENT_UNIT_NOUNS)
    for m in re.finditer(pat, text or '', re.I):
        stem = _UNIT2STEM.get(m.group(3).lower(), m.group(3).lower())
        out.setdefault(stem, int(m.group(1)))
    return out


def _v5_ordinal_pair(p, q, gate):
    """M7：序数计数 vs 点值计数（仅双侧同名词干）。返回 dict 或 None。"""
    op, oq = _ordinal_unit_claims(p), _ordinal_unit_claims(q)
    cp, cq = _count_unit_claims(p), _count_unit_claims(q)
    for stem in sorted(set(op) | set(oq)):
        if stem in op and stem in cq:
            a, b = op[stem], cq[stem]
        elif stem in oq and stem in cp:
            a, b = cp[stem], oq[stem]
        else:
            continue
        if not gate(a, b):
            continue
        verdict = 'SUPPORT' if a == b else 'CONFLICT'
        why = 'M7:%s=%d vs %d' % (stem, a, b)
        return {'cls': 'ORDINAL:' + stem, 'vals_p': [str(a)], 'vals_q': [str(b)],
                'aug_p': {}, 'aug_q': {}, 'verdict': verdict, 'rule': 38, 'why': why}
    return None


def v5_num_pair(p, q, pred_rel, contain, event_unit):
    """增强数值层（v1.7）：所有提升都必须在**双门**下发生——
    Tier1 文本关联（contain ≥ 0.20）或 Tier2 同形角色锚（shooter|shooter）；
    且 event_unit 命中时一律不提升。"""
    sp = SLOT.slots_of(p)[0]
    sq = SLOT.slots_of(q)[0]
    if _number_salad(p) or _number_salad(q):
        return None
    # M7：序数计数 vs 点值计数（仅双侧同事件单位名词；Tier1 门）
    if pred_rel in ('same_family', 'not_applicable'):
        od = _v5_ordinal_pair(p, q, lambda a, b: (not event_unit) and contain >= 0.20)
        if od is not None:
            return od
    best = None
    for cls in set(sp) | set(sq):
        vp_raw = set(sp.get(cls, set()))
        vq_raw = set(sq.get(cls, set()))
        if not vp_raw and not vq_raw:
            continue
        vp, ip = augment_side(p, cls, vp_raw)
        vq, iq = augment_side(q, cls, vq_raw)
        if not vp or not vq:
            continue                       # 单侧值 ⇒ 无从比较
        num_p = {v for v in vp if v.isdigit()}
        num_q = {v for v in vq if v.isdigit()}
        vag_p = {v for v in vp if v in VAGUE_LB}
        vag_q = {v for v in vq if v in VAGUE_LB}
        d = {'cls': cls, 'vals_p': sorted(vp), 'vals_q': sorted(vq),
             'aug_p': ip, 'aug_q': iq, 'verdict': None, 'rule': 0, 'why': ''}
        # --- 双门（v1.7 起，所有提升的前提）---
        def _gate(lo_v, hi_v):
            if event_unit:
                return False
            if contain >= 0.20:
                return True
            return _role_lemma_same(p, q, cls, lo_v, hi_v) or _role_lemma_same(p, q, cls, hi_v, lo_v)
        # 同侧多值 ⇒ 大值=总量、小值=子项（同侧蕴含）
        tot_p = max(num_p, key=int) if num_p else None
        tot_q = max(num_q, key=int) if num_q else None
        inter = num_p & num_q
        if inter:
            d.update(verdict='SAME', rule=0, why='same_value')
            best = best or d
            continue
        # --- M2 模糊 vs 点值 ---
        if vag_p and num_q:
            lb = VAGUE_LB[max(vag_p, key=VAGUE_LB.get)]
            pt = max(num_q, key=int)
            if not _gate(pt, pt):
                continue
            if int(pt) >= lb:
                d.update(verdict='SUPPORT', rule=36, why='vague%d<=%s' % (lb, pt))
            else:
                d.update(verdict='CONFLICT', rule=37, why='vague%d>%s' % (lb, pt))
            best = best or d
            continue
        if vag_q and num_p:
            lb = VAGUE_LB[max(vag_q, key=VAGUE_LB.get)]
            pt = max(num_p, key=int)
            if not _gate(pt, pt):
                continue
            if int(pt) >= lb:
                d.update(verdict='SUPPORT', rule=36, why='vague%d<=%s' % (lb, pt))
            else:
                d.update(verdict='CONFLICT', rule=37, why='vague%d>%s' % (lb, pt))
            best = best or d
            continue
        if vag_p and vag_q:
            # 模糊 vs 模糊：仅在同形角色锚下提升
            if not (not event_unit and _role_lemma_same(p, q, cls, '', '')):
                continue
            d.update(verdict='SUPPORT', rule=36, why='vague_vague')
            best = best or d
            continue
        if not num_p or not num_q:
            continue
        # --- 双侧纯数值 ---
        # 对每对 (vp, vq) 逐一判；顺序：同值 ⇒ 子项释放 ⇒ 方向性 ⇒ 点值互斥
        conflict_pairs, compat_pairs = [], []
        for a in num_p:
            for b in num_q:
                if int(a) == int(b):
                    compat_pairs.append((a, b, 'same'))
                    continue
                lo_s, hi_s = (a, b) if int(a) < int(b) else (b, a)
                # --- 子项释放（先于方向性；v1.7 ② 收紧后的正典路径）---
                lo_is_min_p = (lo_s == min(num_p, key=int) if lo_s in num_p else False)
                lo_is_min_q = (lo_s == min(num_q, key=int) if lo_s in num_q else False)
                if lo_is_min_p:
                    sub_lo, w1 = is_true_subitem(p, cls, lo_s)
                    if sub_lo:
                        compat_pairs.append((a, b, 'subitem_p:%s' % w1))
                        continue
                if lo_is_min_q:
                    sub_lo_q, w2 = is_true_subitem(q, cls, lo_s)
                    if sub_lo_q:
                        compat_pairs.append((a, b, 'subitem_q:%s' % w2))
                        continue
                # --- 方向性（M3 / A3-ⓓ）---
                lo_lower = lo_s in ip['lower'] or lo_s in iq['lower']
                hi_lower = hi_s in ip['lower'] or hi_s in iq['lower']
                if lo_lower or hi_lower:
                    if hi_lower:                      # hi 侧是下界 ⇒ 点值 lo 与 lb 比较
                        if int(lo_s) >= int(hi_s):
                            compat_pairs.append((a, b, 'lb_hi_ge'))
                        else:
                            conflict_pairs.append((a, b, 'M3:point%s<lb%s' % (lo_s, hi_s)))
                    else:                              # lo 侧是下界 ⇒ hi ≥ lb 恒兼容
                        compat_pairs.append((a, b, 'lb_lo'))
                else:
                    conflict_pairs.append((a, b, 'point_exclusive'))
        # --- M4：总量互斥优先（需双门 + 无 more 续计 + 无 of-总体转移）---
        at_p = asserted_totals(p, cls, vp)
        at_q = asserted_totals(q, cls, vq)
        if at_p and at_q and not (at_p & at_q):
            ap = sorted(at_p, key=int)[0]
            aq = sorted(at_q, key=int)[0]
            if (_gate(ap, aq) and not _has_more_marker(p, ap) and not _has_more_marker(q, aq)
                    and not _population_shift(p, cls, ap) and not _population_shift(q, cls, aq)):
                d.update(verdict='CONFLICT', rule=34,
                         why='M4:totals %s vs %s' % (sorted(at_p), sorted(at_q)))
                best = best or d
                continue
            # 门未过 ⇒ 不清空 compat/conflict，继续常规判定
        if conflict_pairs and not compat_pairs:
            c = conflict_pairs[0]
            if _gate(c[0], c[1]) and not _has_more_marker(p, c[0]) and not _has_more_marker(q, c[0]) \
                    and not _population_shift(p, cls, c[0]) and not _population_shift(q, cls, c[0]):
                d.update(verdict='CONFLICT', rule=35, why='M5:' + c[2])
                best = best or d
                continue
        if compat_pairs and not conflict_pairs:
            c = compat_pairs[0]
            if _gate(c[0], c[1]):
                d.update(verdict='SUPPORT',
                         rule=32 if any(x[2].startswith('lb') for x in compat_pairs) else 0,
                         why=c[2])
                best = best or d
                continue
        best = best or d
    if best is None:
        return None
    fam_ok = pred_rel in ('same_family', 'not_applicable')
    if not fam_ok:
        return None
    return best


def features(p, q):
    f = V4.features(p, q)
    f['handle_trace'] = int(bool(_handles(p) & _handles(q))
                            or bool(re.search(r'\brt\b\s*@', q, re.I))
                            or bool(re.search(r'\bvia\s+@', q, re.I)))
    f['head_attrib_p'] = int(bool(re.match(r'^["\'“”\s]*@\w+\s*:', p or '')))
    f['report_verb'] = int(bool(re.search(
        r'\b(authorities?|police|officials?|experts?|witnesses?|media|prosecutors?)\b'
        r'[^.]{0,40}\b(say|says|said|confirm|confirms|confirmed|report|reports|reported)\b',
        (p or '') + ' || ' + (q or ''), re.I)))
    # v1.7：p 是否为"转述体"（内容源头在第三方）+ q 是否新增内容/次序断裂
    f['p_transmission'] = int(f['head_attrib_p']
                              or bool(re.search(r'\bvia\s+@', p or '', re.I))
                              or bool(re.search(
                                  r'\b(authorities?|police|officials?|experts?|witnesses?|media)\b'
                                  r'[^.]{0,40}\b(say|says|said|confirm|confirms|confirmed)\b',
                                  p or '', re.I))
                              or bool(re.search(
                                  r'@\w+\s+(is\s+)?(report|reports|reporting|confirms?|says?)',
                                  p or '', re.I)))
    nn, ratio, new_set = _new_content_ratio(p, q)
    f['new_content_n'] = nn
    f['new_content_ratio'] = round(ratio, 3)
    f['new_content'] = '|'.join(sorted(new_set))[:120]
    nw, wratio, _ = _new_word_count(p, q)
    f['new_word_n'] = nw
    f['new_word_ratio'] = wratio
    f['order_break'] = int(_order_break(p, q))
    f['num_salad'] = int(_number_salad(p) or _number_salad(q))
    np_ = v5_num_pair(p, q, f['pred_rel'], f['contain'], f['event_unit'])
    f['v5_num_verdict'] = np_['verdict'] if np_ else ''
    f['v5_num_rule'] = np_['rule'] if np_ else 0
    f['v5_num_why'] = np_['why'] if np_ else ''
    return f


def predict(f, struct_edge):
    lab, rule = V3.predict(f, struct_edge)
    if f['pred_block'] and lab in ('CONFLICT', 'SUPPORT'):
        return 'UNRELATED', 51
    # M8（v1.7 收尾）：乱码数值守卫——`20, dead 12 injured 57 Hostage` 这类数结构
    #   不可信 ⇒ 不得据其断冲突（E13-0630，判为 UNRELATED）
    if f.get('num_salad') and lab == 'CONFLICT':
        return 'UNRELATED', 52
    # M1 平行转述守卫（v1.7 收尾收紧）：**实质新增词** ≥3（hashtag 不计入新增内容）
    #   ⇒ 不可能是纯复制 ⇒ SUPPORT；或 p 是转述体且 q 的共享内容次序断裂 ⇒ SUPPORT
    if rule in (2, 3) and lab == 'DERIVE':
        fire = (f['new_word_n'] >= 3 and f['new_content_ratio'] >= 0.12) \
            or (f['p_transmission'] and f['order_break'])
        if fire:
            return 'SUPPORT', 31
    # 数值增强层
    v, nr = f['v5_num_verdict'], f['v5_num_rule']
    if v == 'CONFLICT' and nr in (33, 34, 35, 37, 38):
        return 'CONFLICT', nr
    if v == 'SUPPORT' and lab == 'UNRELATED':
        return 'SUPPORT', nr or 36
    return lab, rule


_SELFTEST_REAL = [
    # (pair_id, 期望) —— 人类盲标 19 条靶向（真实文本）
    ('E13-0735', 'CONFLICT'), ('E13-0560', 'CONFLICT'), ('E13-0334', 'CONFLICT'),
    ('E13-0270', 'CONFLICT'), ('E13-0753', 'CONFLICT'), ('E13-0102', 'CONFLICT'),
    ('E13-0086', 'UNRELATED'), ('E13-0630', 'UNRELATED'),
    ('E13-0297', 'SUPPORT'), ('E13-0105', 'SUPPORT'), ('E13-0140', 'SUPPORT'),
    ('E13-0579', 'SUPPORT'), ('E13-0698', 'SUPPORT'),
    ('E13-0203', 'SUPPORT'), ('E13-0399', 'SUPPORT'), ('E13-0415', 'SUPPORT'),
    ('E13-0596', 'CONFLICT'), ('E13-0601', 'CONFLICT'), ('E13-0687', 'CONFLICT'),
    # 对照（v4 已对，不得漂移）
    ('E13-0281', 'UNRELATED'), ('E13-0019', 'SUPPORT'), ('E13-0139', 'UNRELATED'),
]


def _selftest():
    rows = {r['pair_id']: r for r in csv.DictReader(
        open(os.path.join(BASE, 'E13_标注集_v3.csv'), encoding='utf-8-sig'))}
    ok = True
    for pid, want in _SELFTEST_REAL:
        r = rows[pid]
        f = features(r['p_text'], r['q_text'])
        struct_edge = (r.get('system_prior') or '').startswith('structural')
        got, rule = predict(f, struct_edge)
        flag = 'OK ' if got == want else 'FAIL'
        if got != want:
            ok = False
        print('%s %s want=%-9s got=%-9s rule=%-3s num=%-8s why=%s' % (
            flag, pid, want, got, rule, f['v5_num_verdict'], f['v5_num_why'][:36]))
    return ok


def main():
    src = os.path.join(BASE, 'E13_标注集_v3.csv')
    rows = list(csv.DictReader(open(src, encoding='utf-8-sig')))
    print('读入 %d 对（%s）' % (len(rows), os.path.basename(src)))
    out = []
    for r in rows:
        struct_edge = (r.get('system_prior') or '').startswith('structural')
        f = features(r['p_text'], r['q_text'])
        ps, rs = predict(f, struct_edge)
        out.append(dict(r, **f, struct_edge=int(struct_edge),
                        system_pred=ps, system_pred_rule=rs))
    cols = ['pair_id', 'stratum', 'subtype', 'event', 'struct_edge',
            'j_tok', 'contain', 'char3_j', 'pol_opposite', 'rt_v3', 'src_q_only',
            'event_unit', 'a3_subset', 'slot_cls', 'slot_kind',
            'pred_rel', 'pred_fam_p', 'pred_fam_q', 'pred_inter', 'pred_block',
            'handle_trace', 'head_attrib_p', 'report_verb', 'num_salad',
            'v5_num_verdict', 'v5_num_rule', 'v5_num_why',
            'system_pred', 'system_pred_rule', 'p_text', 'q_text']
    with open(os.path.join(BASE, 'E13_系统预测_v5_全量.csv'), 'w',
              encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(out)
    print('[save] E13_系统预测_v5_全量.csv')
    return out


if __name__ == '__main__':
    print('===== v1.7 自检（靶向 13 + 对照 2）=====')
    ok = _selftest()
    print('自检%s' % ('通过' if ok else '未通过'))
