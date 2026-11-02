# -*- coding: utf-8 -*-
"""
E13 — $Q_4$ 血缘准确率：**量槽（slot）抽取库 + 画像**

背景（为什么需要它）
--------------------
先导标注证明：**不做槽位匹配就抽不到 CONFLICT 样本**（S3「veracity 相反」层的
冲突率实测为 **0/12**）。协议 v1.1 因此新增 **S6 同槽位取值互斥** 层，
但 S6 依赖「主体-谓词」抽取——协议 v1.0 §8 把它列为范围外。
用户 2026-10-11 拍板 **(a) 纳入 E13**，本文件即该实现的**可复用部分**。

槽位的诚实定义（重要）
----------------------
标注指南 v1.3 把**槽位键**定义为 $(主体 \times 谓词 \times 时点)$，**量不属于键**，
并新增 **T5「量的兼容性」**。在英文推文上，完整 $(s,p,t)$ 抽取不可靠；
本库只抽取**槽位键的一个可机械化投影**——

    **量槽 = (量纲 head-class, 取值集合)**

即「N 个 X」里的 $X$（量纲）与 $N$（取值）。这正是 T5 判"能否同时为真"所作用的层。
本库因此**不声称**完成 $(s,p,t)$ 抽取，只声称完成 **T5 作用层的投影抽取**；
S6 是**候选生成器**，其 CONFLICT **产量（yield）由先导标注实测**，不靠假设。

对外接口
--------
    slots_of(text) -> {量纲: {取值, ...}}
    match_slots(p, q) -> list of SlotMatch(量纲, vals_p, vals_q, kind)
        kind ∈ {'num_exclusive' 双侧纯数值且交集空,
                'vague_exclusive' 至少一侧为模糊量词且交集空,
                'same' 交集非空}
    quantity_conflict(p, q) -> (量纲, vals_p, vals_q, strict_bool) | None
    polarity_conflict(p, q) -> (量纲, ...) | None

产出
----
    E13_槽位抽取.json   —— 覆盖率、量纲分布、示例；供报告与论文引用
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
import pickle
import collections

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = _KB('数据', 'PHEME')

# ---------------------------------------------------------------- 词汇表
NUM_WORD = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
    'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
    'hundred': 100, 'thousand': 1000, 'million': 1000000, 'dozen': 12,
}
# 模糊量词：与任何具体数值**语义上可兼容**（T5 的难点正落在这里）
VAGUE = {
    'multiple', 'several', 'many', 'scores', 'dozens', 'hundreds', 'thousands',
    'numerous', 'various', 'few', 'couple', 'some', 'numerous', 'plenty',
}

# 量纲词典：canonical class -> 触发词（含复数与屈折）
CANON = {
    'DEAD': {'dead', 'death', 'deaths', 'killed', 'kill', 'kills', 'killing',
             'fatalities', 'fatality', 'fatal', 'died', 'die', 'dies', 'body',
             'bodies', 'corpse', 'corpses', 'victim', 'victims', 'toll',
             'murdered', 'slain', 'casualty', 'casualties'},
    'INJURED': {'injured', 'injury', 'injuries', 'wounded', 'wound', 'wounds',
                'hurt', 'hospitalized', 'hospitalised', 'hospital', 'critical',
                'injure', 'hurting'},
    'HOSTAGE': {'hostage', 'hostages', 'captive', 'captives', 'abducted',
                'abduction', 'seized', 'detained'},
    'SUSPECT': {'suspect', 'suspects', 'gunman', 'gunmen', 'attacker',
                'attackers', 'shooter', 'shooters', 'killer', 'killers',
                'perpetrator', 'perpetrators', 'terrorist', 'terrorists',
                'bomber', 'bombers', 'accomplice', 'accomplices', 'assailant',
                'assailants'},
    'PERSON': {'person', 'persons', 'people', 'man', 'men', 'woman', 'women',
               'individual', 'individuals', 'citizen', 'citizens', 'user',
               'users', 'student', 'students'},
    'POLICE': {'police', 'officer', 'officers', 'cop', 'cops', 'agent',
               'agents', 'trooper', 'troopers'},
    'YEAR': {'year', 'years'},
    'AIRPLANE': {'plane', 'planes', 'aircraft', 'flight', 'flights', 'jet',
                 'jets'},
    'BUILDING': {'building', 'buildings', 'house', 'houses', 'storey',
                 'storeys', 'floor', 'floors'},
}
TRIG2CLASS = {}
for _c, _ws in CANON.items():
    for _w in _ws:
        TRIG2CLASS.setdefault(_w, _c)
        if _w.endswith('s'):
            TRIG2CLASS.setdefault(_w[:-1], _c)

# 量纲优先级：**度量类优先于泛化类**——"12 people were killed" 的量纲是 DEAD
# （people=PERSON 是句法中心词，killed 才是被量化的度量），故 PERSON 置最低。
CANON_PRIO = ['DEAD', 'INJURED', 'HOSTAGE', 'SUSPECT', 'AIRPLANE', 'POLICE',
              'BUILDING', 'YEAR', 'PERSON']
ZERO_PREV = {'no', 'not', 'none', 'zero', 'nobody'}

# ---------------- 计数合法性守卫（精度关键） ----------------
# 实测踩坑：`by 9 ET, police chief says` / `In 2009, #Ferguson police beat a man` /
# `was 18 and assaulted a police officer` 都曾被判成 POLICE 的计数槽 —— 数字与
# 量纲词只是**共现**，不是"数字修饰量纲"。以下守卫把"数字"限定为**真计数**：
TIME_MARK = {'et', 'am', 'pm', 'oclock', 'utc', 'gmt', 'est', 'pst', 'edt', 'cst',
             'mst', 'bst', 'aest', 'hrs', 'hr', 'hour', 'hours', 'minute', 'minutes',
             'second', 'seconds', 'st', 'nd', 'rd', 'th'}
UNIT = {'usd', 'eur', 'gbp', 'dollar', 'dollars', 'cents', 'kg', 'km', 'mi',
        'miles', 'mph', 'ft', 'feet', 'meter', 'meters', 'metre', 'metres',
        'percent', 'pct', 'k', 'm', 'bn'}
PRE_AGE = {'in', 'since', 'from', 'until', 'till', 'by', 'year', 'years', 'age',
           'aged', 'old', 'circa', 'about', 'around', 'before', 'after'}
# 数字与量纲词之间的**破坏性**间隔词（介词/连词/时间标记）⇒ 非计数
REJECT_BETWEEN = (PRE_AGE | TIME_MARK | UNIT |
                  {'and', 'but', 'or', 'that', 'which', 'who', 'whom', 'whose',
                   'when', 'while', 'then', 'also', 'still', 'even', 'only',
                   'with', 'of', 'for', 'on', 'off', 'into', 'onto', 'as',
                   'because', 'so', 'if', 'though', 'although'})
# "死亡人数升到 12" 型**反向**模式：量纲词在前、数值在后，只允许报告型动词间隔
TOLL_HEAD = {'toll', 'death', 'deaths', 'number', 'total', 'fatalities', 'fatality',
             'injuries', 'injured', 'dead', 'victims', 'victim', 'casualties',
             'damage', 'figure', 'figures', 'count'}
TOLL_VB = {'has', 'have', 'had', 'is', 'was', 'were', 'been', 'now', 'risen',
           'rose', 'rises', 'rise', 'climbed', 'climb', 'climbs', 'increased',
           'increase', 'increases', 'up', 'to', 'at', 'of', 'the', 'a', 'an',
           'reached', 'reach', 'reaches', 'stands', 'stand', 'put', 'revised',
           'revise', 'confirmed', 'confirm', 'set', 'jumped', 'jump', 'soared',
           'more', 'than', 'least', 'over', 'about', 'around', 'nearly', 'almost',
           'least', 'isn'}


def _numval(i, tk):
    """token i 的量化取值；'no one'/'not one' -> 0；非量词返回 None。"""
    tok, isnum, val = tk[i]
    if isnum:
        return val
    if tok in NUM_WORD:
        if i > 0 and tk[i - 1][0] in ZERO_PREV:
            return '0'
        if tok == 'one' and i > 0 and tk[i - 1][0].endswith("n't"):
            return '0'
        return str(NUM_WORD[tok])
    if tok in VAGUE:
        return tok
    return None


def _is_yearlike(v):
    """1900–2100 的 4 位数几乎总是年份而非计数（实测最大的假阳性来源）。"""
    return v.isdigit() and 1900 <= int(v) <= 2100


def _is_count_context(i, tk):
    """数字 i 是否处在"计数"语境（而非时间/年龄/单位/年份）。"""
    if i + 1 < len(tk) and tk[i + 1][0] in (TIME_MARK | UNIT):
        return False
    if i > 0 and tk[i - 1][0] in PRE_AGE:
        return False
    return True

NEG = set("""not n't no never none nobody nothing neither nor without
deny denies denied denial false fake fakes hoax hoaxed falsely debunk debunked
untrue fabricated fabricate wrong incorrect disprove disproven refute refuted
misinformation misleading conspiracy nonsense lie lies lied rumour rumor myth
staged photoshopped doctored""".split())

URL_RE = re.compile(r'http\S+|www\.\S+')
MENT_RE = re.compile(r'@[A-Za-z0-9_]+')
RT_RE = re.compile(r'^\s*rt\b[:\s]', re.I)
# ★ 必须整体匹配字母数字串：否则 `b4`（=before）、`1st`、`2day` 会被切成
#   'b'+'4' / '1'+'st'，凭空造出数字 4 / 1（实测在 S6 里刷出大量假冲突）。
TOK_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
NUMTOK_RE = re.compile(r'^\d[\d,]*(?:\.\d+)?$')
ALPHA_RE = re.compile(r'^[A-Za-z]+$')

# 计数现实量级上限：超过则不是该量纲的计数（"24 wounds" 不是受伤人数）
COUNT_CAP = {'DEAD': 5000, 'INJURED': 5000, 'HOSTAGE': 500, 'SUSPECT': 200,
             'AIRPLANE': 100, 'BUILDING': 1000, 'POLICE': 5000, 'YEAR': 200,
             'PERSON': 100000}

STOP = set("""a an the and or but if then than so because as of to in on at by for with from into over under
this that these those it its is are was were be been being am do does did done doing have has had having
i you he she we they me him her us them my your his their our mine yours theirs
will would shall should can could may might must shall
there here what which who whom whose when where why how
all any both each few more most other some such no nor not only own same too very
just now also get got go going went make makes made say says said rt via amp
u ur im dont cant wont isnt arent wasnt didnt doesnt thats whats lets
""".split())


def _toks(text):
    """返回 [(tok_lower, is_number, value)]；混合串（b4 / 1st）既非数字也非词。"""
    t = html.unescape(text or '')
    t = URL_RE.sub(' ', t)
    t = MENT_RE.sub(' ', t)
    t = RT_RE.sub(' ', t)
    out = []
    for m in TOK_RE.finditer(t):
        s = m.group(0)
        low = s.lower()
        if NUMTOK_RE.match(s):
            try:
                v = int(float(s.replace(',', '')))
            except Exception:
                continue
            out.append((low, True, str(v)))
        elif ALPHA_RE.match(s):
            out.append((low, False, None))
        # 混合串：丢弃（防 b4 -> 4）
    return out


def slots_of(text, window=4, classes=None):
    """抽取**量槽** {量纲: {取值}}。

    规则（三重守卫，见上方 TIME_MARK/UNIT/PRE_AGE/REJECT_BETWEEN 注释）：
      ① 数字必须处在**计数语境**（非时间/年龄/单位/年份）；
      ② 与量纲触发词的间隔内不得有介词/连词/时间标记（反向"toll 型"另用白名单动词）；
      ③ 量纲按**度量类优先于泛化类**（DEAD > … > PERSON），再按距离最近。
    找不到量纲触发词时退化为最近内容名词（`RAW:` 前缀），仅画像用，候选生成不用。
    """
    tk = _toks(text)
    if not tk:
        return {}, tk
    cand_pool = classes or CANON_PRIO
    res = collections.defaultdict(set)
    for i in range(len(tk)):
        qval = _numval(i, tk)
        if qval is None:
            continue
        if _is_yearlike(qval):
            continue                      # 年份不是计数
        if not _is_count_context(i, tk):
            continue
        best = None
        for off in range(1, window + 1):
            for j in (i + off, i - off):
                if not (0 <= j < len(tk)):
                    continue
                c = TRIG2CLASS.get(tk[j][0])
                if not c or c not in cand_pool:
                    continue
                if j > i:                 # 正向：NUM … CLASS
                    mid = [w for w, _, _ in tk[i + 1:j]]
                    if any(w in REJECT_BETWEEN for w in mid):
                        continue
                else:                     # 反向：CLASS … NUM，只允许 toll 型报告动词
                    if tk[j][0] not in TOLL_HEAD:
                        continue
                    mid = [w for w, _, _ in tk[j + 1:i]]
                    if any(w not in TOLL_VB for w in mid):
                        continue
                rank = cand_pool.index(c)
                cand = (rank, off, c)
                if best is None or cand < best:
                    best = cand
        if best is not None:
            c = best[2]
            if qval.isdigit() and int(qval) > COUNT_CAP.get(c, 10 ** 9):
                continue          # 超现实量级 ⇒ 不是该量纲的计数
            res[c].add(qval)
            continue
        # 退化：最近的内容名词
        if classes:
            continue                      # 候选生成模式不用 RAW
        for off in range(1, window + 1):
            done = False
            for j in (i + off, i - off):
                if 0 <= j < len(tk):
                    w = tk[j][0]
                    if (not tk[j][1]) and w not in STOP and len(w) > 2 \
                            and w not in NUM_WORD and w not in VAGUE:
                        res['RAW:' + (w[:-1] if w.endswith('s') and len(w) > 4 else w)].add(qval)
                        done = True
                        break
            if done:
                break
    return dict(res), tk


def _polarity(text):
    tk = _toks(text)
    c = sum(1 for w, isn, _ in tk if (not isn) and (w in NEG or w.endswith("n't")))
    return (1 if c > 0 else 0), c


class SlotMatch(object):
    __slots__ = ('cls', 'vals_p', 'vals_q', 'kind')

    def __init__(self, cls, vals_p, vals_q, kind):
        self.cls, self.vals_p, self.vals_q, self.kind = cls, vals_p, vals_q, kind

    def as_dict(self):
        return {'cls': self.cls, 'vals_p': sorted(self.vals_p),
                'vals_q': sorted(self.vals_q), 'kind': self.kind}


def _numeric_side(vals):
    """返回 (全为纯数值?, 数值集合)"""
    nums = set(v for v in vals if v.isdigit())
    return (len(nums) == len(vals) and len(nums) > 0), nums


def match_slots(p, q, window=4):
    """返回 p,q 之间所有"同量纲"的匹配（含同值）。"""
    sp = slots_of(p, window)[0]
    sq = slots_of(q, window)[0]
    out = []
    for c in set(sp) & set(sq):
        vp, vq = sp[c], sq[c]
        inter = vp & vq
        if inter:
            out.append(SlotMatch(c, vp, vq, 'same'))
            continue
        np_, npv = _numeric_side(vp)
        nq_, nqv = _numeric_side(vq)
        if np_ and nq_:
            out.append(SlotMatch(c, vp, vq, 'num_exclusive'))
        else:
            out.append(SlotMatch(c, vp, vq, 'vague_exclusive'))
    return out


def quantity_conflict(p, q, window=4):
    """同量槽 + 取值互斥 ⇒ (量纲, vals_p, vals_q, strict)。strict = 双侧纯数值。

    只返回**最强的**一条（优先 strict，其次量纲优先级 DEAD>INJURED>HOSTAGE>…）。
    """
    ms = match_slots(p, q, window)
    excl = [m for m in ms if m.kind in ('num_exclusive', 'vague_exclusive')]
    if not excl:
        return None
    def key(m):
        strict = 0 if m.kind == 'num_exclusive' else 1
        pr = CANON_PRIO.index(m.cls) if m.cls in CANON_PRIO else len(CANON_PRIO)
        return (strict, pr)
    excl.sort(key=key)
    m = excl[0]
    strict = (m.kind == 'num_exclusive')
    return (m.cls, sorted(m.vals_p), sorted(m.vals_q), strict)


def polarity_conflict(p, q, window=4):
    """同量槽 + 两侧极性相反 ⇒ (量纲, vals_p, vals_q)。"""
    pol_p, _ = _polarity(p)
    pol_q, _ = _polarity(q)
    if pol_p == pol_q:
        return None
    for m in match_slots(p, q, window):
        return (m.cls, sorted(m.vals_p), sorted(m.vals_q))
    return None


# ---------------------------------------------------------------- 画像 CLI
def _clean(t, n=320):
    return ' '.join((t or '').split())[:n]


def main():
    print('[1/2] 装载 PHEME …')
    G = pickle.load(open(os.path.join(DATA, 'pheme_cascades.gpickle'), 'rb'))
    texts = []
    per_event = collections.Counter()
    for g in G:
        ev = g.graph.get('event', '?')
        for n, d in g.nodes(data=True):
            if not str(n).isdigit():
                continue
            t = d.get('text')
            if t and len(t.strip()) >= 10:
                texts.append((ev, _clean(t)))
                per_event[ev] += 1
    print('    节点文本 %d ｜ 事件 %d' % (len(texts), len(per_event)))

    print('[2/2] 抽取量槽 …')
    cls_cnt = collections.Counter()
    cls_cnt_raw = collections.Counter()
    n_with_slot = 0
    n_with_num_slot = 0
    slot_sizes = []
    examples = []
    for ev, t in texts:
        s, _ = slots_of(t)
        if s:
            n_with_slot += 1
            slot_sizes.append(len(s))
            has_num = False
            for c, vals in s.items():
                cls_cnt_raw[c] += 1
                if not c.startswith('RAW:'):
                    cls_cnt[c] += 1
                if any(v.isdigit() for v in vals):
                    has_num = True
            if has_num:
                n_with_num_slot += 1
            if len(examples) < 12 and ('DEAD' in s or 'INJURED' in s):
                examples.append({'event': ev, 'text': t, 'slots': {k: sorted(v) for k, v in s.items()}})

    N = len(texts)
    res = {
        'n_texts': N,
        'n_with_any_slot': n_with_slot,
        'n_with_numeric_slot': n_with_num_slot,
        'pct_with_any_slot': round(100.0 * n_with_slot / N, 2),
        'pct_with_numeric_slot': round(100.0 * n_with_num_slot / N, 2),
        'mean_slots_per_text': round(sum(slot_sizes) / len(slot_sizes), 3) if slot_sizes else None,
        'canonical_class_counts': dict(cls_cnt.most_common()),
        'raw_class_top20': dict(cls_cnt_raw.most_common(20)),
        'events': dict(per_event.most_common()),
        'examples': examples,
        'definition': ('量槽 = (量纲 head-class, 取值集合)，是标注指南 v1.3 槽位键 '
                       '(主体×谓词×时点) 在 T5「量的兼容性」作用层上的**可机械化投影**；'
                       '本库不声称完成完整 (s,p,t) 抽取。'),
        'params': {'window': 4, 'num_words': len(NUM_WORD), 'vague': sorted(VAGUE),
                   'canonical_classes': {k: sorted(v) for k, v in CANON.items()}},
    }
    with open(os.path.join(BASE, 'E13_槽位抽取.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('[save] E13_槽位抽取.json')
    print('  含任意量槽 %.2f%% ｜ 含数值量槽 %.2f%% ｜ 平均 %s 槽/贴'
          % (res['pct_with_any_slot'], res['pct_with_numeric_slot'],
             res['mean_slots_per_text']))
    print('  正规量纲分布：%s' % dict(cls_cnt.most_common(10)))

    # 自检：合成对 + 真实假阳性回归
    print('\n[自检]')
    cases = [
        ("Police say 12 people were killed in the attack.", "At least 3 people were killed, reports say.", 'num_exclusive/DEAD'),
        ("12 dead in Paris shooting.", "12 dead in Paris shooting.", 'same'),
        ("Multiple suspects are still at large.", "3 suspects have been arrested.", 'vague_exclusive/SUSPECT'),
        ("No one was killed in the blast.", "Two people were killed in the blast.", 'polarity/DEAD'),
        ("The plane crashed in the Alps.", "Weather was bad that morning.", 'none'),
        ("BREAKING: 1 dead, 2 injured in Ottawa.", "Police confirm 3 dead in Ottawa.", 'num_exclusive/DEAD'),
        ("Name of #Ferguson cop is expected to be released by 9 ET, police chief says", "x", 'FP-guard: 9 ET 非计数'),
        ("In 2009, #Ferguson police beat a man and charged him", "x", 'FP-guard: 2009 是年份'),
        ("was 18 and assaulted a police officer", "x", 'FP-guard: 18 是年龄'),
        ("The death toll has risen to 12", "The death toll has risen to 8", 'toll 反向模式'),
        ("Just heard two police chiefs", "x", 'POLICE 类已停用'),
    ]
    for p, q, tag in cases:
        print('  [%-22s] P=%s' % (tag, slots_of(p, classes=['DEAD', 'INJURED', 'HOSTAGE', 'SUSPECT', 'AIRPLANE', 'POLICE', 'BUILDING'])[0]))
        if q != 'x':
            print('  %-24s Q=%s  -> qconf=%s pconf=%s'
                  % ('', slots_of(q, classes=['DEAD', 'INJURED', 'HOSTAGE', 'SUSPECT', 'AIRPLANE', 'POLICE', 'BUILDING'])[0],
                     quantity_conflict(p, q), polarity_conflict(p, q)))
    return res


if __name__ == '__main__':
    main()
