# -*- coding: utf-8 -*-
"""
E13 — A4「谓词同一性」：**封闭谓词族表 + 机械化匹配器**

为什么要它
----------
判据 v1.5 的 A4 只给了一张**5 行反例清单（枚举、非穷尽）**。全量标注时若再现新族，
标注员无据可依 ⇒ A4 是判据里**唯一没有封闭**的条款。系统侧（`e13_claim_graph_v3.py`）
更彻底：v3 根本没有实现 A4。

本文件把 A4 **封闭化**为一个有限族表，并给出机械化匹配。

★ 核心设计：A4 = 把「量纲」细化为 (属性, 取值)
------------------------------------------------
现有量槽抽取器（`e13_slot_extract.py`）的量纲 `HOSTAGE` **太粗**——它把
「被扣押」(`hostages held`) 与「逃离」(`hostages escaped`) 归成同一类，
而 A4 恰恰要求把这两者分开（一个是静态、一个是状态转移）。同理 `SUSPECT`
把「存在嫌犯」与「嫌犯身份/背景」混为一类。**这就是 v3 做不到 A4 的根因。**

于是：`FAMILY` 是 `CANON` 的一个**严格细化**——
    HOSTAGE → CUSTODY_HELD / CUSTODY_EXITED
    SUSPECT → PERPETRATOR_EXISTS / PERPETRATOR_IDENTITY
    DEAD    → FATALITY
    INJURED → INJURY
其余量纲（PERSON/POLICE/YEAR/AIRPLANE/BUILDING）**不进族表**（A4 只作用于事实谓词）。

判定规则（可直接写进判据）
--------------------------
    same_family    两侧的**数值**绑定到同一族 ⇒ 同一槽位，继续 ③c T5
    diff_family    两侧都绑到族、但族不同 ⇒ **UNRELATED**（不同属性或不同动作/阶段）
    out_of_table   任一侧的数值绑不到族（含"整句无家族数值"）⇒ **UNRELATED**（兜底）

「未命中 ⇒ UNRELATED」是**封闭世界假设**，它把「判据漏了新族」变成**可观测事件**
（全量标注时登记 `pred-out-of-table`，而不是悄悄放过）。

守卫：**复用 `e13_slot_extract` 的六条计数合法性守卫**（不重造）——
计数语境 / 年份 / 间隔破坏词 / 混合字母数字串不拆 token / 现实量级上限 / toll 反向。

对外接口
--------
    family_slots_of(text, window=4) -> ({family: {vals}}, tk)
    family_mentions(text)           -> set(family)          # 不要求带数值
    predicate_relation(p, q)        -> (rel, detail)        # rel ∈ 三值
    self_test()                     -> (ok, failures)
    write_table_csv(path)           -> 谓词族表（供人判版引用）

产物：E13_谓词族表_v1.csv / E13_谓词匹配.json
"""
import os
import re
import csv
import json
import collections

import e13_slot_extract as SLOT

BASE = os.path.dirname(os.path.abspath(__file__))

# ================================================================ 封闭族表
# family_id -> (属性, 取值, 说明)
ATTR = {
    'FATALITY':            ('生命状态', '死亡', ''),
    'INJURY':              ('生命状态', '受伤', ''),
    'AMBIG_PERSON':        ('生命状态', '存殁未定', '上位词：victim/casualty 未说明死活 —— **已登记的缝**'),
    'CUSTODY_HELD':        ('拘押状态', '被扣押', '静态'),
    'CUSTODY_EXITED':      ('拘押状态', '脱离拘押', '状态转移：逃离/获释/被救'),
    'PERPETRATOR_EXISTS':  ('施害者', '存在/数量', ''),
    'PERPETRATOR_IDENTITY': ('施害者', '身份/背景/动机', ''),
    'FORCE_SHOTS':         ('武力', '枪击/开枪', ''),
    'FORCE_EXPLOSION':     ('武力', '爆炸', ''),
    'POLICE_OPERATION':    ('处置', '警方行动/围困', ''),
}

# 族优先级（决定"一个数被哪个谓词量化"）。两条排序律：
#   律 1（承 SLOT.CANON_PRIO）**度量类优先于泛化类/角色类**：`12 people were killed`
#        的量纲是 FATALITY 而非 PERSON。
#   律 2（本表新增，★ A4-V/N 规则）**状态转移谓词优先于静态状态的名词论元**：
#        `5 hostages escaped` 里 `escaped`（CUSTODY_EXITED）才是被断言的谓词，
#        `hostages`（CUSTODY_HELD）只是它的论元 ⇒ 取 EXITED。
#        这是 A4 能在系统侧落地的关键一步（v3 做不到 A4 的根因即量纲 HOSTAGE 未拆分）。
FAM_PRIO = ['FATALITY', 'INJURY', 'AMBIG_PERSON', 'CUSTODY_EXITED', 'CUSTODY_HELD',
            'PERPETRATOR_IDENTITY', 'PERPETRATOR_EXISTS', 'FORCE_SHOTS',
            'FORCE_EXPLOSION', 'POLICE_OPERATION']

# 表面形式（闭集，含屈折）——**只有列在这里的形态才被承认**
SURF = {
    'FATALITY': {
        'kill', 'killed', 'kills', 'killing', 'die', 'died', 'dies', 'dying',
        'dead', 'death', 'deaths', 'fatal', 'fatality', 'fatalities', 'slain',
        'murder', 'murdered', 'murders', 'corpse', 'corpses', 'body', 'bodies',
        'toll', 'massacre', 'massacred', 'perished', 'succumbed',
    },
    'INJURY': {
        'injure', 'injured', 'injures', 'injury', 'injuries', 'wound', 'wounded',
        'wounds', 'hurt', 'hurting', 'hospitalized', 'hospitalised', 'hospital',
        'critical', 'stable', 'condition', 'paramedics', 'medics', 'treatment',
        'treated', 'recovering',
    },
    # ★ 上位词：未说明死活 ⇒ 单列一族，不与 FATALITY/INJURY 归并（已登记的缝）
    'AMBIG_PERSON': {'victim', 'victims', 'casualty', 'casualties', 'affected'},
    'CUSTODY_HELD': {
        'hostage', 'hostages', 'captive', 'captives', 'kidnap', 'kidnapped',
        'kidnapping', 'abducted', 'abduction', 'seized', 'detained', 'detention',
        'custody', 'captured',
    },
    'CUSTODY_EXITED': {
        'escaped', 'escape', 'escapes', 'escaping', 'fled', 'flee', 'flees',
        'fleeing', 'released', 'freed', 'rescued', 'rescue', 'walked',
        'left', 'exit', 'exited', 'exiting',      # "5 hostages have left"（实测 E13-0540）
    },
    'PERPETRATOR_EXISTS': {
        'gunman', 'gunmen', 'shooter', 'shooters', 'suspect', 'suspects',
        'attacker', 'attackers', 'terrorist', 'terrorists', 'assailant',
        'assailants', 'killer', 'killers', 'perpetrator', 'perpetrators',
        'bomber', 'bombers', 'accomplice', 'accomplices', 'jihadist', 'jihadists',
        'militant', 'militants', 'extremist', 'extremists', 'hostage-taker',
        'hostage-takers', 'gunmen',
    },
    'PERPETRATOR_IDENTITY': {
        'named', 'identified', 'identity', 'background', 'motive', 'motives',
        'profile', 'convert', 'refugee', 'offender', 'radicalized', 'radicalised',
        'history', 'record', 'known', 'linked', 'tied',
    },
    'FORCE_SHOTS': {
        'gunfire', 'gunshot', 'gunshots', 'shooting', 'shootings', 'shootout',
        'shots', 'shot', 'gunshots',
    },
    'FORCE_EXPLOSION': {
        'explosion', 'explosions', 'blast', 'blasts', 'bomb', 'bombs', 'bombing',
        'bombings', 'detonated', 'grenade', 'shelling', 'airstrike', 'airstrikes',
    },
    'POLICE_OPERATION': {
        'standoff', 'standoffs', 'siege', 'sieges', 'lockdown', 'surrounded',
        'surround', 'stormed', 'raid', 'raids', 'evacuated', 'evacuate',
        'evacuation', 'deployed', 'cordon', 'cordoned', 'besieged',
    },
}

# 多 token 短语（结束于末 token）——列表里已有的**不重复**
PHRASE = {
    ('holed', 'up'): 'POLICE_OPERATION',
    ('let', 'go'): 'CUSTODY_EXITED',
    ('opened', 'fire'): 'FORCE_SHOTS',
    ('opening', 'fire'): 'FORCE_SHOTS',
    ('shot', 'dead'): 'FATALITY',
    ('found', 'dead'): 'FATALITY',
    ('confirmed', 'dead'): 'FATALITY',
    ('pronounced', 'dead'): 'FATALITY',
    ('lost', 'their'): 'FATALITY',      # lost their lives
    ('lost', 'his'): 'FATALITY',
    ('lost', 'her'): 'FATALITY',
    ('held', 'hostage'): 'CUSTODY_HELD',
    ('taken', 'hostage'): 'CUSTODY_HELD',
    ('taking', 'hostage'): 'CUSTODY_HELD',
    ('hostage', 'situation'): 'CUSTODY_HELD',
    ('hostage', 'situations'): 'CUSTODY_HELD',
    ('hostage', 'taking'): 'CUSTODY_HELD',
    ('burst', 'in'): 'FORCE_SHOTS',
    ('stable', 'condition'): 'INJURY',
    ('serious', 'condition'): 'INJURY',
}

TRIG2FAM = {}
for _f, _ws in SURF.items():
    for _w in _ws:
        TRIG2FAM.setdefault(_w, _f)

FAM_CAP = {'FATALITY': 5000, 'INJURY': 5000, 'AMBIG_PERSON': 100000,
           'CUSTODY_HELD': 500, 'CUSTODY_EXITED': 500,
           'PERPETRATOR_EXISTS': 200, 'PERPETRATOR_IDENTITY': 200,
           'FORCE_SHOTS': 5000, 'FORCE_EXPLOSION': 500, 'POLICE_OPERATION': 100}


def _fam_at(tk, j):
    """token j 处的族（含 2-token 短语，短语优先）。"""
    if j >= 1:
        fam = PHRASE.get((tk[j - 1][0], tk[j][0]))
        if fam:
            return fam
    return TRIG2FAM.get(tk[j][0])


# ---- 数字替代词守卫（实测：Twitter 上 `4 all`="for all"、`2 many`="too many" 会
#      凭空造出"枪击 4 起"这类假计数）------------------------------------------------
NUM_SUBST_ADJ = {'all', 'many', 'much', 'u', 'ur', 'you', 'me', 'my', 'him', 'her',
                 'them', 'us', 'we'}

# 隔断词：**在 SLOT.REJECT_BETWEEN 基础上只去掉 `in`**。
# 理由：族谓词多为**形容词/名词**（`stable (condition)`），系动词链 `is/are in X` 是常态
# （实测 `2 are in stable condition` 因 `in` 被整条丢弃）。去掉 `in` 不会放行时间/年份
# ——那两类由 `_is_count_context`（PRE_AGE）与 `_is_yearlike` 两道**独立**守卫拦住。
# ⚠️ 只在本族抽取器内生效，**不改 SLOT.REJECT_BETWEEN**（单变量纪律）。
FAM_REJECT = SLOT.REJECT_BETWEEN - {'in'}


def _num_substitution(i, tk):
    """单数字 token（1–9）紧邻代词/量词 ⇒ 判为拼写替代（4=for / 2=too），非计数。"""
    tok, isnum, _ = tk[i]
    if not (isnum and len(tok) == 1):
        return False
    for j in (i - 1, i + 1):
        if 0 <= j < len(tk) and tk[j][0] in NUM_SUBST_ADJ:
            return True
    return False


def family_slots_of(text, window=5):
    """抽取**族槽** {family: {取值}}：数值绑定到其**谓词族**（而非粗量纲）。

    与 SLOT.slots_of 的两处差别（均为 A4 落地所必需）：
      ① 类空间从 CANON 换成 FAMILY（严格细化——HOSTAGE 拆成 HELD/EXITED 等）；
      ② window 默认 5（族触发的**表征**更稀：`six hostages have managed to escape`
         的谓词与数字隔 5 个 token，窗口 4 会漏掉谓词而只绑到名词论元）。
    其余三重语义守卫（计数语境 / 间隔破坏词 / 族优先级+距离）**逐条复用 SLOT 的实现**，
    并新增**数字替代词守卫**（`4 all` / `2 many`）。
    """
    tk = SLOT._toks(text)
    if not tk:
        return {}, tk
    res = collections.defaultdict(set)
    for i in range(len(tk)):
        qval = SLOT._numval(i, tk)
        if qval is None:
            continue
        if SLOT._is_yearlike(qval):
            continue
        if not SLOT._is_count_context(i, tk):
            continue
        if _num_substitution(i, tk):
            continue
        best = None
        for off in range(1, window + 1):
            for j in (i + off, i - off):
                if not (0 <= j < len(tk)):
                    continue
                fam = _fam_at(tk, j)
                if not fam:
                    continue
                if j > i:                                  # 正向 NUM … FAM
                    mid = [w for w, _, _ in tk[i + 1:j]]
                    if any(w in FAM_REJECT for w in mid):
                        continue
                else:                                      # 反向 FAM … NUM
                    if tk[j][0] not in SLOT.TOLL_HEAD:
                        continue
                    mid = [w for w, _, _ in tk[j + 1:i]]
                    if any(w not in SLOT.TOLL_VB for w in mid):
                        continue
                cand = (FAM_PRIO.index(fam), off, fam)
                if best is None or cand < best:
                    best = cand
        if best is not None:
            fam = best[2]
            if qval.isdigit() and int(qval) > FAM_CAP.get(fam, 10 ** 9):
                continue                                   # 超现实量级 ⇒ 非该族计数
            res[fam].add(qval)
    return dict(res), tk


def family_mentions(text):
    """文本**提到**的族（不要求带数值）——只用于画像/诊断，不用于 A4 判定。"""
    tk = SLOT._toks(text)
    out = set()
    for j in range(len(tk)):
        f = _fam_at(tk, j)
        if f:
            out.add(f)
    return out


def same_attr(fa, fb):
    return ATTR[fa][0] == ATTR[fb][0]


def predicate_relation(p, q, window=5, p_slots=None, q_slots=None):
    """A4 机械化判定。返回 (rel, detail)。

    rel ∈ {'same_family'   两侧数值绑到同一族 ⇒ 同一槽位，继续 ③c T5
           'diff_family'   两侧都绑到族、但族不同 ⇒ UNRELATED（不同属性或不同动作/阶段）
           'not_applicable' 任一侧无数值族 ⇒ **A4 不适用**，交回 ④ §2.1 / ⑤ 兜底}

    ⚠️ `not_applicable` 是**必须**的第三态：S5 的 97 条纯转述对里 95 条两侧都没有数值族，
    若把"无数值"当作 UNRELATED，整层会被误杀（实测）。
    """
    sp = p_slots if p_slots is not None else family_slots_of(p, window)[0]
    sq = q_slots if q_slots is not None else family_slots_of(q, window)[0]
    inter = sorted(set(sp) & set(sq))
    detail = {'sp': {k: sorted(v) for k, v in sp.items()},
              'sq': {k: sorted(v) for k, v in sq.items()},
              'inter': inter}
    if not sp or not sq:
        detail['reason'] = 'no_family_slot_on_%s' % ('p' if not sp else 'q')
        return 'not_applicable', detail
    if inter:
        return 'same_family', detail
    detail['reason'] = 'same_attr_diff_value' if any(
        same_attr(a, b) for a in sp for b in sq) else 'diff_attr'
    return 'diff_family', detail


def predicate_gate(p, q, window=5):
    """A4 门的二值投影：pass（放行到 T5）/ block（⇒ UNRELATED）/ not_applicable。"""
    rel, d = predicate_relation(p, q, window)
    return {'same_family': 'pass', 'diff_family': 'block',
            'not_applicable': 'not_applicable'}[rel], d


# ================================================================ 自检
_SELFTEST = [
    # (text, 期望族, 期望取值)
    ('At least 11 dead in the Paris shooting', 'FATALITY', '11'),
    ('The 4 French cartoonists killed in Paris', 'FATALITY', '4'),
    ('40-50 hostages held by gunmen', 'CUSTODY_HELD', '40'),      # 量纲细化前会被当成 HOSTAGE
    ('Two gunmen holed up inside', 'PERPETRATOR_EXISTS', '2'),
    ('When the 5 hostages escaped, the gunman fled', 'CUSTODY_EXITED', '5'),
    ('Holding 6 hostages at the supermarket', 'CUSTODY_HELD', '6'),
    ('three hostage escaping from the Lindt cafe', 'CUSTODY_EXITED', '3'),
    ('2 hostage situations near Paris', 'CUSTODY_HELD', '2'),
    ('12 dead including 2 police', 'FATALITY', '12'),
    ('Hospital received 3 victims, 2 are in stable condition', 'INJURY', '2'),
    ('Three victims have been identified', 'AMBIG_PERSON', '3'),   # 上位词 ≠ FATALITY
    ('Looking for 3 suspects, one has been dealt with', 'PERPETRATOR_EXISTS', '3'),
    ('up to 50 hostages', 'CUSTODY_HELD', '50'),
    ('2 #CharlieHebdo suspects are holed up', 'PERPETRATOR_EXISTS', '2'),
    ('Total 5 hostages have left Lindt', 'CUSTODY_EXITED', '5'),
    ('Six hostages have managed to escape from the cafe', 'CUSTODY_EXITED', '6'),
    # 反例：不是计数
    ('by 9 ET, police chief says', None, None),
    ('In 2009, police beat a man', None, None),
    ('was 18 and assaulted a police officer', None, None),
    ('yes people are taking selfies as 4 all pray', None, None),     # 4=for
    ('2 many people r 2 busy working', None, None),                  # 2=too
]


def self_test():
    fails = []
    for text, fam, val in _SELFTEST:
        sl, _ = family_slots_of(text)
        if fam is None:
            if sl:
                fails.append((text, 'expected none, got %s' % sl))
            continue
        got = sl.get(fam, set())
        if val not in got:
            fails.append((text, 'expected %s:%s, got %s' % (fam, val, sl)))
    # 关键细化不可逆：HOSTAGE 不再是一个族
    assert 'HOSTAGE' not in FAM_PRIO
    return (not fails), fails


# ================================================================ 表落盘
def write_table_csv(path=None):
    path = path or os.path.join(BASE, 'E13_谓词族表_v1.csv')
    rows = []
    for f in FAM_PRIO:
        a, v, note = ATTR[f]
        forms = sorted(SURF[f])
        phrases = sorted('%s %s' % k for k, vv in PHRASE.items() if vv == f)
        rows.append({'family_id': f, 'attribute': a, 'value': v,
                     'priority': FAM_PRIO.index(f),
                     'n_single_forms': len(forms),
                     'n_phrases': len(phrases),
                     'single_forms': ' '.join(forms),
                     'phrases': ' | '.join(phrases),
                     'note': note})
    with open(path, 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['family_id', 'attribute', 'value',
                                           'priority', 'n_single_forms', 'n_phrases',
                                           'single_forms', 'phrases', 'note'])
        w.writeheader()
        w.writerows(rows)
    return path, rows


def main():
    ok, fails = self_test()
    print('[selftest] %s' % ('PASS' if ok else 'FAIL'))
    for f in fails:
        print('   ✗', f)
    path, rows = write_table_csv()
    print('[save] %s（%d 族）' % (os.path.basename(path), len(rows)))
    for r in rows:
        print('  %-22s %s/%s  单形=%-3d 短语=%-2d' % (
            r['family_id'], r['attribute'], r['value'], r['n_single_forms'], r['n_phrases']))
    res = {'selftest_ok': ok, 'selftest_fails': fails,
           'families': {r['family_id']: {'attribute': r['attribute'], 'value': r['value'],
                                         'priority': r['priority'],
                                         'n_single_forms': r['n_single_forms'],
                                         'n_phrases': r['n_phrases']} for r in rows},
           'fam_prio': FAM_PRIO, 'fam_cap': FAM_CAP,
           'note': ('A4 机械化 = 把量纲细化为 (属性,取值)；same_family⇒同一槽位；'
                    'diff_family/out_of_table⇒UNRELATED（封闭世界）')}
    with open(os.path.join(BASE, 'E13_谓词匹配.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print('[save] E13_谓词匹配.json')
    return res


if __name__ == '__main__':
    main()
