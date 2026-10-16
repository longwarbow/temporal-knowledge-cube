# -*- coding: utf-8 -*-
"""
E15 / Phase 0 —— 残余缝探针：R2-b「同事件不同面」与 R4「同槽类不同指称」的实测体量
-----------------------------------------------------------------------------
问题（方法论㉞）：缝目前只有「破例计数」（R2-b 6/755、R4 27/755），
缺一个独立实测：**一个命题级对齐探针能不能把缝判对**。
  能  => 通道可修（进入 Phase 0b：事件共指通道）
  不能=> 保持标界，负结果本身就是「超出现有廉价通道」的实证

设计
  1) 合成探针试错（10 条，先验已知）—— 方法论①：先合成再上真实数据
  2) 真实缝集
     R2-b 缝 = E13_v9_回归.json 中 rule 39 触发且 v6 != 真值 的 6 条
     R2-b 对照 = 同 rule 39 触发但 v6 == 真值 的 18 条（机制相同、结果相反 ⇒ 最佳对照）
     R4  缝 = S6 层 v5 == CONFLICT 且 v1.7r 真值 == UNRELATED（27 条）
     R4  对照 = S6 层 v5 == CONFLICT 且 v1.7r 真值 == CONFLICT（17 条）
  3) 四个通道（两遍措辞 × 两版极性口径）
       A1 四分类 / B1 二值直问        —— 口径 v1：否定 ⇒ 非同命题（初版，有定义错位）
       A2 四分类 / B2 二值直问        —— 口径 v2：**同一命题的相反断言 = 同一命题**（对齐判据：
                                        CONFLICT 恰是「同一命题 + 相反极性」，非 UNRELATED）
     二值判定 = (verdict == SAME_PROPOSITION)
  4) 判定规则（方案 §2.2）
     缝判别率 >= 80% 且 对照保真率 >= 80% => 通道可修；否则 => 保持标界。用 Youden J 兼报净判别力。

用法：
  python e15_p0_seam_probe.py            # 全跑（4 通道）
  python e15_p0_seam_probe.py --skip-synth
"""
import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
E13 = os.path.abspath(os.path.join(BASE, "..", "E13_Q4血缘准确率"))
ROOT = os.path.abspath(os.path.join(BASE, "..", "..", ".."))   # 10.keyan

SYS = (
    "You are an expert annotator for a rumour-analysis benchmark. "
    "You compare two short social-media assertions (A and B) and judge their relation "
    "AT THE PROPOSITION LEVEL, not by surface similarity. "
    "Everything inside a news event looks similar; your job is to see through that."
)

# ---------------- 口径 v1：否定 ⇒ 非同命题（初版） ----------------
PROMPT_A1 = """Assertion A (earlier): "{p}"
Assertion B (later):   "{q}"

Choose exactly one verdict:

- SAME_PROPOSITION : A and B assert the SAME proposition — same subject/actor, same predicate,
  same referent (the same entity, incident or quantity), and compatible polarity. A paraphrase,
  a more/less detailed phrasing, or a report of the same fact by another source all count.
- SAME_EVENT_DIFF_FACET : same event, but a DIFFERENT proposition — a different property is
  asserted (e.g. "N hostages held" vs "one hostage escaped"; "X was killed" vs "X was a reservist").
  Different facets of one event do NOT support each other.
- DIFF_REFERENT_SAME_SLOT : same slot type (same actor+predicate+time pattern) but a DIFFERENT
  referent — a different incident, location or sub-event (e.g. two different shootings).
- UNRELATED : different events, or no propositional link.

Rules:
- Quantity/scale, identity, location and casualty-count are FACETS. Equal-sounding numbers do not
  make two assertions the same proposition if the referent differs.
- Polarity matters: an assertion and its denial are not the same proposition.

Answer ONLY with compact JSON:
{{"verdict":"<one of the four>","confidence":<0-1>,"why":"<=20 words"}}"""

PROMPT_B1 = """Assertion A (earlier): "{p}"
Assertion B (later):   "{q}"

Question: do A and B assert the SAME proposition?

Answer YES only if they state the same fact about the same referent, so that believing A
gives you a reason to believe B. Being about the same event is NOT enough: a different
property (scale, identity, casualty count, a later development), a different incident with
a similar shape, or a denial of A all mean NO.

Answer ONLY with compact JSON: {{"same":true|false,"confidence":<0-1>,"why":"<=20 words"}}"""

# ---------------- 口径 v2：同一命题的相反断言 = 同一命题（对齐判据） ----------------
_POLICY_V2 = """
**Crucial definition.** "Same proposition" is about WHICH FACT IS AT STAKE, not about whether
the two assertions agree. If A says "13 hostages" and B says "50 hostages", they are the SAME
proposition with contradictory values (a conflict) — NOT different propositions. Likewise
"there was one gunman" and "there were two gunmen" are the same proposition under dispute.
So: contradicting a claim about the same referent still counts as SAME_PROPOSITION."""

PROMPT_A2 = """Assertion A (earlier): "{p}"
Assertion B (later):   "{q}"

Choose exactly one verdict:
- SAME_PROPOSITION : A and B put the SAME fact at stake — same subject/actor, same predicate,
  same referent (same entity, same incident, same quantity being measured). They may agree
  (support) or contradict each other (conflict); either way it is the same proposition.
- SAME_EVENT_DIFF_FACET : same event, but a DIFFERENT fact is at stake — a different property
  (e.g. "N hostages held" vs "one hostage escaped"; "X was killed" vs "X was a reservist").
- DIFF_REFERENT_SAME_SLOT : same slot type but a DIFFERENT referent — a different incident,
  location or sub-event (e.g. two different shootings in the same city).
- UNRELATED : different events, or no propositional link.
""" + _POLICY_V2 + """
Answer ONLY with compact JSON:
{{"verdict":"<one of the four>","confidence":<0-1>,"why":"<=20 words"}}"""

PROMPT_B2 = """Assertion A (earlier): "{p}"
Assertion B (later):   "{q}"

Question: do A and B put the SAME fact at stake (same subject, same predicate, same referent)?
""" + _POLICY_V2 + """
A different property of the same event (scale, identity, casualty count, a later development),
a different incident with a similar shape, or mere topical overlap all mean NO.
Disagreeing about the value of the same quantity still means YES.
Answer ONLY with compact JSON: {{"same":true|false,"confidence":<0-1>,"why":"<=20 words"}}"""

PASSES = [("A1", "verdict", PROMPT_A1, "四分类·口径v1"),
          ("B1", "same",    PROMPT_B1, "二值直问·口径v1"),
          ("A2", "verdict", PROMPT_A2, "四分类·口径v2(极性修正)"),
          ("B2", "same",    PROMPT_B2, "二值直问·口径v2(极性修正)")]

SYNTH = [
    ("Two more hostages have just escaped from the Sydney cafe.",
     "Another hostage flees the scene of the Sydney siege.", "SAME_PROPOSITION", "同命题改写"),
    ("The Germanwings flight crashed in the French Alps.",
     "Germanwings plane crashed in France.", "SAME_PROPOSITION", "同命题精简"),
    ("Police confirmed that three people died in the shooting.",
     "Three people were killed, police confirm.", "SAME_PROPOSITION", "同命题被动改写"),
    ("Up to 20 people are being held hostage in the Lindt Cafe.",
     "Five hostages have escaped the besieged Lindt Cafe.", "SAME_EVENT_DIFF_FACET", "规模×逃走：不同面"),
    ("Two gunmen are inside the cafe.",
     "Police are negotiating with the gunmen inside the cafe.", "SAME_EVENT_DIFF_FACET", "存在×谈判：不同面"),
    ("The plane was carrying 150 people.",
     "The airline has grounded its entire fleet.", "SAME_EVENT_DIFF_FACET", "事故×应对：不同面"),
    ("Three shots were fired at the Ottawa War Memorial.",
     "Several shots were fired inside the Canadian Parliament building.", "DIFF_REFERENT_SAME_SLOT", "同槽不同地点"),
    ("A man was shot dead in the Paris suburb of Saint-Denis.",
     "Two people were shot in central Paris this morning.", "DIFF_REFERENT_SAME_SLOT", "同槽不同事件"),
    ("Heavy rain caused flooding across the city.",
     "The parliament passed a new budget yesterday.", "UNRELATED", "无关"),
    ("The suspect was arrested at the airport.",
     "Ticket prices rose five percent this year.", "UNRELATED", "无关"),
]


def load_env():
    p = os.path.join(ROOT, ".env_e14")
    env = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def make_client(env):
    from openai import OpenAI
    return OpenAI(base_url=env["E14_LLM_BASE_URL"], api_key=env["E14_LLM_API_KEY"], timeout=60.0)


def norm(v):
    if isinstance(v, bool):
        return "SAME_PROPOSITION" if v else "NOT_SAME_PROPOSITION"
    v = str(v).strip().upper().replace(" ", "_")
    if v in ("YES", "TRUE"):
        return "SAME_PROPOSITION"
    if v in ("NO", "FALSE"):
        return "NOT_SAME_PROPOSITION"
    return v


def ask(client, model, p, q, template, key, retries=4):
    msg = [{"role": "system", "content": SYS},
           {"role": "user", "content": template.format(p=p.strip(), q=q.strip())}]
    last = None
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=msg, temperature=0.0, max_tokens=120)
            txt = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.S)
            if not m:
                last = {"v": "PARSE_FAIL", "why": txt[:80]}
                continue
            obj = json.loads(m.group(0))
            raw = obj.get(key, obj.get("verdict", obj.get("same")))
            if raw is None:                       # 四分类模板被问成布尔时
                raw = obj.get(list(obj.keys())[0])
            return {"v": norm(raw), "why": str(obj.get("why", ""))[:120],
                    "conf": obj.get("confidence")}
        except Exception as e:                    # noqa: BLE001
            last = {"v": "ERR", "why": f"{type(e).__name__}: {e}"[:80]}
            time.sleep(1.5 * (i + 1))
    return last or {"v": "ERR", "why": ""}


def read_csv(name):
    return list(csv.DictReader(open(os.path.join(E13, name), encoding="utf-8-sig")))


def build_sets():
    ann = {r["pair_id"]: r for r in read_csv("E13_标注集_v3.csv")}
    truth = {r["pair_id"]: (r["truth"] or "").strip().upper()
             for r in read_csv("E13_全量真值_v5.csv")}
    v5 = {r["pair_id"]: (r["system_pred"] or "").strip().upper()
          for r in read_csv("E13_系统预测_v5_全量.csv")}
    v9 = json.load(open(os.path.join(E13, "E13_v9_回归.json"), encoding="utf-8"))
    flips = v9["flips"]
    seam_r2b = [f["pair_id"] for f in flips
                if str(f["v6_rule"]) == "39" and f["v6"] != f["truth"]]
    ctrl_r2b = [f["pair_id"] for f in flips
                if str(f["v6_rule"]) == "39" and f["v6"] == f["truth"]]
    S6 = [k for k, r in ann.items() if r["stratum"] == "S6_same_slot_conflicting"]
    seam_r4 = [k for k in S6 if v5[k] == "CONFLICT" and truth[k] == "UNRELATED"]
    ctrl_r4 = [k for k in S6 if v5[k] == "CONFLICT" and truth[k] == "CONFLICT"]
    return ann, truth, {"R2B_SEAM": seam_r2b, "R2B_CTRL": ctrl_r2b,
                        "R4_SEAM": seam_r4, "R4_CTRL": ctrl_r4}


def run_synth(client, model):
    rows = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {}
        for tag, key, tpl, lab in PASSES:
            for j, (a, b, _, _) in enumerate(SYNTH):
                futs[(tag, j)] = ex.submit(ask, client, model, a, b, tpl, key)
        for j, (a, b, want, why) in enumerate(SYNTH):
            r = {"p": a, "q": b, "want": want, "note": why}
            for tag, key, tpl, lab in PASSES:
                res = futs[(tag, j)].result()
                r[tag] = res["v"]
                r[tag + "_ok"] = (res["v"] == "SAME_PROPOSITION") == (want == "SAME_PROPOSITION")
                r[tag + "_exact"] = res["v"] == want
            rows.append(r)
    print("  合成探针（want = 设计答案；binary_ok = 判定口径是否正确）")
    for r in rows:
        line = "    %-24s" % r["want"]
        for tag, _, _, _ in PASSES:
            line += " %s:%-22s%s" % (tag, r[tag], "Y" if r[tag + "_ok"] else "n")
        print(line)
    summary = {tag: {"exact": sum(r[tag + "_exact"] for r in rows),
                     "binary": sum(r[tag + "_ok"] for r in rows), "n": len(rows)}
               for tag, _, _, _ in PASSES}
    print("  ——", {k: f"精确{v['exact']}/{v['n']} 二值{v['binary']}/{v['n']}" for k, v in summary.items()})
    return rows, summary


def run_sets(client, model, ann, truth, sets):
    out = {}
    for name, ids in sets.items():
        items = [(i, ann[i]["p_text"], ann[i]["q_text"]) for i in ids if i in ann]
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {}
            for tag, key, tpl, lab in PASSES:
                for j, (_, p, q) in enumerate(items):
                    futs[(tag, j)] = ex.submit(ask, client, model, p, q, tpl, key)
            res = []
            for j, (i, p, q) in enumerate(items):
                rec = {"pair_id": i, "stratum": ann[i]["stratum"], "truth": truth[i],
                       "p": p, "q": q}
                for tag, key, tpl, lab in PASSES:
                    r = futs[(tag, j)].result()
                    rec[tag] = r["v"]
                    rec[tag + "_why"] = r["why"]
                res.append(rec)
        out[name] = res
        print("  %-9s n=%-3d " % (name, len(res)) +
              " ".join("%s判同=%-3d" % (tag, sum(1 for r in res if r[tag] == "SAME_PROPOSITION"))
                       for tag, _, _, _ in PASSES))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-synth", action="store_true")
    args = ap.parse_args()

    env = load_env()
    model = env.get("E14_LLM_MODEL", "glm-4-flash")
    client = make_client(env)
    print("模型：%s @ %s\n" % (model, env["E14_LLM_BASE_URL"]))
    report = {"model": model, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}

    if not args.skip_synth:
        print("[1] 合成探针试错")
        rows, summary = run_synth(client, model)
        report["synth"] = {"rows": rows, "summary": summary}

    print("\n[2] 真实缝集 / 对照集")
    ann, truth, sets = build_sets()
    print("  集合规模：", {k: len(v) for k, v in sets.items()})
    res = run_sets(client, model, ann, truth, sets)
    report["real"] = res
    report["set_sizes"] = {k: len(v) for k, v in sets.items()}

    def rate(nm, want, tag):
        rs = res[nm]
        ok = sum(1 for r in rs if (r[tag] == "SAME_PROPOSITION") == want)
        return ok, len(rs), 100.0 * ok / max(len(rs), 1)

    stats = {}
    print("\n  【每通道度量】缝判别率（越高越好） / 对照保真率（越高越好） / Youden J / 平衡准确率")
    for tag, _, _, lab in PASSES:
        s2, c2 = rate("R2B_SEAM", False, tag), rate("R2B_CTRL", True, tag)
        s4, c4 = rate("R4_SEAM", False, tag), rate("R4_CTRL", True, tag)
        j2 = c2[2] / 100 + s2[2] / 100 - 1
        j4 = c4[2] / 100 + s4[2] / 100 - 1
        ba2 = (c2[2] / 100 + s2[2] / 100) / 2
        ba4 = (c4[2] / 100 + s4[2] / 100) / 2
        se, ce = (s2[2] + s4[2]) / 2, (c2[2] + c4[2]) / 2
        ok = se >= 80 and min(c2[2], c4[2]) >= 80
        stats[tag] = {"label": lab, "seam_r2b": s2[2], "ctrl_r2b": c2[2],
                      "seam_r4": s4[2], "ctrl_r4": c4[2], "seam_mean": se, "ctrl_mean": ce,
                      "J_r2b": round(j2, 3), "J_r4": round(j4, 3),
                      "BA_r2b": round(ba2, 3), "BA_r4": round(ba4, 3),
                      "repairable": bool(ok)}
        print("    %-26s | R2-b 缝%3.0f%% 对照%3.0f%% J=%.2f BA=%.2f | R4 缝%3.0f%% 对照%3.0f%% J=%.2f BA=%.2f | %s"
              % (tag + " " + lab, s2[2], c2[2], j2, ba2, s4[2], c4[2], j4, ba4,
                 "可修" if ok else "保持标界"))
    report["stats"] = stats

    # 两遍（同口径）都判对
    both = {}
    for pair, (t1, t2) in {"口径v1": ("A1", "B1"), "口径v2": ("A2", "B2")}.items():
        for nm, want in (("R2B_SEAM", False), ("R4_SEAM", False),
                         ("R2B_CTRL", True), ("R4_CTRL", True)):
            rs = res[nm]
            o = sum(1 for r in rs if (r[t1] == "SAME_PROPOSITION") == want
                    and (r[t2] == "SAME_PROPOSITION") == want)
            both[f"{pair}|{nm}"] = round(100.0 * o / max(len(rs), 1), 1)
    report["both_passes"] = both
    print("\n  两遍都判对：", both)

    dec = "REPAIRABLE" if any(v["repairable"] for v in stats.values()) else "KEEP_BOUNDARY"
    report["decision"] = dec
    print("\n  ===> 结论：%s" % ("通道可修（进入 Phase 0b，需另行拍板）"
                              if dec == "REPAIRABLE" else "保持标界（负结果）"))

    out = os.path.join(BASE, "E15_P0_缝探针.json")
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→ ", out)


if __name__ == "__main__":
    main()
