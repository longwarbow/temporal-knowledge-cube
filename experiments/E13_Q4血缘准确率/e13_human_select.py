# -*- coding: utf-8 -*-
"""
E13 人类盲标 —— 抽题脚本（可复现）

目的：为「人类 kappa（独立证据）」抽一份**分层过采样**的盲标样本。

设计理由
--------
真值分布（755 条）：DERIVE 105 / SUPPORT 145 / CONFLICT 43 / UNRELATED 462。
CONFLICT 仅占 5.7%，若按自然比例抽 48 题只能拿到 ≈3 条 CONFLICT
⇒ kappa 会被 UNRELATED 主导，稀有类的信度区间宽到无意义。
因此采用**按标签定量的分层抽样**（每类等量抽取，类内按 stratum 轮转以保证情形多样），
并在评分时同时报告：
  (a) 平衡样本 kappa（本设计的 kappa，需声明非自然比例）
  (b) 逐类一致率 / 逐类 P-R-F1（与基率无关，是论文里真正要引用的量）

产出：
  E13_人类盲标_题目.md       —— 人读版（HTML 实体已反转义）
  E13_人类盲标_作答.csv      —— 待填作答模板（两遍：直觉 / 依指南）
  E13_人类盲标_索引.json     —— Q编号 -> pair_id / stratum / subtype（评分用，标注员不看）

盲性保证：本文件不写出任何真值；盲版仅含 Q 编号 + 两条推文文本 + 事件名。
"""

import csv
import json
import html
import random
from collections import defaultdict

SEED = 20261012
N_TARGET = {"DERIVE": 10, "SUPPORT": 12, "CONFLICT": 12, "UNRELATED": 14}  # 合计 48
TRUTH_CSV = "E13_全量真值_v3.csv"
TEXT_CSV = "E13_盲版标注表_v3.csv"

OUT_MD = "E13_人类盲标_题目.md"
OUT_ANS = "E13_人类盲标_作答.csv"
OUT_IDX = "E13_人类盲标_索引.json"


def load():
    truth = list(csv.DictReader(open(TRUTH_CSV, encoding="utf-8-sig")))
    text = {}
    for r in csv.DictReader(open(TEXT_CSV, encoding="utf-8-sig")):
        text[r["pair_id"]] = (r["event"], r["p_text"], r["q_text"])
    return truth, text


def clean(s):
    return html.unescape(s or "").replace("\u2028", " ").strip()


def round_robin(pool, n, rng):
    """类内按 stratum 轮转抽取 —— 保证情形多样，不让单一大层垄断。"""
    by = defaultdict(list)
    for r in pool:
        by[r["stratum"]].append(r)
    for k in by:
        rng.shuffle(by[k])
    order = sorted(by.keys())
    got, i = [], 0
    while len(got) < n and any(by[k] for k in order):
        s = order[i % len(order)]
        if by[s]:
            got.append(by[s].pop())
        i += 1
    return got


def main():
    truth, text = load()
    rng = random.Random(SEED)

    picked = []
    for lab, n in N_TARGET.items():
        pool = [r for r in truth if r["truth"] == lab and r["pair_id"] in text]
        got = round_robin(pool, n, rng)
        assert len(got) == n, (lab, len(got), n)
        picked += got

    rng.shuffle(picked)  # 打乱顺序，避免同类/同层相邻暴露分组

    idx = {}
    ans_rows, md_lines = [], []
    md_lines.append("# E13 人类盲标 · 题目（共 %d 题）\n" % len(picked))
    md_lines.append("> **只判 $q$ 相对 $p$ 的关系**：`DERIVE` / `SUPPORT` / `CONFLICT` / `UNRELATED`，四选一。")
    md_lines.append("> $p$ 较早、$q$ 较晚，两者来自同一真实事件。**不要**猜这对是怎么挑出来的。")
    md_lines.append("> 作答请填 `E13_人类盲标_作答.csv`（列名见该文件表头）。\n")

    for i, r in enumerate(picked, 1):
        qid = "Q%02d" % i
        pid = r["pair_id"]
        ev, p, q = text[pid]
        idx[qid] = {"pair_id": pid, "stratum": r["stratum"], "subtype": r.get("subtype", "")}
        ans_rows.append({"qid": qid,
                         "label_intuition": "", "conf_intuition": "", "note_intuition": "",
                         "label_guided": "", "conf_guided": "", "note_guided": ""})
        md_lines.append("### %s" % qid)
        md_lines.append("- 事件：`%s`" % ev.replace("-all-rnr-threads", ""))
        md_lines.append("- **p**（较早）：%s" % clean(p))
        md_lines.append("- **q**（较晚）：%s" % clean(q))
        md_lines.append("")

    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md_lines))
    with open(OUT_ANS, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "label_intuition", "conf_intuition", "note_intuition",
                                          "label_guided", "conf_guided", "note_guided"])
        w.writeheader()
        w.writerows(ans_rows)
    json.dump({"seed": SEED, "n": len(picked), "target": N_TARGET, "index": idx},
              open(OUT_IDX, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # --- 自检：真实分层构成（不写进盲版，仅打印供我核对）---
    st = defaultdict(int)
    for r in picked:
        st[(r["stratum"], r["truth"])] += 1
    print("抽中 %d 题。分层×真值构成：" % len(picked))
    for k in sorted(st):
        print("   %-34s %-10s %d" % (k[0], k[1], st[k]))
    print("已写出：%s / %s / %s" % (OUT_MD, OUT_ANS, OUT_IDX))


if __name__ == "__main__":
    main()
