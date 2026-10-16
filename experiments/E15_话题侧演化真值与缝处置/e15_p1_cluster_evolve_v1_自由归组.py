# -*- coding: utf-8 -*-
"""
E15 / Phase 1 —— L2 话题节点构建 + L3 演化操作（BIRTH/MERGE/SPLIT/CONTINUE/RETIRE）
-----------------------------------------------------------------------------
输入：e15_p1_extract.py 产出的 E15_P1_声明_{event}.jsonl

方法（标准"话题演化"构造；不用相似度定义话题 —— 见 维度设计_话题维度与粒度_v1.md §3.1）
  A. **日局部节点** L(d)：把当日 (aspect, family) 原始标签批量交给 LLM 归组为节点
     —— 通道 A / 通道 B 两遍（不同呈现序）⇒ 标签级"同节点"判定的 Cohen's κ
  B. **跨日对齐**：L(d) 的节点 → L(d+1) 的节点（多对多），双通道 + 仲裁
  C. **操作判定**（相邻日）：
       BIRTH     : L(d+1) 节点无前驱
       RETIRE    : L(d) 节点无后继
       CONTINUE  : 1–1 匹配
       MERGE     : ≥2 个 L(d) 节点 → 1 个 L(d+1) 节点
       SPLIT     : 1 个 L(d) 节点 → ≥2 个 L(d+1) 节点
  D. **客观锚**：各节点内线程的 veracity 分布（BIRTH vs CONTINUE 是否系统不同）

输出：E15_P1_话题_{event}.json / E15_P1_节点_{event}_{day}.json

用法：
  python e15_p1_cluster_evolve.py --event sydneysiege-all-rnr-threads
"""
import argparse
import collections
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, "..", "..", ".."))

SYS = ("You are an annotator building the TOPIC dimension of a rumour-analysis benchmark. "
       "You group narrative labels and align narratives across days. Be conservative: "
       "two labels belong together only if they assert about the SAME property of the SAME "
       "object, not merely because they are about the same event.")

GROUP_PROMPT = """Event: {event}
Day: {day}

Below are the distinct narrative aspect labels extracted from the posts of this day, with
their thread counts. The predicate family is given in brackets:

{items}

Group them into NARRATIVE NODES. Two labels belong to the same node only if a reader would say
they assert about the same property of the same object (e.g. "hostage_count" and "hostages_held"
= one node; "hostage_count" and "gunman_identity" = different nodes).
Nodes must not overlap and must cover every label exactly once.
Name each node with a short English snake_case label (2-4 words).

Answer ONLY with compact JSON:
{{"nodes":[{{"node":"snake_case","family":"<dominant family>","members":["label A","label B"]}}]}}"""

ALIGN_PROMPT = """Event: {event}   Transition: {d1} -> {d2}

PREVIOUS day narrative nodes (id: label, n threads):
{prev}

NEXT day narrative nodes (id: label, n threads):
{next}

Decide how the narratives evolved. Every next-day node must appear in exactly one link.
- CONTINUE : a previous node and a next node are the SAME narrative
- SPLIT    : one previous node corresponds to SEVERAL next nodes
- MERGE    : SEVERAL previous nodes correspond to ONE next node
- BIRTH    : a next node has no previous counterpart (prev = null)
- RETIRE   : a previous node has no next counterpart (next = null)

Answer ONLY with compact JSON:
{{"links":[{{"prev":<id or null>,"next":<id or null>,"type":"CONTINUE|SPLIT|MERGE|BIRTH|RETIRE"}}]}}"""


def load_env():
    env = {}
    for line in open(os.path.join(ROOT, ".env_e14"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def call(client, model, prompt, retries=3, mt=1200):
    msg = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
    for i in range(retries):
        try:
            r = client.chat.completions.create(model=model, messages=msg,
                                               temperature=0.0, max_tokens=mt)
            txt = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                s = re.sub(r",\s*([}\]])", r"\1", m.group(0))
                return json.loads(s)
        except Exception as e:                                     # noqa: BLE001
            if i == retries - 1:
                return {"_err": f"{type(e).__name__}: {e}"[:140]}
            time.sleep(1.5 * (i + 1))
    return {"_err": "parse_fail"}


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def label_to_node(nodes):
    m = {}
    for nd in nodes or []:
        name = str(nd.get("node", "?")).strip()
        for mem in nd.get("members", []) or []:
            m[str(mem).strip()] = name
    return m


def kappa_pairs(mapA, mapB, labels, fam=None):
    a = b = c = d = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            p, q = labels[i], labels[j]
            if fam is not None and fam.get(p) != fam.get(q):
                continue          # 跨族对：按 §2.5 规则必然"不同槽位"，不计入可靠性
            sameA = mapA.get(p) is not None and mapA.get(p) == mapA.get(q)
            sameB = mapB.get(p) is not None and mapB.get(p) == mapB.get(q)
            if sameA and sameB:
                a += 1
            elif sameA:
                b += 1
            elif sameB:
                c += 1
            else:
                d += 1
    n = a + b + c + d
    if n == 0:
        return 0.0, (a, b, c, d)
    po = (a + d) / n
    pa, pb = (a + b) / n, (a + c) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), (a, b, c, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=55)
    ap.add_argument("--max-nodes", type=int, default=40)
    ap.add_argument("--slices", type=int, default=0,
                    help="等量时序切片数；0 = 按自然日")
    args = ap.parse_args()

    env = load_env()
    model = env.get("E14_LLM_MODEL", "glm-4-flash")
    from openai import OpenAI
    client = OpenAI(base_url=env["E14_LLM_BASE_URL"], api_key=env["E14_LLM_API_KEY"], timeout=180.0)

    short = args.event.split("-")[0]
    rows = [json.loads(l) for l in
            open(os.path.join(BASE, "E15_P1_声明_%s.jsonl" % short), encoding="utf-8")]
    rows.sort(key=lambda r: r["ms"])

    # ---- 时间切片：等量切片（默认）或自然日 ----
    if args.slices and args.slices > 1:
        n = len(rows)
        for i, r in enumerate(rows):
            k = min(args.slices - 1, int(i * args.slices / n))
            r["slice"] = k
        sl = {}
        for k in range(args.slices):
            sub = [r for r in rows if r["slice"] == k]
            if not sub:
                continue
            sl[k] = "S%d(%s→%s)" % (k + 1, sub[0]["iso"][5:16], sub[-1]["iso"][5:16])
    else:
        for r in rows:
            r["slice"] = r["day"]
        sl = {k: k for k in sorted({r["slice"] for r in rows})}
    order = sorted(sl)
    print("事件 %s：%d 条声明 / %d 个时序切片（等量切片数=%s）" %
          (short, len(rows), len(order), args.slices or "日"))
    for k in order:
        print("   %-26s n=%d" % (sl[k], sum(1 for r in rows if r["slice"] == k)))

    by_day = collections.defaultdict(list)
    for r in rows:
        by_day[r["slice"]].append(r)
    days = order

    # ---------------- A. 日局部节点（按 A4 族分块，双通道归组） ----------------
    print("\n[A] 切片局部节点（按 A4 谓词族分块 + 双通道归组）")
    L = {}
    for day in days:
        rs = by_day[day]
        cnt = collections.Counter((r["aspect"] or "?").strip() for r in rs)
        fam = {}
        for r in rs:
            fam.setdefault((r["aspect"] or "?").strip(), r["family"])
        labels = [a for a, _ in cnt.most_common()]

        # 分块：A4 族（同族内才可能同节点；跨族由 §2.5 规则判为不同槽位）
        blocks = collections.defaultdict(list)
        for a in labels:
            blocks[fam.get(a, "NONE") or "NONE"].append(a)

        tasks = []          # (order_tag, block, chunk_items)
        for blk, labs in sorted(blocks.items(), key=lambda x: -len(x[1])):
            items = sorted([(a, cnt[a]) for a in labs], key=lambda x: (-x[1], x[0]))
            if len(items) == 1:
                tasks.append(("A", blk, items, True))
                tasks.append(("B", blk, items, True))
                continue
            for ch in chunks(items, args.chunk):
                tasks.append(("A", blk, ch, False))
            itemsB = sorted(items, key=lambda x: x[0])
            for ch in chunks(itemsB, args.chunk):
                tasks.append(("B", blk, ch, False))

        def render(ch, blk):
            return "\n".join("- [%d] %s  (family=%s, n=%d)" %
                             (i, a, fam.get(a, "NONE"), n) for i, (a, n) in enumerate(ch))

        print("  %s：%d 条 / %d 标签 / %d 块 / %d 次调用" %
              (day, len(rs), len(labels), len(blocks), len(tasks)))
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut = [(t, ex.submit(call, client, model,
                                 GROUP_PROMPT.format(event=short, day=sl[day],
                                                     items=render(t[2], t[1])))) for t in tasks]
            outs = {"A": [], "B": []}
            for t, f in fut:
                r = f.result() or {}
                if t[3]:                                    # 单标签块 → 单例节点
                    a = t[2][0][0]
                    outs[t[0]].append({"node": a, "family": t[1], "members": [a]})
                else:
                    outs[t[0]] += (r.get("nodes", []) or [])
        nodesA, nodesB = outs["A"], outs["B"]

        # 同名节点合并 + 覆盖补漏（两遍皆未覆盖的标签各自成单例）
        def merge(nodes):
            acc = {}
            for nd in nodes:
                nm = str(nd.get("node", "?")).strip() or "?"
                key = nm if nm in acc else nm
                if key not in acc:
                    acc[key] = {"node": nm, "family": nd.get("family", "NONE"), "members": []}
                for m in nd.get("members", []) or []:
                    m = str(m).strip()
                    if m in cnt and m not in acc[key]["members"]:
                        acc[key]["members"].append(m)
            return [v for v in acc.values() if v["members"]]
        nodesA, nodesB = merge(nodesA), merge(nodesB)
        covA = {m for nd in nodesA for m in nd["members"]}
        covB = {m for nd in nodesB for m in nd["members"]}
        for lab in labels:
            if lab not in covA:
                nodesA.append({"node": lab, "family": fam.get(lab, "NONE"),
                               "members": [lab], "_singleton": True})
            if lab not in covB:
                nodesB.append({"node": lab, "family": fam.get(lab, "NONE"),
                               "members": [lab], "_singleton": True})
        k_all, tab_all = kappa_pairs(label_to_node(nodesA), label_to_node(nodesB), labels)
        k_in, tab_in = kappa_pairs(label_to_node(nodesA), label_to_node(nodesB), labels, fam)
        # 节点线程量（按通道 A）
        def size(nd):
            return sum(cnt.get(m, 0) for m in nd["members"])
        ver_by_node = {}
        for nd in nodesA:
            c = collections.Counter()
            for r in rs:
                if (r["aspect"] or "?").strip() in nd["members"]:
                    c[r["label"]] += 1
            ver_by_node[nd["node"]] = dict(c)
        L[day] = {"n_threads": len(rs), "labels": labels, "counts": dict(cnt), "fam": fam,
                  "nodesA": sorted(nodesA, key=size, reverse=True), "nodesB": nodesB,
                  "kappa": round(k_in, 4), "kappa_tab": tab_in,
                  "kappa_all": round(k_all, 4), "kappa_all_tab": tab_all,
                  "ver": ver_by_node}
        json.dump({"day": day, "label": sl[day], "kappa_intra_family": round(k_in, 4),
                   "kappa_tab": tab_in, "kappa_all": round(k_all, 4),
                   "nodesA": L[day]["nodesA"], "nodesB": nodesB, "counts": dict(cnt),
                   "fam": fam},
                  open(os.path.join(BASE, "E15_P1_节点_%s_%s.json" % (short, day)), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
        if nodesA:
            top = max(L[day]["nodesA"], key=size)
            print("      → 通道A %d 节点 / 通道B %d 节点 | κ(族内)=%.4f %s | κ(全)=%.4f | 最大节点 %s(%d)" %
                  (len(nodesA), len(nodesB), k_in, tab_in, k_all, top["node"], size(top)))

    # ---------------- B/C. 跨日对齐 + 操作 ----------------
    print("\n[B/C] 跨日对齐（双通道 + 仲裁）与演化操作")
    ops = collections.Counter()
    transitions = []
    for d1, d2 in zip(days, days[1:]):
        A, B = L[d1]["nodesA"][:args.max_nodes], L[d2]["nodesA"][:args.max_nodes]
        if not A or not B:
            continue
        sa = {nd["node"]: sum(L[d1]["counts"].get(m, 0) for m in nd["members"]) for nd in A}
        sb = {nd["node"]: sum(L[d2]["counts"].get(m, 0) for m in nd["members"]) for nd in B}
        pv = "\n".join("%d: %s (%d threads)" % (i, nd["node"], sa[nd["node"]])
                       for i, nd in enumerate(A))
        nx = "\n".join("%d: %s (%d threads)" % (i, nd["node"], sb[nd["node"]])
                       for i, nd in enumerate(B))
        pr = ALIGN_PROMPT.format(event=short, d1=d1, d2=d2, prev=pv, next=nx)
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1, f2 = ex.submit(call, client, model, pr), ex.submit(call, client, model, pr)
            r1, r2 = f1.result(), f2.result()

        def key(links):
            return {(l.get("prev"), l.get("next"), str(l.get("type", "")).upper())
                    for l in (links or [])}
        s1, s2 = key(r1.get("links")), key(r2.get("links"))
        agree, disputed = len(s1 & s2), len(s1 ^ s2)
        for p_, n_, t_ in s1:
            ops[t_ if (p_, n_, t_) in s2 else "DISPUTED"] += 1
        transitions.append({"d1": d1, "d2": d2, "n_prev": len(A), "n_next": len(B),
                            "chA_links": r1.get("links", []), "chB_links": r2.get("links", []),
                            "agree": agree, "disputed": disputed,
                            "errA": r1.get("_err"), "errB": r2.get("_err")})
        print("  %s→%s  前 %d 后 %d | A links=%d B links=%d 一致=%d 争议=%d" %
              (d1, d2, len(A), len(B), len(s1), len(s2), agree, disputed))

    # ---------------- D. 客观锚 ----------------
    anchor = {}
    for day in days:
        for nd in L[day]["nodesA"]:
            anchor.setdefault(nd["node"], collections.Counter())
            anchor[nd["node"]].update(L[day]["ver"].get(nd["node"], {}))
    out = {"event": short, "days": days, "n_claims": len(rows),
           "per_day": {d: {"n_threads": L[d]["n_threads"], "n_labels": len(L[d]["labels"]),
                           "n_nodes_A": len(L[d]["nodesA"]), "kappa": L[d]["kappa"]}
                       for d in days},
           "nodes": {d: L[d]["nodesA"] for d in days},
           "transitions": transitions, "ops": dict(ops),
           "veracity_anchor": {k: dict(v) for k, v in anchor.items()}}
    json.dump(out, open(os.path.join(BASE, "E15_P1_话题_%s.json" % short), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n操作统计：", dict(ops))
    print("→ E15_P1_话题_%s.json" % short)


if __name__ == "__main__":
    main()
