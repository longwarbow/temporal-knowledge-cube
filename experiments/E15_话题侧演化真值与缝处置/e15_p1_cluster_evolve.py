# -*- coding: utf-8 -*-
"""
E15 / Phase 1 —— L2 话题节点 + L3 演化操作（BIRTH/MERGE/SPLIT/CONTINUE/RETIRE）
   [v2 修订] 自由归组通道不可复现（两遍节点数 66 vs 124、κ=0.056）⇒ 改为
             **先归纳并冻结全局词表 → 再做指派（分类任务）**；MERGE/SPLIT 完全落在对齐层。
             v1 留档：e15_p1_cluster_evolve_v1_自由归组.py
-----------------------------------------------------------------------------
输入：E15_P1_声明_{event}.jsonl（e15_p1_extract.py 产出）

步骤
  A0 **词表归纳（一次，冻结）**：把覆盖 85% 声明的 aspect 标签交给 LLM，归纳 14–22 个
     话题节点（snake_case + 一句描述）⇒ E15_P1_词表_{event}.json（冻结后不再改）
  A1 **指派**：每个 aspect 标签 → **一个**词表节点（或 0=OTHER）；通道 A/B 两遍（不同序）
     ⇒ 标签级 Cohen's κ（度量"词表可执行性"，不是"话题是否真实存在"）
  B  **跨切片对齐**：相邻切片的**活跃节点**做多对多匹配（双通道 + 仲裁）
  C  **操作判定**：BIRTH / RETIRE / CONTINUE / MERGE / SPLIT
  D  **客观锚**：各节点内线程的 PHEME veracity 分布（BIRTH vs CONTINUE）

用法：
  python e15_p1_cluster_evolve.py --event sydneysiege-all-rnr-threads --slices 4
  python e15_p1_cluster_evolve.py --event germanwings-crash-all-rnr-threads --slices 6
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
       "You work with a FROZEN node vocabulary and must be reproducible: the same label always "
       "gets the same node, and two labels share a node only if they assert about the SAME "
       "property of the SAME object.")

VOCAB_PROMPT = """Event: {event}

Below are the distinct narrative aspect labels extracted from the posts of this event, with
thread counts. They cover {cov:.0f}% of the posts.

{items}

Induce a FROZEN vocabulary of {kmin}-{kmax} NARRATIVE NODES for this event.
Requirements:
- every label above must be coverable by exactly one node (or be left as OTHER, but keep OTHER small);
- granularity: two nodes must never assert about the same property of the same object
  (e.g. "hostage_count" and "hostage_escape" are DIFFERENT nodes; "hostage_count" and
  "number_of_hostages" would be the SAME node and must not both appear);
- each node: snake_case label (2-4 words) + one-line description.

Answer ONLY with compact JSON:
{{"nodes":[{{"node":"snake_case","desc":"<=15 words"}}]}}"""

ASSIGN_PROMPT = """Event: {event}   Time slice: {slice_label}

FROZEN node vocabulary (assign to exactly one id, or 0 = none fits):
{vocab}

Assign each aspect label below to exactly one node id:
{items}

Answer ONLY with compact JSON:
{{"assign":[{{"label":"<the label>","node":<id>}}]}}"""

ALIGN_PROMPT = """Event: {event}   Transition: {d1} -> {d2}

PREVIOUS slice active narrative nodes:
{prev}

NEXT slice active narrative nodes:
{next}

For EACH next-slice node, list the nodes of the previous slice that describe the SAME narrative
(same property of the same object, even if wording/count changed). If a next-slice node has no
counterpart, give it an EMPTY list. Use the exact node names as written above.

Answer ONLY with compact JSON:
{{"map":[{{"next":"<next node name>","prev":["<prev node name>", ...]}}]}}"""


def derive_ops(mapping, prev_names, next_names):
    """从匹配结构机械推导操作（不采信模型自报的类型）"""
    ops = collections.Counter()
    nxt2prev = {m.get("next"): [p for p in (m.get("prev") or []) if p] for m in mapping or []}
    prev2nxt = collections.defaultdict(list)
    for nx, ps in nxt2prev.items():
        for p in ps:
            prev2nxt[p].append(nx)
    for nx in next_names:
        ps = nxt2prev.get(nx) or []
        if len(ps) == 0:
            ops["BIRTH"] += 1
        elif len(ps) >= 2:
            ops["MERGE"] += 1
        else:
            ops["SPLIT" if len(prev2nxt.get(ps[0], [])) >= 2 else "CONTINUE"] += 1
    for p in prev_names:
        if not prev2nxt.get(p):
            ops["RETIRE"] += 1
    return ops, nxt2prev


def load_env():
    env = {}
    for line in open(os.path.join(ROOT, ".env_e14"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def call(client, model, prompt, retries=3, mt=1500):
    msg = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
    for i in range(retries):
        try:
            r = client.chat.completions.create(model=model, messages=msg,
                                               temperature=0.0, max_tokens=mt)
            txt = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except Exception as e:                                     # noqa: BLE001
            if i == retries - 1:
                return {"_err": f"{type(e).__name__}: {e}"[:140]}
            time.sleep(1.5 * (i + 1))
    return {"_err": "parse_fail"}


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def kappa(mapA, mapB, labels):
    a = b = c = d = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            p, q = labels[i], labels[j]
            sa = mapA.get(p) is not None and mapA.get(p) == mapA.get(q)
            sb = mapB.get(p) is not None and mapB.get(p) == mapB.get(q)
            if sa and sb:
                a += 1
            elif sa:
                b += 1
            elif sb:
                c += 1
            else:
                d += 1
    n = a + b + c + d
    if n == 0:
        return 0.0, (0, 0, 0, 0), 0.0
    po = (a + d) / n
    pa, pb = (a + b) / n, (a + c) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), (a, b, c, d), po


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=45)
    ap.add_argument("--slices", type=int, default=4)
    ap.add_argument("--kmin", type=int, default=14)
    ap.add_argument("--kmax", type=int, default=22)
    ap.add_argument("--force-vocab", action="store_true")
    ap.add_argument("--min-count", type=int, default=1,
                    help="只指派出现次数 >= N 的标签（诊断：不可靠性是否在长尾）")
    ap.add_argument("--l2-only", action="store_true",
                    help="浅做模式：只跑 L2（词表+指派+κ），跳过 L3 LLM 对齐通道；确定性层仍计算")
    ap.add_argument("--tag", default="", help="输出文件名后缀")
    args = ap.parse_args()

    env = load_env()
    model = env.get("E14_LLM_MODEL", "glm-4-flash")
    from openai import OpenAI
    client = OpenAI(base_url=env["E14_LLM_BASE_URL"], api_key=env["E14_LLM_API_KEY"], timeout=180.0)

    short = args.event.split("-")[0]
    rows = [json.loads(l) for l in
            open(os.path.join(BASE, "E15_P1_声明_%s.jsonl" % short), encoding="utf-8")]
    rows = [r for r in rows if (r.get("family") or "") not in ("ERR", "PARSE_FAIL")]
    rows.sort(key=lambda r: r["ms"])
    n = len(rows)
    for i, r in enumerate(rows):
        r["sl"] = min(args.slices - 1, int(i * args.slices / n))
    slices = []
    for k in range(args.slices):
        sub = [r for r in rows if r["sl"] == k]
        if sub:
            slices.append({"k": k, "label": "S%d %s→%s" %
                           (k + 1, sub[0]["iso"][5:16], sub[-1]["iso"][5:16]), "sub": sub})
    print("事件 %s：%d 条声明 / %d 切片" % (short, n, len(slices)))
    for s in slices:
        print("   %-24s n=%d" % (s["label"], len(s["sub"])))

    total = collections.Counter((r["aspect"] or "?").strip() for r in rows)

    # ---------------- A0. 词表归纳（冻结） ----------------
    vpath = os.path.join(BASE, "E15_P1_词表_%s.json" % short)
    if os.path.exists(vpath) and not args.force_vocab:
        voc = json.load(open(vpath, encoding="utf-8"))
        print("\n[A0] 复用已冻结词表：%d 节点 ← %s" % (len(voc["nodes"]), os.path.basename(vpath)))
    else:
        print("\n[A0] 词表归纳（一次，冻结）")
        items = sorted(total.items(), key=lambda x: (-x[1], x[0]))
        kept, cum = [], 0
        for a, c in items:
            kept.append((a, c))
            cum += c
            if cum / n >= 0.85:
                break
        cov = 100.0 * cum / n
        txt = "\n".join("- %s  (n=%d)" % (a, c) for a, c in kept)
        out = call(client, model, VOCAB_PROMPT.format(
            event=short, items=txt, cov=cov, kmin=args.kmin, kmax=args.kmax))
        nodes = out.get("nodes", []) or []
        assert nodes, "词表归纳失败：%r" % out
        voc = {"event": short, "nodes": nodes, "n_label_used": len(kept),
               "coverage_pct": round(cov, 1)}
        json.dump(voc, open(vpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("   → %d 节点（标签覆盖 %.0f%%）冻结于 %s" % (len(nodes), cov, os.path.basename(vpath)))
    for i, nd in enumerate(voc["nodes"], 1):
        print("     %2d %-28s %s" % (i, nd["node"], nd.get("desc", "")))
    nid = {nd["node"]: i for i, nd in enumerate(voc["nodes"], 1)}

    # ---------------- A1. 指派（双通道） ----------------
    print("\n[A1] 指派（aspect → 冻结节点；通道 A/B 两遍）")
    voc_txt = "\n".join("%d = %s — %s" % (nid[nd["node"]], nd["node"], nd.get("desc", ""))
                        for nd in voc["nodes"])
    L = {}
    for s in slices:
        sub = s["sub"]
        cnt = collections.Counter((r["aspect"] or "?").strip() for r in sub)
        labels = [a for a, _ in cnt.most_common() if cnt[a] >= args.min_count]

        def render(ch):
            return "\n".join('- [%d] "%s"  (n=%d)' % (i, a, cnt[a]) for i, a in enumerate(ch))

        tasks = [("A", ch) for ch in chunks(labels, args.chunk)]
        tasks += [("B", ch) for ch in chunks(sorted(labels), args.chunk)]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut = [(t, ex.submit(call, client, model, ASSIGN_PROMPT.format(
                event=short, slice_label=s["label"], vocab=voc_txt, items=render(t[1]))))
                for t in tasks]
            ma, mb = {}, {}
            for t, f in fut:
                for it in (f.result() or {}).get("assign", []) or []:
                    lab = str(it.get("label", "")).strip()
                    try:
                        nn = int(it.get("node", 0))
                    except Exception:                                  # noqa: BLE001
                        nn = 0
                    if lab:
                        (ma if t[0] == "A" else mb)[lab] = nn
        for lab in labels:
            ma.setdefault(lab, 0)
            mb.setdefault(lab, 0)
        k, tab, po = kappa(ma, mb, labels)
        node_lab = collections.defaultdict(list)
        for lab, nn in ma.items():
            if 1 <= nn <= len(voc["nodes"]):
                node_lab[voc["nodes"][nn - 1]["node"]].append(lab)
        node_threads = {nd: sum(cnt[m] for m in ms) for nd, ms in node_lab.items()}
        ver_by_node = {}
        for nd, ms in node_lab.items():
            c = collections.Counter()
            for r in sub:
                if (r["aspect"] or "?").strip() in ms:
                    c[r["label"]] += 1
            ver_by_node[nd] = dict(c)
        L[s["k"]] = {"label": s["label"], "n_threads": len(sub), "labels": labels,
                     "counts": dict(cnt), "node_members": dict(node_lab),
                     "node_threads": node_threads, "kappa": round(k, 4),
                     "kappa_tab": tab, "po": round(po, 4), "ver": ver_by_node,
                     "n_other": sum(1 for lab in labels if ma[lab] == 0)}
        json.dump({"slice": s["label"], "kappa": round(k, 4), "kappa_tab": tab,
                   "po": round(po, 4), "n_labels": len(labels),
                   "n_other": L[s["k"]]["n_other"], "node_threads": node_threads,
                   "node_members": dict(node_lab), "assign_A": ma, "assign_B": mb},
                  open(os.path.join(BASE, "E15_P1_指派_%s_S%d.json" % (short, s["k"] + 1)), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
        print("   %-24s %d 标签 → %d 活跃节点 (+%d OTHER) | κ=%.3f 一致率=%.3f" %
              (s["label"], len(labels), len(node_threads), L[s["k"]]["n_other"], k, po))

    # ---------------- B/C. 跨切片对齐（匹配由 LLM、操作由结构推导） ----------------
    ops = collections.Counter()
    transitions = []
    ks = sorted(L)
    if args.l2_only:
        print("\n[B/C] --l2-only：跳过 L3 LLM 对齐通道（浅做）")
    else:
        print("\n[B/C] 跨切片对齐（双通道：模型只给匹配，操作机械推导）")
    for a, b in zip(ks, ks[1:]):
        if args.l2_only:
            continue
        A = sorted(L[a]["node_threads"].items(), key=lambda x: -x[1])
        B = sorted(L[b]["node_threads"].items(), key=lambda x: -x[1])
        namesA = [nd for nd, _ in A]
        namesB = [nd for nd, _ in B]

        def run(orderA, orderB):
            pv = "\n".join("- %s (%d threads)" % (nd, c) for nd, c in orderA)
            nx = "\n".join("- %s (%d threads)" % (nd, c) for nd, c in orderB)
            return call(client, model, ALIGN_PROMPT.format(
                event=short, d1=L[a]["label"], d2=L[b]["label"], prev=pv, next=nx))

        with ThreadPoolExecutor(max_workers=2) as ex:
            # 通道 A：按线程量降序；通道 B：按名称字母序（**扰动呈现**，非同题重复）
            f1 = ex.submit(run, A, B)
            f2 = ex.submit(run, sorted(A), sorted(B))
            r1, r2 = f1.result(), f2.result()

        def edges(r):
            return {(m.get("next"), p) for m in (r.get("map") or []) for p in (m.get("prev") or []) if p}
        e1, e2 = edges(r1), edges(r2)
        e1 = {(nx, p) for nx, p in e1 if nx in namesB and p in namesA}
        e2 = {(nx, p) for nx, p in e2 if nx in namesB and p in namesA}
        agree, disp = len(e1 & e2), len(e1 ^ e2)
        opsA, mapA = derive_ops(r1.get("map"), namesA, namesB)
        opsB, mapB = derive_ops(r2.get("map"), namesA, namesB)
        for kk, vv in opsA.items():
            ops[kk] += vv
        transitions.append({"d1": L[a]["label"], "d2": L[b]["label"],
                            "n_prev": len(A), "n_next": len(B),
                            "map_A": mapA, "map_B": mapB,
                            "ops_A": dict(opsA), "ops_B": dict(opsB),
                            "edges_agree": agree, "edges_disputed": disp,
                            "err": r1.get("_err") or r2.get("_err")})
        print("   %s → %s | 前 %d 后 %d 节点 | 边 A=%d B=%d 一致=%d 争议=%d" %
              (L[a]["label"], L[b]["label"], len(A), len(B), len(e1), len(e2), agree, disp))
        print("       通道A 操作 %s | 通道B 操作 %s" % (dict(opsA), dict(opsB)))
        for nx, ps in sorted(mapA.items(), key=lambda x: str(x[0])):
            kk = len(ps)
            tag = "BIRTH" if kk == 0 else ("MERGE" if kk >= 2 else "CONTINUE")
            print("       %-8s %s <- %s" % (tag, nx, ps or "[]"))

    # -------- 确定性操作：节点在场/缺席（100% 可复现，不依赖 LLM） --------
    print("\n[B2] 确定性操作（节点在场/缺席，可完全复现）")
    det_ops_total = collections.Counter()
    det_per_trans = []
    for a, b in zip(ks, ks[1:]):
        A, B = set(L[a]["node_threads"]), set(L[b]["node_threads"])
        d = {"CONTINUE": len(A & B), "BIRTH": len(B - A), "RETIRE": len(A - B)}
        det_per_trans.append({"d1": L[a]["label"], "d2": L[b]["label"], "det": d,
                              "born": sorted(B - A), "retired": sorted(A - B)})
        for kk, vv in d.items():
            det_ops_total[kk] += vv
        print("   %s → %s | CONTINUE %d | BIRTH %d %s | RETIRE %d %s" %
              (L[a]["label"], L[b]["label"], d["CONTINUE"], d["BIRTH"],
               sorted(B - A) if B - A else "", d["RETIRE"], sorted(A - B) if A - B else ""))

    # 节点寿命表
    span = {}
    for nd in {x for k in ks for x in L[k]["node_threads"]}:
        act = [k for k in ks if nd in L[k]["node_threads"]]
        span[nd] = {"first": L[min(act)]["label"], "last": L[max(act)]["label"],
                    "n_slices": len(act),
                    "threads": [L[k]["node_threads"].get(nd, 0) for k in ks]}
    print("   节点寿命（threads 逐切片）：")
    for nd, v in sorted(span.items(), key=lambda x: -sum(x[1]["threads"])):
        print("     %-24s %s" % (nd, v["threads"]))

    # -------- 客观锚：BIRTH 节点 vs CONTINUE 节点的 veracity 构成 --------
    def ver_of(node_names):
        c = collections.Counter()
        for k in ks:
            for nd in node_names:
                for lab, cntv in (L[k]["ver"].get(nd) or {}).items():
                    pass
                c.update(L[k]["ver"].get(nd) or {})
        return dict(c)

    born_nodes = sorted({x for t in det_per_trans for x in t["born"]})
    cont_nodes = sorted({nd for nd in span if span[nd]["n_slices"] >= 2})
    anchor = {"born_nodes": born_nodes, "continue_nodes": cont_nodes,
              "ver_born": ver_of(born_nodes), "ver_continue": ver_of(cont_nodes)}
    print("   客观锚 veracity：BIRTH 节点 %s | 持续节点 %s" %
          (anchor["ver_born"], anchor["ver_continue"]))

    out = {"event": short, "n_claims": n, "vocab": voc,
           "min_count": args.min_count, "l2_only": bool(args.l2_only),
           "slices": {k: {kk: vv for kk, vv in L[k].items() if kk != "ver"} for k in L},
           "kappa_mean": round(sum(L[k]["kappa"] for k in L) / max(len(L), 1), 4),
           "po_mean": round(sum(L[k]["po"] for k in L) / max(len(L), 1), 4),
           "transitions": transitions, "ops_llm": dict(ops),
           "det_ops_total": dict(det_ops_total), "det_per_trans": det_per_trans,
           "node_lifespan": span, "anchor": anchor,
           "veracity_anchor": {L[k]["label"]: L[k]["ver"] for k in L}}
    json.dump(out, open(os.path.join(BASE, "E15_P1_话题_%s%s.json" % (short, args.tag)), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nκ 均值 = %.3f | 一致率均值 = %.3f" % (out["kappa_mean"], out["po_mean"]))
    print("LLM 归因操作 = %s" % dict(ops))
    print("确定性操作 = %s" % dict(det_ops_total))
    print("→ E15_P1_话题_%s%s.json" % (short, args.tag))


if __name__ == "__main__":
    main()
