# -*- coding: utf-8 -*-
"""
E15 —— 确定性层话题演化真值：汇总导出（E-e 拍板 (a) 的交付物）
=============================================================================
把各事件的 E15_P1_话题_{short}.json / _head.json 汇总成：
  1) E15_真值_确定性层_v1.csv  —— 一行 = (事件, 话题节点, 切片) 的在场记录；
     再由在场/缺席机械推出 CONTINUE/BIRTH/RETIRE（唯一可复现的操作层）
  2) E15_真值_跨事件总表.csv   —— 一行 = 一个事件（切片数、活跃节点、三操作计数、κ/po）
  3) 控制台打印论文可引用的读数

口径（冻结，与 E15_P1_报告 §2 一致）：
  * 节点的"在场"判定 = 该切片内有 >=1 条声明的 aspect 被指派到该节点（通道 A）。
  * 操作完全由"相邻切片在场集合"机械推导，**不采信任何 LLM 自报**：
      CONTINUE = 相邻两片都在场；BIRTH = 本片在场而上片不在；RETIRE = 上片在场而本片不在。
  * MERGE / SPLIT 需要跨片"同叙事"归因 ⇒ **不可复现** ⇒ 登记为残余，不写入真值。
  * _head（min-count 诊断）版本单独成行，不与全量混算。

用法：
  python e15_truth_export.py
  python e15_truth_export.py --pattern "E15_P1_话题_*.json"
"""
import argparse
import collections
import csv
import glob
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SKIP = {"E15_P1_话题_跨事件总表", "E15_真值"}


def load_one(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def det_per_transition(a, b):
    """相邻两片在场集合 → 三操作（纯确定性）"""
    A, B = set(a), set(b)
    return ({"CONTINUE": len(A & B), "BIRTH": len(B - A), "RETIRE": len(A - B)},
            sorted(B - A), sorted(A - B))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="E15_P1_话题_*.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(BASE, args.pattern)))
    files = [f for f in files if os.path.basename(f).replace(".json", "") not in SKIP
             and os.path.basename(f) not in SKIP]
    rows_cell, rows_event, summary_print = [], [], []

    for fp in files:
        o = load_one(fp)
        short = o.get("event") or os.path.basename(fp)
        variant = "head" if "_head" in os.path.basename(fp) else "full"
        sl = o.get("slices") or {}
        ks = sorted(sl, key=lambda x: int(x))
        if not ks:
            continue
        active = {k: set(sl[k]["node_threads"]) for k in ks}
        # ---- 空格行：每个 (node, slice) 的在场记录 ----
        allnodes = sorted({nd for k in ks for nd in active[k]})
        prev = None
        det_tot = collections.Counter()
        for k in ks:
            d, born, retired = (det_per_transition(active[prev], active[k])
                                if prev is not None else ({}, [], []))
            for nd in allnodes:
                in_here = nd in active[k]
                if prev is None:
                    op = "ENTER" if in_here else ""
                elif in_here and nd in active[prev]:
                    op = "CONTINUE"
                elif in_here:
                    op = "BIRTH"
                elif nd in active[prev]:
                    op = "RETIRE"
                else:
                    op = ""
                if op:
                    rows_cell.append({
                        "event": short, "variant": variant,
                        "slice_idx": int(k) + 1, "slice_label": sl[k]["label"],
                        "node": nd, "in_slice": 1 if in_here else 0,
                        "threads_in_slice": sl[k]["node_threads"].get(nd, 0),
                        "n_member_labels": len(sl[k]["node_members"].get(nd) or []),
                        "op_vs_prev": op, "n_claims_in_slice": sl[k]["n_threads"],
                    })
            for kk, vv in d.items():
                det_tot[kk] += vv
            prev = k
        # ---- 事件级总表 ----
        rows_event.append({
            "event": short, "variant": variant,
            "n_claims": o.get("n_claims"), "n_slices": len(ks),
            "n_nodes_vocab": len((o.get("vocab") or {}).get("nodes") or []),
            "vocab_coverage_pct": (o.get("vocab") or {}).get("coverage_pct"),
            "n_nodes_active_union": len(allnodes),
            "CONTINUE": det_tot.get("CONTINUE", 0),
            "BIRTH": det_tot.get("BIRTH", 0),
            "RETIRE": det_tot.get("RETIRE", 0),
            "kappa_mean": o.get("kappa_mean"), "po_mean": o.get("po_mean"),
            "l2_only": bool(o.get("l2_only")),
            "n_llm_transitions": len(o.get("transitions") or []),
            "llm_ops": json.dumps(o.get("ops_llm") or {}, ensure_ascii=False),
            "src_file": os.path.basename(fp),
        })
        summary_print.append((short, variant, len(ks), len(allnodes),
                              det_tot.get("CONTINUE", 0), det_tot.get("BIRTH", 0),
                              det_tot.get("RETIRE", 0),
                              o.get("kappa_mean"), o.get("po_mean"),
                              o.get("n_claims")))

    # ---- 写 CSV ----
    p1 = os.path.join(BASE, "E15_真值_确定性层_v1.csv")
    if rows_cell:
        with open(p1, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_cell[0].keys()))
            w.writeheader()
            w.writerows(rows_cell)
    p2 = os.path.join(BASE, "E15_真值_跨事件总表.csv")
    if rows_event:
        with open(p2, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_event[0].keys()))
            w.writeheader()
            w.writerows(rows_event)

    # ---- 控制台 ----
    print("=" * 100)
    print("E15 确定性层真值 —— 跨事件总表（CONTINUE/BIRTH/RETIRE 为可复现三操作）")
    print("=" * 100)
    print("%-16s %-5s %7s %6s %6s %9s %6s %7s %8s %7s  %s" %
          ("event", "var", "claims", "slices", "nodes", "CONTINUE", "BIRTH", "RETIRE",
           "kappa", "po", "flag"))
    for r in summary_print:
        k, po, ncl = r[7], r[8], r[9]
        flag = ""
        if isinstance(k, (int, float)) and k >= 0.70:
            flag = ("⚠ 退化：n=%d 过小，κ 不可信" % ncl) if (ncl or 0) < 80 else "✅ 过门"
        elif isinstance(ncl, int) and ncl < 80:
            flag = "⚠ n=%d 功效不足" % ncl
        print("%-16s %-5s %7s %6d %6d %9d %6d %7d %8s %7s  %s" %
              (r[0], r[1], ncl, r[2], r[3], r[4], r[5], r[6],
               ("%.3f" % k) if isinstance(k, (int, float)) else "-",
               ("%.3f" % po) if isinstance(po, (int, float)) else "-", flag))

    full = [r for r in summary_print if r[1] == "full"]
    if full:
        print("-" * 100)
        print("full 事件数 = %d | BIRTH 合计 = %d | RETIRE 合计 = %d | CONTINUE 合计 = %d" %
              (len(full), sum(r[5] for r in full), sum(r[6] for r in full),
               sum(r[4] for r in full)))
        sized = [r for r in full if isinstance(r[9], int) and r[9] >= 80]
        kk = [r[7] for r in sized if isinstance(r[7], (int, float))]
        print("达到功效门（n≥80）的事件数 = %d" % len(sized))
        if kk:
            print("  κ 范围 = %.3f–%.3f（均值 %.3f）| 门限 0.70 ⇒ %s" %
                  (min(kk), max(kk), sum(kk) / len(kk),
                   "全部未过门（负结果跨事件成立）" if max(kk) < 0.70 else "存在过门事件"))
    print("\n→ %s（%d 行）" % (os.path.basename(p1), len(rows_cell)))
    print("→ %s（%d 行）" % (os.path.basename(p2), len(rows_event)))


if __name__ == "__main__":
    main()
