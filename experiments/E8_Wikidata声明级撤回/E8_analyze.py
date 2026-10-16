# -*- coding: utf-8 -*-
"""E8 分析汇总：把四张 CSV + 标签 JSON 汇成报告用的关键数字。"""
import csv, json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent


def load(name):
    with open(BASE / name, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


props = load("E8_属性级撤回率.csv")
reasons = load("E8_原因分布.csv")
samples = load("E8_声明级样本.csv")
p813 = load("E8_P813年度分布.csv")
labels = json.load(open(BASE / "E8_原因标签.json", encoding="utf-8"))

print("== 1) 属性级撤回率 ==")
exact = [(r["pid"], r["gloss"], int(r["deprecated"]), float(r["retraction_rate"]))
         for r in props if r["dep_is_lower_bound"] == "0"]
lb = [(r["pid"], r["gloss"], int(r["deprecated"]), float(r["retraction_rate"]))
      for r in props if r["dep_is_lower_bound"] == "1"]
exact.sort(key=lambda x: -x[3])
tot_exact = sum(x[2] for x in exact)
tot_all = sum(int(r["deprecated"]) for r in props)
tot_stmt = sum(int(r["total_statements"]) for r in props)
print(f"exact props={len(exact)} dep_sum={tot_exact}; LB props={len(lb)} dep_sum={sum(x[2] for x in lb)}")
print(f"curated dep total={tot_all} / statements total={tot_stmt}")
for x in exact:
    print("  %-5s %-24s dep=%6d rate=%.4f%%" % (x[0], x[1], x[2], x[3] * 100))
for x in sorted(lb, key=lambda x: -x[3]):
    print("  %-5s %-24s dep>=%6d rate>=%.4f%% (LB)" % (x[0], x[1], x[2], x[3] * 100))

print("\n== 2) 原因分布 Top15（跨属性合计）==")
agg = defaultdict(int)
for r in reasons:
    agg[r["reason"]] += int(r["n"])
for q, n in sorted(agg.items(), key=lambda x: -x[1])[:15]:
    qid = q.rsplit("/", 1)[-1]
    en, zh = labels.get(q, ("", ""))
    print("  %-12s %8d  %s | %s" % (qid, n, en, zh))
print("  distinct reasons=%d, total with-reason=%d" % (len(agg), sum(agg.values())))

print("\n== 3) P813(retrieved) 年度分布（裁定时刻上界）==")
year_agg = defaultdict(int)
per_prop = defaultdict(lambda: defaultdict(int))
for r in p813:
    year_agg[int(r["year"])] += int(r["n"])
    per_prop[r["pid"]][int(r["year"])] = int(r["n"])
tot_p813 = sum(year_agg.values())
print("  total P813 on reason-anchored deprecated:", tot_p813)
for y in sorted(year_agg):
    bar = "#" * max(1, year_agg[y] // 200)
    print("  %d %6d %s" % (y, year_agg[y], bar))
pre2017 = sum(n for y, n in year_agg.items() if y < 2017)
post2017 = tot_p813 - pre2017
print("  <2017: %d (%.1f%%)  >=2017: %d (%.1f%%)"
      % (pre2017, 100 * pre2017 / tot_p813, post2017, 100 * post2017 / tot_p813))

print("\n== 4) 声明级样本质量 ==")
by = defaultdict(list)
for s in samples:
    by[s["pid"]].append(s)
for pid, ss in by.items():
    n = len(ss)
    p582 = sum(1 for x in ss if x.get("endtime"))
    p813c = sum(1 for x in ss if x.get("retrieved"))
    reas = sum(1 for x in ss if x.get("reason"))
    print("  %s: n=%d reason_cov=%.2f p582_cov=%.2f p813_cov=%.2f anchored=%s"
          % (pid, n, reas / n, p582 / n, p813c / n, ss[0].get("anchored")))

print("\n== 5) 样本示例（声明级四元组+时刻代理）==")
shown = 0
for s in samples:
    if s.get("endtime") and s.get("reason") and shown < 5:
        qid = s["reason"].rsplit("/", 1)[-1]
        en, zh = labels.get(s["reason"], ("", ""))
        print("  [%s] %s --%s--> %s | end=%s ret=%s | %s/%s"
              % (s["pid"], s["item"].rsplit("/", 1)[-1], s["value"].rsplit("/", 1)[-1],
                 s["value"].rsplit("/", 1)[-1] if s["pid"] not in ("P569",) else "(date)",
                 s.get("endtime", ""), s.get("retrieved", ""), en, zh))
        shown += 1
print("\n== 6) 互反声明对称撤回（跨声明失效传播的直接证据）==")
p26 = [s for s in samples if s["pid"] == "P26"]
idx = {(s["item"].rsplit("/", 1)[-1], s["value"].rsplit("/", 1)[-1]): s for s in p26}
pairs = []
for (i, v), s in idx.items():
    rev = idx.get((v, i))
    if rev and i < v:
        pairs.append((i, v, s.get("endtime", ""), rev.get("endtime", ""),
                      s.get("reason", "").rsplit("/", 1)[-1], rev.get("reason", "").rsplit("/", 1)[-1]))
same_end = [p for p in pairs if p[2] and p[2] == p[3]]
same_reason = [p for p in pairs if p[4] and p[4] == p[5]]
print("  P26 样本 %d 条 → 互反对 %d；其中 end time 完全相同 %d 对、原因相同 %d 对"
      % (len(p26), len(pairs), len(same_end), len(same_reason)))
for p in same_end[:5]:
    print("    %s <-> %s  end=%s  reason=%s/%s" % (p[0], p[1], p[2], p[4], p[5]))

