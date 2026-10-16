# -*- coding: utf-8 -*-
"""E9 分析：裁定时刻样本统计 + 机制分布 + 与 E8 时刻代理的交叉校验。

输入：
  E9_裁定时刻.csv        （E9_extract_adjudication.py）
  E9_机制核验.csv       （E9_flip_mechanism.py，可选）
输出：
  E9_统计.json
"""
import csv, json, re, statistics as st
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent


def pdate(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:                                            # noqa: BLE001
        return None


def qtl(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


rows = list(csv.DictReader(open(BASE / "E9_裁定时刻.csv", encoding="utf-8-sig")))
# ★ 两级可用性：
#   usable = 拿到 flip_ts 的行（裁定时刻统计可用；含「只拿到 flip、没拿到 birth」的部分行）
#   full   = 同时拿到 birth（存活时长统计可用）
usable = [r for r in rows if r.get("flip_ts")]
full = [r for r in usable if r.get("birth_ts") and "partial" not in (r.get("status") or "")]
ok = usable

print("== 0) 样本与状态 ==")
print("total=%d  有 flip=%d  含 birth=%d  (%.1f%% / %.1f%%)"
      % (len(rows), len(usable), len(full), 100 * len(usable) / max(1, len(rows)),
         100 * len(full) / max(1, len(rows))))
for k, v in Counter(r["status"][:60] for r in rows).most_common():
    print("   %-60s %d" % (k, v))
print("matched_by:", dict(Counter(r.get("matched_by", "") for r in usable)))
print("属性分布:", dict(Counter(r["pid"] for r in usable)))
print("边界核验 flip_prev_state:", dict(Counter(r.get("flip_prev_state", "") for r in usable)))

print("\n== 1) 裁定时刻（flip）年份分布 ==")
flips = [(r, pdate(r["flip_ts"])) for r in ok if pdate(r["flip_ts"])]
yr = Counter(d.year for _, d in flips)
for y in sorted(yr):
    print("   %d %4d %s" % (y, yr[y], "#" * max(1, yr[y] * 40 // max(yr.values(), default=1))))
print("   flips=%d  范围 %s → %s" % (len(flips), min(d for _, d in flips).date(),
                                     max(d for _, d in flips).date()))
print("   ★ 裁定时刻落在 2017 年及以后: %d/%d (%.1f%%)"
      % (sum(1 for _, d in flips if d.year >= 2017), len(flips),
         100 * sum(1 for _, d in flips if d.year >= 2017) / max(1, len(flips))))

print("\n== 2) 声明存活时长（flip − birth）==")
life = [float(r["lifespan_days"]) for r in ok if r.get("lifespan_days") not in (None, "")]
if life:
    print("n=%d  min=%.2f  p10=%.2f  p25=%.2f  median=%.2f  p75=%.2f  p90=%.2f  max=%.2f (天)"
          % (len(life), min(life), qtl(life, .10), qtl(life, .25), qtl(life, .50),
             qtl(life, .75), qtl(life, .90), max(life)))
    print("   年化: median=%.2f 年  p90=%.2f 年" % (qtl(life, .50) / 365.25, qtl(life, .90) / 365.25))
    names = ["<=1d", "1-7d", "7-30d", "30-365d", ">1y"]
    idx = [0 if d <= 1 else 1 if d <= 7 else 2 if d <= 30 else 3 if d <= 365 else 4 for d in life]
    c = Counter(idx)
    for i, n in enumerate(names):
        print("   %-8s %4d (%.1f%%)" % (n, c.get(i, 0), 100 * c.get(i, 0) / len(life)))
    print("   ★ 存活 <=1 天（近乎即时撤销/同日修正）: %d (%.1f%%)"
          % (c.get(0, 0), 100 * c.get(0, 0) / len(life)))
    print("\n   -- 按属性分组 --")
    byp = defaultdict(list)
    for r in ok:
        if r.get("lifespan_days") not in (None, ""):
            byp[r["pid"]].append(float(r["lifespan_days"]))
    for pid, xs in sorted(byp.items()):
        print("   %-5s n=%2d  median=%8.2f 天  p90=%8.2f 天" % (pid, len(xs), qtl(xs, .5), qtl(xs, .9)))

print("\n== 3) 翻转修订的编辑标签分布（编辑机制）==")
tags = Counter()
for r in ok:
    cm = r.get("flip_comment", "") or ""
    mm = re.search(r"wbsetclaim-[a-z]+|wbremoveclaims|rest-wbsetclaim-\w+|revert", cm)
    tags[mm.group(0) if mm else ("(no tag)" if not cm else cm[:24])] += 1
for k, v in tags.most_common(10):
    print("   %-32s %d" % (k, v))

print("\n== 4) ★ 机制核验：裁定那一刻做了什么（REVISE vs RETRACT）==")
mech_path = BASE / "E9_机制核验.csv"
mech = {}
mech_rows = []
if mech_path.exists():
    mech_rows = list(csv.DictReader(open(mech_path, encoding="utf-8-sig")))
    valid = [r for r in mech_rows if not str(r.get("kind", "")).startswith("ERROR")]
    mc = Counter(r["kind"] for r in valid)
    for k, v in mc.most_common():
        print("   %-20s %3d (%.1f%%)" % (k, v, 100 * v / max(1, len(valid))))
    mech = dict(mc)
    # flip 版里该属性仍存在的 active 声明数
    va = Counter(str(r.get("n_active_after", "")) for r in valid)
    print("   flip 版该属性仍有 active 声明数：%s" % dict(va))
    zero = va.get("0", 0)
    print("   ⇒ 撤回后该属性**无**可用值（纯撤回，替代值另行补入）：%d/%d (%.1f%%)"
          % (zero, len(valid), 100 * zero / max(1, len(valid))))
    print("   ⇒ 同编辑内引入「替换新值」的条数：%d" % mc.get("REVISE_NEW_VALUE", 0))
    mech["n_active_after_0_pct"] = round(100 * zero / max(1, len(valid)), 1)
else:
    print("   （E9_机制核验.csv 未生成，跳过）")

print("\n== 5) ★ 与 E8 代理的交叉校验（P813 retrieved 是否为上界）==")
# 5a) 用 E8 样本文件里的 retrieved（覆盖低，仅少量可比）
both = [(r, pdate(r["flip_ts"]), pdate(r["e8_retrieved"])) for r in ok if r.get("e8_retrieved")]
both = [(r, f, ret) for r, f, ret in both if f and ret]
p813 = {}
if both:
    lt = [x for x in both if x[2] < x[1]]
    ge = [x for x in both if x[2] >= x[1]]
    print("[5a: 来自 E8 样本文件] 可比对 %d 条：retrieved < flip %d (%.1f%%)，>= flip %d (%.1f%%)"
          % (len(both), len(lt), 100 * len(lt) / len(both), len(ge), 100 * len(ge) / len(both)))
    for r, f, ret in sorted(lt, key=lambda x: (x[2] - x[1]).days)[:5]:
        print("   ★反例 %-12s %-5s flip=%s retrieved=%s（差 %+d 天）"
              % (r["item"], r["pid"], f.date(), ret.date(), (f - ret).days))
    p813 = {"comparable": len(both), "retrieved_lt_flip": len(lt), "retrieved_ge_flip": len(ge)}
else:
    print("[5a: 来自 E8 样本文件] 无可比对条数")

# 5b) ★ 来自 flip 版实体 JSON 的 P813（限定符 / 来源），可比样本大得多
print("[5b: 来自 flip 版实体 JSON 的 P813 —— 主结论以此为准]")
if mech_rows:
    cmp2 = []
    for r in mech_rows:
        f_ = pdate(r.get("flip_ts", ""))
        ret = pdate(r.get("q_P813") or r.get("r_P813") or "")
        if f_ and ret:
            cmp2.append((r, f_, ret))
    if cmp2:
        lt2 = [x for x in cmp2 if x[2] < x[1]]
        print("   有 P813 的条目 %d/%d；可比对 %d 条：retrieved < flip %d (%.1f%%) ⇒ P813 %s上界"
              % (sum(1 for r in mech_rows if (r.get("q_P813") or r.get("r_P813"))), len(mech_rows),
                 len(cmp2), len(lt2), 100 * len(lt2) / len(cmp2), "**不是**" if lt2 else "是"))
        for r, f_, ret in sorted(lt2, key=lambda x: (x[2] - x[1]).days)[:6]:
            print("     ★反例 %-12s %-5s flip=%s retrieved=%s（差 %+d 天）"
                  % (r["item"], r["pid"], f_.date(), ret.date(), (f_ - ret).days))
        p813["from_flip_json"] = {"with_p813": sum(1 for r in mech_rows if (r.get("q_P813") or r.get("r_P813"))),
                                  "comparable": len(cmp2), "retrieved_lt_flip": len(lt2)}
    else:
        print("   无可比对条目")
else:
    print("   （E9_机制核验.csv 未生成，跳过）")

# 5c) P582(end time) 与 flip 的关系（计划性失效 vs 事后裁定）
if mech_rows:
    n582 = [r for r in mech_rows if r.get("q_P582")]
    if n582:
        print("   附：目标声明带 P582(end time) 的条目 %d/%d（计划性失效时刻，非裁定时刻）"
              % (len(n582), len(mech_rows)))

print("\n== 6) 成本与规模 ==")
calls = [int(r["n_api_calls"]) for r in ok if r.get("n_api_calls")]
revs = [int(r["n_revs"]) for r in ok if r.get("n_revs")]
if calls:
    print("API 调用/条：median=%.0f  mean=%.1f  max=%d  total=%d"
          % (st.median(calls), st.mean(calls), max(calls), sum(calls)))
if revs:
    print("页面修订数：median=%.0f  mean=%.0f  max=%d" % (st.median(revs), st.mean(revs), max(revs)))

out = {
    "n_total": len(rows), "n_usable_flip": len(usable), "n_full_lifetime": len(full),
    "n_ok": len(usable),
    "status": dict(Counter(r["status"][:60] for r in rows)),
    "flip_years": {str(k): v for k, v in sorted(yr.items())},
    "flip_ge_2017_pct": round(100 * sum(1 for _, d in flips if d.year >= 2017) / max(1, len(flips)), 1),
    "lifetime_days": {"n": len(life), "median": qtl(life, .50), "p90": qtl(life, .90),
                      "min": min(life) if life else None, "max": max(life) if life else None},
    "lifetime_le1d_pct": round(100 * Counter(0 if d <= 1 else 1 for d in life).get(0, 0) / max(1, len(life)), 1),
    "edit_tags": dict(tags.most_common(10)),
    "mechanism": mech,
    "p813_vs_flip": p813,
    "api_calls_median": st.median(calls) if calls else None,
    "api_calls_total": sum(calls) if calls else None,
}
json.dump(out, open(BASE / "E9_统计.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n[SAVE] E9_统计.json")
