# -*- coding: utf-8 -*-
"""
E13 人类盲标 —— 评分脚本（**中文友好版**）

读：E13_人类盲标_作答.csv（填好后；表头/标签**中英皆可**） + E13_人类盲标_索引.json + E13_全量真值_v3.csv
算：
  · Cohen's kappa（人类 vs 冻结真值），含 bootstrap 95% CI
  · 一致率（raw agreement）
  · 逐类 P / R / F1 + 混淆矩阵
  · 分歧清单（逐条列出，便于回炉定位）
两遍分别评：直觉遍（不看指南）与依指南遍（依指南复核）。

用法：
  python e13_human_score.py                 # 两份都评
  python e13_human_score.py intuition       # 只评直觉遍
  python e13_human_score.py guided          # 只评依指南遍

标签写法（大小写不限、中英皆可）：
  DERIVE   派生 / 转述 / 引用 / DERIVE
  SUPPORT  支撑 / 支持 / SUPPORT
  CONFLICT 冲突 / CONFLICT
  UNRELATED 无关 / 不相关 / UNRELATED

阈值参考（Landis & Koch 1979）：<0.20 差 / 0.21-0.40 一般 / 0.41-0.60 中等
  / 0.61-0.80 显著 / 0.81-1.00 近乎完全
"""

import csv
import json
import os
import random
import sys
from collections import Counter

ANS = "E13_人类盲标_作答.csv"
ANS_XLSX = "E13_人类盲标_作答.xlsx"   # 若存在且作答更全，自动优先用它
IDX = "E13_人类盲标_索引.json"
TRUTH = "E13_全量真值_v3.csv"
LABELS = ["DERIVE", "SUPPORT", "CONFLICT", "UNRELATED"]
CN = {"DERIVE": "派生", "SUPPORT": "支撑", "CONFLICT": "冲突", "UNRELATED": "无关"}

# --- 中英文标签别名（大小写/空格不敏感）---
ALIAS = {}
for _canon, _keys in {
    "DERIVE": ["derive", "派生", "转述", "引用", "引述", "转发"],
    "SUPPORT": ["support", "支撑", "支持"],
    "CONFLICT": ["conflict", "冲突", "矛盾"],
    "UNRELATED": ["unrelated", "无关", "不相关", "无关的"],
}.items():
    for _k in _keys:
        ALIAS[_k] = _canon

# --- 列名别名（中英皆可；首位是该列的规范名，规范化后的表头也认）---
COL_ALIAS = {
    "qid": ["qid", "题号", "编号"],
    "intuition": ["intuition", "label_intuition", "直觉遍_判定", "直觉遍_标签", "直觉判定"],
    "conf_intuition": ["conf_intuition", "直觉遍_置信度", "直觉遍_置信", "直觉信心"],
    "note_intuition": ["note_intuition", "直觉遍_备注", "直觉备注"],
    "guided": ["guided", "label_guided", "依指南遍_判定", "依指南遍_标签", "指南判定"],
    "conf_guided": ["conf_guided", "依指南遍_置信度", "依指南遍_置信", "指南信心"],
    "note_guided": ["note_guided", "依指南遍_备注", "指南备注"],
}
CANON = list(COL_ALIAS.keys())


def norm_label(v):
    s = (v or "").strip()
    if not s:
        return ""
    return ALIAS.get(s.lower(), s.upper())


def conf_num(v):
    """置信度归一化：兼容 int / float / str（xlsx 下拉会存成数字，csv 里是字符串）"""
    s = str(v if v is not None else "").strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return int(f) if abs(f - int(f)) < 1e-9 else f


def resolve_cols(fieldnames, required=("qid",)):
    norm = {(f or "").lstrip("\ufeff").strip(): f for f in fieldnames}
    out = {}
    for key, cands in COL_ALIAS.items():
        hit = ""
        for c in cands:
            if c in norm:
                hit = norm[c]
                break
        out[key] = hit
    missing = [k for k in required if not out[k]]
    if missing:
        raise SystemExit("!! 作答文件缺少必需列：%s（现有表头：%s）" % (missing, list(norm)))
    return out


def _read_csv_src(path):
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        fns = [(x or "").lstrip("\ufeff").strip() for x in (rd.fieldnames or [])]
        rows = [{(k or "").lstrip("\ufeff").strip(): (v if v is not None else "")
                 for k, v in r.items()} for r in rd]
    return fns, rows


def _read_xlsx_src(path):
    """遍历**所有**工作表，按列名把两遍的作答合并成一份（返回规范列名）"""
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=True)
    merged = {}
    scanned = 0
    for ws in wb.worksheets:
        vals = [[("" if c is None else str(c).strip()) for c in row] for row in ws.values]
        vals = [r for r in vals if any(r)]
        if len(vals) < 2:
            continue
        head = [(h or "").lstrip("\ufeff").strip() for h in vals[0]]
        try:
            C = resolve_cols(head)          # 非答题表（如「怎么填」页）会在此被跳过
        except SystemExit:
            continue
        scanned += 1
        for r in vals[1:]:
            d = {head[i]: (r[i] if i < len(r) else "") for i in range(len(head))}
            qid = (d.get(C["qid"]) or "").strip()
            if not qid:
                continue
            tgt = merged.setdefault(qid, {k: "" for k in CANON})
            tgt["qid"] = qid
            for k in CANON:
                if k != "qid" and C[k] and (d.get(C[k]) or "").strip():
                    tgt[k] = d[C[k]].strip()
    print("       （xlsx 里识别到 %d 张答题表）" % scanned)
    return CANON, list(merged.values())


def _src_score(fns, rows):
    """该来源里「能识别成合法标签」的作答格数（用于判断哪个来源是主源）"""
    try:
        C = resolve_cols(fns)
    except SystemExit:
        return -1, None
    n = 0
    for r in rows:
        if not (r.get(C["qid"]) or "").strip():
            continue
        for k in ("intuition", "guided"):
            if C[k] and norm_label(r.get(C[k])) in LABELS:
                n += 1
    return n, C


def _to_canon(C, rows):
    """把某来源的行转成统一规范列名（qid/intuition/guided/...）"""
    out = {}
    for r in rows:
        qid = (r.get(C["qid"]) or "").strip()
        if not qid:
            continue
        row = {"qid": qid}
        for k in CANON:
            if k != "qid":
                row[k] = (r.get(C[k]) or "").strip() if C[k] else ""
        out[qid] = row
    return out


def pick_answers():
    """csv 与 xlsx 都读：主源 = 有效作答更多的那份（同分优先 xlsx）；
    另一份**只回填空缺、绝不覆盖**，因此两份各填一半也不会丢。"""
    cands = []
    for tag, path, reader in [("csv", ANS, _read_csv_src), ("xlsx", ANS_XLSX, _read_xlsx_src)]:
        if not os.path.exists(path):
            continue
        try:
            fns, rows = reader(path)
        except Exception as e:
            print("  !! 读取 %s 失败：%s" % (path, e))
            continue
        s, C = _src_score(fns, rows)
        print("  发现 %-4s  %-26s  有效作答格=%-4d%s" % (tag, os.path.basename(path),
                                                   max(s, 0), "" if C is not None else "（无题号列，跳过）"))
        cands.append({"tag": tag, "path": path, "score": max(s, 0), "C": C, "rows": rows})
    good = [c for c in cands if c["C"] is not None]
    if not good:
        raise SystemExit("!! 找不到可用的作答文件（%s / %s）" % (ANS, ANS_XLSX))
    good.sort(key=lambda c: (-c["score"], 0 if c["tag"] == "xlsx" else 1))
    main = good[0]
    print("  ⇒ 主源：%s（%s）" % (main["tag"].upper(), os.path.basename(main["path"])))
    ans = _to_canon(main["C"], main["rows"])
    for other in good[1:]:
        o = _to_canon(other["C"], other["rows"])
        back = 0
        for qid, r in o.items():
            tgt = ans.setdefault(qid, {"qid": qid})
            for k in CANON:
                if k != "qid" and not (tgt.get(k) or "") and (r.get(k) or ""):
                    tgt[k] = r[k]
                    back += 1
        if back:
            print("      从 %s 回填 %d 格（只填空缺，不覆盖主源）" % (other["tag"].upper(), back))
    return ans, {k: k for k in CANON}


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in LABELS) / (n * n)
    if abs(1 - pe) < 1e-12:
        return 1.0
    return (po - pe) / (1 - pe)


def bootstrap_ci(a, b, iters=4000, seed=20261012):
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        k = cohen_kappa([a[i] for i in idx], [b[i] for i in idx])
        if k == k:
            vals.append(k)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def prf(a, b):
    """a=人类, b=真值"""
    out = {}
    for l in LABELS:
        tp = sum(1 for x, y in zip(a, b) if x == l and y == l)
        fp = sum(1 for x, y in zip(a, b) if x == l and y != l)
        fn = sum(1 for x, y in zip(a, b) if x != l and y == l)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[l] = (p, r, f, tp + fn)
    return out


def evaluate(tag, pairs, confs=None):
    a = [h for _, h, _ in pairs]
    b = [t for _, _, t in pairs]
    if confs is None:
        confs = [None] * len(pairs)
    n = len(pairs)
    k = cohen_kappa(a, b)
    ci_lo, ci_hi = bootstrap_ci(a, b)
    raw = sum(1 for x, y in zip(a, b) if x == y) / n if n else float("nan")
    print("\n" + "=" * 72)
    print("[%s]  n=%d" % (tag, n))
    print("  Cohen's kappa = %.4f   (bootstrap 95%% CI: %.4f - %.4f)" % (k, ci_lo, ci_hi))
    print("  一致率 raw agreement = %.4f" % raw)
    if n < 10:
        print("  ⚠️ 目前只评到 %d 条，κ 的数值意义有限（先当链路自检看；填满 48 条才是正式结果）" % n)
    if k >= 0.81:
        print("  强度：近乎完全（≥0.81）")
    elif k >= 0.61:
        print("  强度：显著（0.61-0.80）")
    elif k >= 0.41:
        print("  强度：中等（0.41-0.60）")
    else:
        print("  强度：一般或更差（<0.41）")
    print("  逐类（人类=预测，真值=参考；n = 该类真值数）：")
    m = prf(a, b)
    for l in LABELS:
        p, r, f, nn = m[l]
        print("    %-10s(%s)  P=%.3f  R=%.3f  F1=%.3f   n=%d" % (l, CN[l], p, r, f, nn))
    macro_f = sum(m[l][2] for l in LABELS) / 4
    print("    macro-F1 = %.4f" % macro_f)
    print("  混淆矩阵（行=真值，列=人类）：")
    print("            " + "".join(("%s/%s" % (l, CN[l])).ljust(14) for l in LABELS))
    for t in LABELS:
        row = [sum(1 for x, y in zip(a, b) if y == t and x == h) for h in LABELS]
        print("    " + ("%s/%s" % (t, CN[t])).ljust(14) + "".join("%-14d" % v for v in row))
    dis = [(q, h, t) for q, h, t in pairs if h != t]
    if dis:
        print("  分歧 %d 条：" % len(dis))
        for q, h, t in dis:
            print("    %s  人类=%-9s(%s)  真值=%-9s(%s)" % (q, h, CN[h], t, CN[t]))

    # ---- 置信度校准（协议 §7 的「高置信子集」；顺带检验置信度列是否真有判别力）----
    calib = None
    if any(c is not None for c in confs):
        hi = [(q, h, t) for (q, h, t), c in zip(pairs, confs) if c is not None and c >= 2]
        lo = [(q, h, t) for (q, h, t), c in zip(pairs, confs) if c is not None and c < 2]
        un = [(q, h, t) for (q, h, t), c in zip(pairs, confs) if c is None]
        print("\n  置信度分层：")
        print("    高置信(2-3): n=%-3d  分歧率=%.3f%s"
              % (len(hi), (sum(1 for _, h, t in hi if h != t) / len(hi)) if hi else float("nan"),
                 ("  κ=%.4f" % cohen_kappa([h for _, h, _ in hi], [t for _, _, t in hi])) if len(hi) >= 2 else ""))
        print("    低置信(1)  : n=%-3d  分歧率=%.3f%s"
              % (len(lo), (sum(1 for _, h, t in lo if h != t) / len(lo)) if lo else float("nan"),
                 ("  κ=%.4f" % cohen_kappa([h for _, h, _ in lo], [t for _, _, t in lo])) if len(lo) >= 2 else ""))
        if un:
            print("    未填置信度: n=%d（已排除在上两行之外）" % len(un))
        if hi and lo:
            d_hi = sum(1 for _, h, t in hi if h != t) / len(hi)
            d_lo = sum(1 for _, h, t in lo if h != t) / len(lo)
            if d_lo > d_hi:
                print("    ⇒ 低置信处分歧率更高（%.3f > %.3f）⇒ 你的置信度**有判别力**（不是随手填的）" % (d_lo, d_hi))
            else:
                print("    ⇒ 两者分歧率接近（%.3f vs %.3f）⇒ 置信度**未带来额外信息**，需在论文中如实说明" % (d_lo, d_hi))
        if lo:
            print("    低置信清单（判据边界候选，回炉 v1.7 优先看这些）：")
            for q, h, t in lo:
                print("      %s  人类=%-9s(%s)  真值=%-9s(%s)%s"
                      % (q, h, CN[h], t, CN[t], "  ← 也分歧" if h != t else ""))
        calib = {"n_high": len(hi), "n_low": len(lo), "n_unset": len(un),
                 "disagree_high": sum(1 for _, h, t in hi if h != t),
                 "disagree_low": sum(1 for _, h, t in lo if h != t),
                 "low_conf_qids": [q for q, _, _ in lo]}

    return {"n": n, "kappa": k, "ci": [ci_lo, ci_hi], "raw": raw,
            "per_class": {l: {"P": m[l][0], "R": m[l][1], "F1": m[l][2], "n": m[l][3]} for l in LABELS},
            "macro_F1": macro_f,
            "calibration": calib,
            "disagreements": [{"qid": q, "human": h, "truth": t} for q, h, t in dis]}


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else "both"
    idx = json.load(open(IDX, encoding="utf-8"))["index"]
    truth = {r["pair_id"]: r["truth"] for r in csv.DictReader(open(TRUTH, encoding="utf-8-sig"))}
    ans, C = pick_answers()
    print("       （列名已归一为：题号 / 直觉遍_判定 / 依指南遍_判定 …）")
    if not (C.get("intuition") or C.get("guided")):
        raise SystemExit("!! 找不到任何判定列（直觉遍_判定 / 依指南遍_判定）")

    result = {}
    for key, tag in [("intuition", "直觉遍（不看指南）"), ("guided", "依指南遍")]:
        if want not in ("both", key):
            continue
        col = C[key]
        if not col:
            print("\n[%s] 作答文件里没有这一列，跳过。" % tag)
            continue
        conf_col = C.get("conf_intuition" if key == "intuition" else "conf_guided")
        pairs, bad, confs = [], [], []
        for qid, meta in idx.items():
            h = norm_label(ans.get(qid, {}).get(col))
            if not h:
                continue
            if h not in LABELS:
                bad.append((qid, ans.get(qid, {}).get(col)))
                continue
            pairs.append((qid, h, truth[meta["pair_id"]]))
            confs.append(conf_num(ans.get(qid, {}).get(conf_col)) if conf_col else None)
        for qid, raw in bad:
            print("  !! %s 的「%s」列填了看不懂的值 %r，已跳过（请改用 派生/支撑/冲突/无关）"
                  % (qid, tag, raw))
        if not pairs:
            print("\n[%s] 尚无有效作答（%s 列为空），跳过。" % (tag, col))
            continue
        result[key] = evaluate(tag, pairs, confs)

    if result:
        json.dump(result, open("E13_人类盲标_评分.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n已写出 E13_人类盲标_评分.json")
    else:
        print("\n没有可评的内容。")


if __name__ == "__main__":
    main()
