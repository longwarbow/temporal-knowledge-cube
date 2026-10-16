# -*- coding: utf-8 -*-
"""
E13 人类盲标 —— 把 48 道题目并进答题卡（一个文件填完，不必横向滚动）

输入：E13_人类盲标_题目_中文.md   中英对照题面（唯一真源）
      E13_人类盲标_作答.csv / .xlsx   已有作答（若有，按题号保留；没有就留空）
输出：E13_人类盲标_作答.csv    12 列，含题面四列（纯文本版）
      E13_人类盲标_作答.xlsx   **两遍各一张表**，每张 5 列：
                              题号 | 题面 | <遍>_判定 | <遍>_置信度 | <遍>_备注
                              + 「怎么填」说明页

设计说明（v2，2026-10-12）：
  旧版把 12 列塞在一张表里、并冻结前 6 列 ⇒ 冻结区宽约 210 字符，
  窄视口下把右侧的填写列整个挤出屏幕，且横向滚动形同虚设。
  新版改为一屏可容的 5 列布局，**只冻结表头行、不冻结列**；
  两遍分表还顺带保证「第二遍不会看到第一遍的答案」（双遍设计的本意）。

用法： python e13_merge_qa.py
"""

import csv
import os
import re

QMD = "E13_人类盲标_题目_中文.md"
ANS_CSV = "E13_人类盲标_作答.csv"
ANS_XLSX = "E13_人类盲标_作答.xlsx"

ANS_COLS = ["直觉遍_判定", "直觉遍_置信度", "直觉遍_备注",
            "依指南遍_判定", "依指南遍_置信度", "依指南遍_备注"]
COLS = ["题号", "事件", "p_原文", "p_中文", "q_原文", "q_中文"] + ANS_COLS

QRE = re.compile(r"\n### (Q\d+)\s*·\s*事件：([^\n]+)\n(.*?)(?=\n---\n|\n## |\Z)", re.S)
PRE = re.compile(r"\*\*p（较早）\*\*\s*\n\s*-\s*原文：([^\n]*)\n\s*-\s*中文：([^\n]*)")
QRE2 = re.compile(r"\*\*q（较晚）\*\*\s*\n\s*-\s*原文：([^\n]*)\n\s*-\s*中文：([^\n]*)")


def one_line(s):
    """单元格里不保留换行（避免 Excel 行高爆炸），首尾空白去掉"""
    return re.sub(r"\s*\n\s*", " ", (s or "").strip()).strip()


def parse_questions():
    txt = open(QMD, encoding="utf-8").read()
    qs = {}
    for m in QRE.finditer(txt):
        qid, event, body = m.group(1), m.group(2).strip(), m.group(3)
        mp, mq = PRE.search(body), QRE2.search(body)
        if not (mp and mq):
            raise SystemExit("!! %s 题面解析失败（p/q 段缺失）" % qid)
        qs[qid] = {"事件": event,
                   "p_原文": one_line(mp.group(1)), "p_中文": one_line(mp.group(2)),
                   "q_原文": one_line(mq.group(1)), "q_中文": one_line(mq.group(2))}
        for k, v in qs[qid].items():
            if not v:
                raise SystemExit("!! %s 的「%s」解析为空，题面文件可能有格式变动" % (qid, k))
    order = sorted(qs, key=lambda x: int(x[1:]))
    if len(order) != 48:
        print("!! 警告：解析到 %d 题（预期 48）" % len(order))
    return order, qs


def face_text(q):
    """题面单列显示：p原 / p中 / q原 / q中 四段（段内换行）"""
    return ("p原文 ▸ %s\np中文 ▸ %s\nq原文 ▸ %s\nq中文 ▸ %s"
            % (q["p_原文"], q["p_中文"], q["q_原文"], q["q_中文"]))


def _disp_w(s):
    """近似显示宽度：CJK 字符按 2 计"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)


def _face_lines(q, width):
    """估算题面格占几行（用于设行高，避免预览器里被截断）"""
    n = 0
    for seg in ("p原文 ▸ " + q["p_原文"], "p中文 ▸ " + q["p_中文"],
                "q原文 ▸ " + q["q_原文"], "q中文 ▸ " + q["q_中文"]):
        n += max(1, -(-_disp_w(seg) // width))
    return n


def _rows_from_csv(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {(k or "").lstrip("\ufeff").strip(): (v or "") for k, v in r.items()}
            qid = (r.get("题号") or r.get("qid") or "").strip()
            if qid:
                out[qid] = {c: (r.get(c) or "").strip() for c in ANS_COLS}
    return out


def _rows_from_xlsx(path):
    """遍历**所有**工作表按列名收集作答（新旧结构都兼容）"""
    out = {}
    if not os.path.exists(path):
        return out
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=True)
    for ws in wb.worksheets:
        vals = [[("" if c is None else str(c).strip()) for c in row] for row in ws.values]
        vals = [r for r in vals if any(r)]
        if len(vals) < 2:
            continue
        head = [(h or "").lstrip("\ufeff").strip() for h in vals[0]]
        if "题号" not in head and "qid" not in head:
            continue
        for r in vals[1:]:
            d = {head[i]: (r[i] if i < len(r) else "") for i in range(len(head))}
            qid = (d.get("题号") or d.get("qid") or "").strip()
            if not qid:
                continue
            tgt = out.setdefault(qid, {c: "" for c in ANS_COLS})
            for c in ANS_COLS:
                if (d.get(c) or "").strip():
                    tgt[c] = d[c].strip()
    return out


def load_old_answers():
    """已有作答按题号保留：csv 与 xlsx 都读，逐格取第一个非空值（改题面不会丢填写）"""
    srcs = [_rows_from_csv(ANS_CSV), _rows_from_xlsx(ANS_XLSX)]
    keep = {}
    for qid in set(s for d in srcs for s in d):
        row = {}
        for c in ANS_COLS:
            vals = [d.get(qid, {}).get(c, "") for d in srcs]
            row[c] = next((v for v in vals if v), "")
        keep[qid] = row
    filled = [q for q, v in keep.items() if any(v.values())]
    return keep, filled


def write_csv(order, qs, old):
    with open(ANS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for qid in order:
            row = {"题号": qid}
            row.update(qs[qid])
            for c in ANS_COLS:
                row[c] = old.get(qid, {}).get(c, "")
            w.writerow(row)
    print("已写出 %s（%d 行 × %d 列）" % (ANS_CSV, len(order), len(COLS)))


def write_xlsx(order, qs, old):
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation

    FACE_W = 78                      # 题面列宽（字符）
    WIDTHS = [6, FACE_W, 10, 9, 20]  # A 题号 / B 题面 / C 判定 / D 置信度 / E 备注
    head_fill = PatternFill("solid", fgColor="DCE6F1")
    fill_ans = PatternFill("solid", fgColor="FFF2CC")
    head_font = Font(bold=True, size=11)
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    waves = [
        ("第1遍_直觉", "直觉遍",
         "【第 1 遍 · 直觉】只看 B 列题面，凭语感判，填 C/D/E 列。\n"
         "先别打开速查卡、也先别看更多规则——这一遍要的是「第一反应」。",
         "备注（**可留空**，不算分）：只在**犹豫过的题**上写才划算。\n"
         "用大白话写「为什么这么判」，例如「q 像在复述 p 的话」「一边说 11 死、一边说 10 死」。\n"
         "「几乎一样」「两件事」这类一眼就定的可以省。\n"
         "这一遍**不要写条款号**——写了就等于提前用了指南，会污染「直觉」这个测量。"),
        ("第2遍_依指南", "依指南遍",
         "【第 2 遍 · 依指南】打开 E13_人类盲标_速查卡.md，\n"
         "按判定顺序 ①→⑤ 逐条走一遍（第一个命中的即结论），再填 C/D/E 列。",
         "备注（**可留空**，不算分）：只在**犹豫过的题**上写。\n"
         "写命中的条款号，速查卡上可直接照抄。\n"
         "T1 / T2 / T3 / A4-同族 CUSTODY_EXITED / A4-不同族 FATALITY↔INJURY / "
         "A3-ⓠ / A3-ⓑ / A5 / ④可追溯 / ⑤兜底 /\n"
         "pred-out-of-table: xxx / needs-world-knowledge / insufficient-info"),
    ]

    wb = Workbook()
    wb.remove(wb.active)

    for sheet_name, prefix, howto, note_txt in waves:
        ws = wb.create_sheet(sheet_name)
        cols = ["题号", "题面", prefix + "_判定", prefix + "_置信度", prefix + "_备注"]
        ws.append(cols)
        for c in range(1, 6):
            cell = ws.cell(row=1, column=c)
            cell.fill = head_fill
            cell.font = head_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border
        ws.row_dimensions[1].height = 30

        for coord, txt, w, h in [
            ("C1", howto, 340, 110),
            ("D1", "置信度 = 你对「自己这个判定」有多确定，\n"
                   "**不是**「这条推文说的事是不是真的（谣言不谣言）」。\n"
                   "1 低 / 2 中 / 3 高。\n"
                   "例：判了【冲突】但心里觉得「这两条也许算相容」⇒ 填 1。\n"
                   "拿不准就如实写低——低置信样本会被单独统计，不会被删。", 380, 140),
            ("E1", note_txt, 380, 140),
        ]:
            cm = Comment(txt, "E13 判据")
            cm.width, cm.height = w, h
            ws[coord].comment = cm

        for qid in order:
            ws.append([qid, face_text(qs[qid]),
                       old.get(qid, {}).get(prefix + "_判定", ""),
                       old.get(qid, {}).get(prefix + "_置信度", ""),
                       old.get(qid, {}).get(prefix + "_备注", "")])

        for i in range(2, len(order) + 2):
            q = qs[order[i - 2]]
            ws.row_dimensions[i].height = max(58, min(240, _face_lines(q, FACE_W) * 14.5 + 6))
            for c in range(1, 6):
                cell = ws.cell(row=i, column=c)
                cell.border = border
                if c == 1:
                    cell.alignment = Alignment(horizontal="center", vertical="top")
                    cell.font = Font(bold=True)
                elif c == 2:
                    cell.alignment = Alignment(wrap_text=True, vertical="top", horizontal="left")
                else:
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    cell.fill = fill_ans

        for col, w in zip("ABCDE", WIDTHS):
            ws.column_dimensions[col].width = w
        # 只冻结表头行。**不冻结列**——冻结列在窄视口下会把右侧的填写列挤出屏幕。
        ws.freeze_panes = "A2"

        dv_lab = DataValidation(type="list", formula1='"派生,支撑,冲突,无关"',
                                allow_blank=True, showDropDown=False)
        dv_con = DataValidation(type="list", formula1='"1,2,3"',
                                allow_blank=True, showDropDown=False)
        ws.add_data_validation(dv_lab)
        ws.add_data_validation(dv_con)
        last = len(order) + 1
        dv_lab.add("C2:C%d" % last)
        dv_con.add("D2:D%d" % last)

    # ---------- 说明页 ----------
    ws2 = wb.create_sheet("怎么填")
    guide = [
        ["E13 人类盲标 · 填写说明（48 题 × 2 遍）", ""],
        ["", ""],
        ["为什么分成两张表", "每张表只有 5 列（题号 / 题面 / 判定 / 置信度 / 备注），一屏放得下，不用横向滚动；两遍分表还保证第二遍不会先看到第一遍的答案。"],
        ["", ""],
        ["【第 1 遍 · 直觉】打开「第1遍_直觉」页", "只看 B 列题面，凭语感判，填 C/D/E 列。先别打开速查卡。"],
        ["【第 2 遍 · 依指南】打开「第2遍_依指南」页", "打开 E13_人类盲标_速查卡.md，按 ①→⑤ 判定顺序逐条走一遍（第一个命中的即结论）再填。"],
        ["", ""],
        ["判定四选一（C 列；有下拉框，手输中文或英文都认）", ""],
        ["派生 DERIVE", "q 是 p 的转述 / 引用 / 转发 / 同文复制，可追溯到 p"],
        ["支撑 SUPPORT", "q 独立断言与 p 相同的命题（或同向的更强/更弱版本），无引用痕迹"],
        ["冲突 CONFLICT", "q 与 p 不相容（不能同时为真）"],
        ["无关 UNRELATED", "q 在说别的事，从 q 推不出关于 p 命题的任何真值约束"],
        ["", ""],
        ["置信度（D 列）：1 低 / 2 中 / 3 高", "★ 问的是「**你对自己这个判定有多确定**」，不是「这条推文说的事是不是真的（谣言不谣言）」。"],
        ["   正例", "判了【冲突】，但心里觉得「这两条其实也可能算相容」⇒ 置信度写 1。"],
        ["   反例（这样填是错的）", "因为「这条看着像谣言」就把置信度写低 —— 与本题无关。"],
        ["   关键", "两条都是**假消息**，只要 q 是 p 的转述，照样判【派生】，置信度可以写 3。"],
        ["   为什么", "本题测的是 p 与 q 的**血缘关系**（能否追溯到同一命题 / 是否相容），与命题本身真假无关；真假属 PHEME veracity 那一套标注，本批样本大多没有真值。"],
        ["   拿不准就如实写低", "低置信样本会被单独统计，不会被删 —— 这是诊断判据边界的关键信息。"],
        ["", ""],
        ["【备注列 E 怎么写】", "**可以完全留空**（备注不算分）。它唯一的作用是：把你的判定与判据/机器不一致的地方，定位到「哪条判据的哪一处」。"],
        ["   ✅ 犹豫过的题：一定要写", "「但…存疑」「我不确定是不是同一件事」「这两条到底什么关系？」—— 这类是判据裂缝的直接证据。"],
        ["   ⬜ 一眼就定的题：可以省", "「几乎一样」「两件事」这类确认型备注价值低（结论已经在判定列里了）。"],
        ["第 1 遍的备注（可选）", "大白话写理由：「q 像在复述 p 的话」「一边说 11 死、一边说 10 死」。**不要写条款号**（写了就等于提前用指南，直觉那一遍就白做了）。"],
        ["第 2 遍的备注（可留空，犹豫的题才写）", "写命中的条款号，速查卡上可直接照抄；一行四个字也够定位："],
        ["   格式示例", "T1 无事实命题｜T2 只否定隐含命题｜T3 主体不同｜A4-同族 CUSTODY_EXITED｜A4-不同族 FATALITY↔INJURY｜A3-ⓠ 计量对象不同类｜A3-ⓑ 总数 vs 总数｜A5 收件人剥离 / 来源标注｜④可追溯｜⑤兜底"],
        ["   三种「必写」标记", "pred-out-of-table: <谓词>（落不进 10 族表）｜needs-world-knowledge（要外部知识才能读懂）｜insufficient-info（过短/纯链接/纯表情）"],
        ["", ""],
        ["题面（B 列）怎么读", "每格四段：p原文 / p中文 / q原文 / q中文（p 是较早那条，q 是较晚那条）。涉及面层线索（句首 @、via @X / RT @X、具体数字）时**以英文原文为准**。"],
        ["两条盲性纪律", "① 不要打开 E13_全量真值_v3.csv（答案）；② 不要打开 E13_系统预测_v4_全量.csv（预测）。"],
        ["填完保存，说一声即可", "评分：e13_human_score.py（Cohen's κ + bootstrap 95% CI + 逐类 P/R/F1 + 分歧清单）"],
    ]
    for row in guide:
        ws2.append(row)
    ws2.column_dimensions["A"].width = 46
    ws2.column_dimensions["B"].width = 86
    ws2["A1"].font = Font(bold=True, size=13)
    for i, row in enumerate(guide, start=1):
        for c in (1, 2):
            ws2.cell(row=i, column=c).alignment = Alignment(wrap_text=True, vertical="top")
        if row[1] == "" and row[0]:
            ws2.cell(row=i, column=1).font = Font(bold=True)

    try:
        wb.save(ANS_XLSX)
    except PermissionError:
        # 文件被 Excel/WPS/预览 占用 ⇒ 降级写到备用名，绝不假定可覆盖
        alt = ANS_XLSX.replace(".xlsx", "_新.xlsx")
        wb.save(alt)
        print("!! %s 正被打开（占用），新版已写到 %s\n"
              "   关掉那个文件后，把它改名成 %s 即可（或再跑一次本脚本自动替换）。"
              % (ANS_XLSX, alt, ANS_XLSX))
        return
    print("已写出 %s（3 个工作表：第1遍_直觉 / 第2遍_依指南 / 怎么填；每表 5 列，不需横向滚动）" % ANS_XLSX)


def main():
    order, qs = parse_questions()
    old, filled = load_old_answers()
    if filled:
        print("检测到已有作答 %d 条，将原样保留：%s" % (len(filled), ",".join(sorted(filled)[:8])))
    else:
        print("作答卡当前为空，生成全新空白表格。")
    write_csv(order, qs, old)
    try:
        write_xlsx(order, qs, old)
    except ImportError as e:
        print("!! 未安装 openpyxl，跳过 xlsx：%s" % e)
    print("完成。")


if __name__ == "__main__":
    main()
