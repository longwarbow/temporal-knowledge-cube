# -*- coding: utf-8 -*-
"""
E2 地基（P0-②）：CED user.location -> 行政区划码 消歧
- 粒度按 D11-rev1 默认 = 省级（属地为用户自填、只做省级可信层）
- 产出：省码映射表.csv / location_映射结果.csv / E2_消歧报告.md
"""
import os, io, json, re, collections

ORIG = r"D:\WSPforWB\10.keyan\场景1-海量短文本\数据\CED\Chinese_Rumor_Dataset-master\CED_Dataset\original-microblog"
OUT  = r"D:\WSPforWB\10.keyan\场景1-海量短文本\实验\E2_空间维消歧"
os.makedirs(OUT, exist_ok=True)

# GB/T 2260 省级码（36 省级单元 + 港澳台）；需在拿到 CTAmap 快照后交叉核对
PROV = [
    ("北京", 11, "北京市"), ("天津", 12, "天津市"), ("河北", 13, "河北省"), ("山西", 14, "山西省"),
    ("内蒙古", 15, "内蒙古自治区"), ("内蒙", 15, "内蒙古自治区"),
    ("辽宁", 21, "辽宁省"), ("吉林", 22, "吉林省"), ("黑龙江", 23, "黑龙江省"),
    ("上海", 31, "上海市"), ("江苏", 32, "江苏省"), ("浙江", 33, "浙江省"), ("安徽", 34, "安徽省"),
    ("福建", 35, "福建省"), ("江西", 36, "江西省"), ("山东", 37, "山东省"),
    ("河南", 41, "河南省"), ("湖北", 42, "湖北省"), ("湖南", 43, "湖南省"), ("广东", 44, "广东省"),
    ("广西", 45, "广西壮族自治区"), ("海南", 46, "海南省"), ("重庆", 50, "重庆市"),
    ("四川", 51, "四川省"), ("贵州", 52, "贵州省"), ("云南", 53, "云南省"), ("西藏", 54, "西藏自治区"),
    ("陕西", 61, "陕西省"), ("甘肃", 62, "甘肃省"), ("青海", 63, "青海省"),
    ("宁夏", 64, "宁夏回族自治区"), ("新疆", 65, "新疆维吾尔自治区"),
    ("台湾", 71, "台湾省"), ("香港", 81, "香港特别行政区"), ("澳门", 82, "澳门特别行政区"),
]
# 按长度降序，保证前缀匹配取最长（内蒙古 > 内蒙）
PREFIX = sorted(PROV, key=lambda x: -len(x[0]))
FULLMAP = {v[2]: (v[1], v[2]) for v in PROV}
NONMAINLAND = {"海外", "其他", "国外", "境外", "未知", "中国", "内地", "全国", "其它", "--", "-"}

def norm_token(t):
    return re.sub(r"[\s\u3000]+", "", t or "")

def to_province(tok):
    """返回 (code, full_name) 或 (None, reason)"""
    t = norm_token(tok)
    if not t:
        return None, "空"
    if t in FULLMAP:
        return FULLMAP[t], None
    if t in NONMAINLAND:
        return None, "非省级(境外/其他)"
    for ab, code, full in PREFIX:          # 前缀匹配（如 内蒙古自治区 / 广西壮族自治区）
        if t.startswith(ab):
            return (code, full), None
    return None, "未识别"

# ---------------- 读取 CED 属地 ----------------
rows = []
raw_counter = collections.Counter()
for fn in sorted(os.listdir(ORIG)):
    try:
        o = json.load(io.open(os.path.join(ORIG, fn), encoding="utf-8"))
    except Exception:
        continue
    if isinstance(o, list):
        o = o[0] if o else {}
    if not isinstance(o, dict):
        continue
    u = o.get("user") or {}
    loc = (u.get("location") or "").strip() if isinstance(u, dict) else ""
    raw_counter[loc] += 1
    parts = [p for p in re.split(r"[\s,，/|]+", loc) if p]
    prov_tok = parts[0] if parts else ""
    pref_tok = parts[1] if len(parts) >= 2 else ""
    (res, reason) = to_province(prov_tok)
    rows.append({
        "file": fn, "location_raw": loc, "prov_token": prov_tok, "pref_token": pref_tok,
        "prov_code": res[0] if res else "", "prov_name": res[1] if res else "",
        "status": "ok" if res else ("非省级" if reason == "非省级(境外/其他)" else ("空" if reason == "空" else "未识别")),
    })

n = len(rows)
n_empty = sum(1 for r in rows if r["status"] == "空")
n_ok = sum(1 for r in rows if r["status"] == "ok")
n_nm = sum(1 for r in rows if r["status"] == "非省级")
n_un = sum(1 for r in rows if r["status"] == "未识别")
n_two = sum(1 for r in rows if r["pref_token"])

LOG = []
def p(x=""):
    LOG.append(str(x))

p("# E2 消歧报告：CED `user.location` → 省级行政区划码（P0-②）")
p("")
p("> 生成：WorkBuddy ｜ 2026-10-09 ｜ 粒度 = 省级（D11-rev1 默认）｜ 码源 = GB/T 2260（待与 CTAmap 快照交叉核对）")
p("")
p("## 1. 覆盖情况（CED 原帖 %d 条）" % n)
p("")
p("| 类别 | 条数 | 占比 |")
p("|---|---|---|")
p("| ✅ 映射到省级码 | **%d** | **%.1f%%** |" % (n_ok, 100.0 * n_ok / n))
p("| 非省级（海外/其他/全国…） | %d | %.1f%% |" % (n_nm, 100.0 * n_nm / n))
p("| 空 | %d | %.1f%% |" % (n_empty, 100.0 * n_empty / n))
p("| 未识别 | %d | %.1f%% |" % (n_un, 100.0 * n_un / n))
p("| 含「省 市/区」级联（可用于市级尽力而为层） | %d | %.1f%% |" % (n_two, 100.0 * n_two / n))
p("")
p("## 2. 映射到省级码的分布（top 15）")
p("")
c = collections.Counter((r["prov_code"], r["prov_name"]) for r in rows if r["status"] == "ok")
p("| 码 | 省级 | 帖数 | 占比(占已映射) |")
p("|---|---|---|---|")
for (code, name), v in c.most_common(15):
    p("| %s | %s | %d | %.1f%% |" % (code, name, v, 100.0 * v / max(1, n_ok)))
p("")
p("## 3. 未识别 / 非省级 的原始取值（top 20）")
p("")
bad = collections.Counter(r["prov_token"] for r in rows if r["status"] in ("未识别", "非省级"))
p("| 原始首段 | 条数 | 判定 |")
p("|---|---|---|")
for tok, v in bad.most_common(20):
    st = to_province(tok)[1] or "ok"
    p("| %s | %d | %s |" % (tok if tok else "(空)", v, st))
p("")
p("## 4. 结论与用途")
p("")
p("- **属地角色（D14 角色 1）在 CED 上可落到省级码**：可用于空间维**省级 roll-up** 与「提及地≠属地」交叉（E2）。")
p("- **市级为尽力而为层**：%.1f%% 的属地可拆出第二段（市/区名），但**码未解析**（需 CTAmap 地级表），符合 D11-rev1 的低置信分层。" % (100.0 * n_two / n))
p("- **诚实声明**：性质 = **用户自填属地**（非 IP 属地、非帖级、2010–2014），须带 `geo_confidence=low`。")
p("- **待办**：拿到 CTAmap 省/地/县三级码表后，① 核对本表省级码；② 补市级码（尽力而为层）。")

io.open(os.path.join(OUT, "E2_消歧报告.md"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")

# 省码映射表
io.open(os.path.join(OUT, "省码映射表.csv"), "w", encoding="utf-8-sig").write(
    "abbr,code,full_name\n" + "\n".join("%s,%d,%s" % (a, c, f) for a, c, f in PROV) + "\n")
# 映射结果
io.open(os.path.join(OUT, "location_映射结果.csv"), "w", encoding="utf-8-sig").write(
    "file,location_raw,prov_token,pref_token,prov_code,prov_name,status\n" +
    "\n".join(",".join([r["file"], '"%s"' % r["location_raw"].replace('"', ""), r["prov_token"],
                        r["pref_token"], str(r["prov_code"]), r["prov_name"], r["status"]])
              for r in rows) + "\n")

print("DONE")
print("n=%d ok=%d nonmainland=%d empty=%d unknown=%d two_token=%d" % (n, n_ok, n_nm, n_empty, n_un, n_two))
