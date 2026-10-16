# -*- coding: utf-8 -*-
"""E8 主画像 v2：Wikidata deprecated rank = 声明级 `retracted` 真值。

实测出的 WDQS 计划规律（本次实验的技术发现，写进报告）：
  - 纯 rank 索引锚定：全局 deprecated 计数 0.46s（1097 万条）；
  - `?item p:Pxxx ?st . ?st rank ?r` 形状：rank 访问 = 每声明一次
    statement-node 读，大属性（>3M 声明）必 504；
  - reason 锚定（pq:P2241 先行）可救回性质级查询；
  - total 计数（纯 p 索引扫描）恒快。

产出：
  1) 全局头条：deprecated 总量；
  2) 16 属性：total / dep(精确或带原因下界) / 撤回率；
  3) P2241 原因分布 + 标签；
  4) 声明级 (s,r,o) 样本 6 属性 × 500（G8 数据基础）；
  5) 裁定时刻代理：P813(retrieved) 年度分布 + P582(end time) 覆盖（G7）。
"""
import csv, json, subprocess, time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "Scene1-TemporalKnowledgeCube-E8/0.1 (research; contact: longwarbow@126.com)"

# (pid, gloss) —— 人类/组织/政治/体育高发降级属性；>3M 声明者只取下界
PROPS = [
    ("P569", "birth date"), ("P570", "death date"), ("P26", "spouse"),
    ("P22", "father"), ("P25", "mother"), ("P40", "child"),
    ("P27", "citizenship"), ("P108", "employer"), ("P69", "educated at"),
    ("P39", "position held"), ("P19", "place of birth"), ("P20", "place of death"),
    ("P119", "burial place"), ("P159", "headquarters location"),
    ("P127", "owned by"), ("P54", "member of sports team"),
]
HEAD6 = ("P569", "P570", "P26", "P108", "P69", "P39")
GIANT_TOTAL = 3_000_000  # 超过则跳过精确 dep 计数（必 504），用带原因下界

Q_TOTAL = "SELECT (COUNT(*) AS ?n) WHERE { ?item p:%s ?st . }"
Q_DEP = ("SELECT (COUNT(*) AS ?n) WHERE { ?item p:%s ?st . "
         "?st wikibase:rank wikibase:DeprecatedRank . }")
Q_REASONS = ("SELECT ?reason (COUNT(*) AS ?n) WHERE { ?item p:%s ?st . "
             "?st wikibase:rank wikibase:DeprecatedRank ; pq:P2241 ?reason . } "
             "GROUP BY ?reason ORDER BY DESC(?n)")
# 声明级样本：先试无锚定（信息更全），504 则退 reason 锚定
Q_SAMPLE = ("SELECT ?item ?value ?reason ?endtime ?retrieved WHERE { "
            "?item p:%s ?st . ?st wikibase:rank wikibase:DeprecatedRank ; "
            "ps:%s ?value . "
            "OPTIONAL { ?st pq:P2241 ?reason } "
            "OPTIONAL { ?st pq:P582 ?endtime } "
            "OPTIONAL { ?st prov:wasDerivedFrom ?ref . ?ref pr:P813 ?retrieved } "
            "} LIMIT %d")
Q_SAMPLE_ANCHORED = Q_SAMPLE  # 同形；锚定靠把 OPTIONAL 提升为必需——见下方构造
Q_SAMPLE_ANCH = ("SELECT ?item ?value ?reason ?endtime ?retrieved WHERE { "
                 "?item p:%s ?st . ?st wikibase:rank wikibase:DeprecatedRank ; "
                 "pq:P2241 ?reason ; ps:%s ?value . "
                 "OPTIONAL { ?st pq:P582 ?endtime } "
                 "OPTIONAL { ?st prov:wasDerivedFrom ?ref . ?ref pr:P813 ?retrieved } "
                 "} LIMIT %d")
Q_P813_YEAR = ("SELECT ?yr (COUNT(*) AS ?n) WHERE { ?item p:%s ?st . "
               "?st wikibase:rank wikibase:DeprecatedRank ; pq:P2241 ?rs ; "
               "prov:wasDerivedFrom ?ref . ?ref pr:P813 ?r . BIND(year(?r) AS ?yr) } "
               "GROUP BY ?yr ORDER BY ?yr")
Q_P582_CNT = ("SELECT (COUNT(*) AS ?n) WHERE { ?item p:%s ?st . "
              "?st wikibase:rank wikibase:DeprecatedRank ; pq:P2241 ?rs ; pq:P582 ?et . }")


def sparql(query, timeout=110, retries=2):
    last_err = None
    for attempt in range(retries):
        cmd = ["curl", "-s", "--noproxy", "*", "-m", str(timeout), "-A", UA,
               "-H", "Accept: application/sparql-results+json",
               "-w", "\n%{http_code}",
               "--data-urlencode", "query=" + query,
               "--data-urlencode", "format=json", ENDPOINT]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout + 15)
            out = proc.stdout.decode("utf-8", "replace")
            body, _, code = out.rpartition("\n")
            if code.strip() == "200" and body.strip():
                return json.loads(body)["results"]["bindings"]
            last_err = "HTTP=%s %r" % (code.strip() or proc.returncode, body[:80])
            if code.strip() == "429":
                time.sleep(30)
        except Exception as e:
            last_err = "%s: %s" % (type(e).__name__, e)
        time.sleep(12 * (attempt + 1))
    raise RuntimeError("sparql fail: %s" % last_err)


def rows2d(rows):
    return [{k: v.get("value") for k, v in r.items()} for r in rows]


prop_rows, reason_rows, sample_rows, p813_rows, extra_rows = [], [], [], [], []


def _dump_all():
    def dump(name, rows, cols):
        with open(BASE / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    dump("E8_属性级撤回率.csv", prop_rows,
         ["pid", "gloss", "total_statements", "deprecated", "dep_is_lower_bound", "retraction_rate"])
    dump("E8_原因分布.csv", reason_rows, ["pid", "reason", "n"])
    dump("E8_声明级样本.csv", sample_rows,
         ["pid", "gloss", "anchored", "item", "value", "reason", "endtime", "retrieved"])
    dump("E8_P813年度分布.csv", p813_rows, ["pid", "year", "n"])
    dump("E8_辅助量.csv", extra_rows, ["key", "value"])


def headline():
    try:
        n = int(rows2d(sparql("SELECT (COUNT(*) AS ?n) WHERE { "
                              "?st wikibase:rank wikibase:DeprecatedRank . }"))[0]["n"])
        extra_rows.append({"key": "global_deprecated_statements", "value": n})
        print("[HEADLINE] global deprecated = %d" % n, flush=True)
    except Exception as e:
        print("[WARN] headline: %s" % e, flush=True)


def main():
    headline()
    _dump_all()
    for pid, gloss in PROPS:
        t0 = time.time()
        try:
            total = int(rows2d(sparql(Q_TOTAL % pid))[0]["n"])
        except Exception as e:
            print("[SKIP] %s total: %s" % (pid, e), flush=True)
            continue
        time.sleep(4)
        dep, lb = -1, 0
        if total <= GIANT_TOTAL:
            try:
                dep = int(rows2d(sparql(Q_DEP % pid))[0]["n"])
            except Exception as e:
                print("    [WARN] %s dep: %s" % (pid, e), flush=True)
            time.sleep(4)
        # reason 分布恒查（快路径，兼作大属性下界）
        try:
            rr = rows2d(sparql(Q_REASONS % pid))
            s = 0
            for x in rr:
                reason_rows.append({"pid": pid, "reason": x["reason"], "n": int(x["n"])})
                s += int(x["n"])
            if dep < 0 and s > 0:
                dep, lb = s, 1
        except Exception as e:
            print("    [WARN] %s reasons: %s" % (pid, e), flush=True)
        rate = dep / total if dep >= 0 and total > 0 else -1.0
        prop_rows.append({"pid": pid, "gloss": gloss, "total_statements": total,
                          "deprecated": dep, "dep_is_lower_bound": lb,
                          "retraction_rate": rate})
        rate_s = ("%.4f%%" % (rate * 100)) if rate >= 0 else "n/a"
        print("[OK] %s %s: total=%d dep=%d%s rate=%s (%.1fs)"
              % (pid, gloss, total, dep, "(LB)" if lb else "", rate_s, time.time() - t0), flush=True)

        if pid in HEAD6:
            time.sleep(4)
            try:
                sr = rows2d(sparql(Q_SAMPLE % (pid, pid, 500), timeout=100))
                anchored = 0
            except Exception:
                sr = rows2d(sparql(Q_SAMPLE_ANCH % (pid, pid, 500), timeout=100))
                anchored = 1
            for x in sr:
                x["pid"] = pid
                x["gloss"] = gloss
                x["anchored"] = anchored
            sample_rows.extend(sr)
            n_p582 = sum(1 for x in sr if x.get("endtime"))
            n_p813 = sum(1 for x in sr if x.get("retrieved"))
            extra_rows.append({"key": "sample_%s_n" % pid, "value": len(sr)})
            extra_rows.append({"key": "sample_%s_anchored" % pid, "value": anchored})
            extra_rows.append({"key": "sample_%s_p582_cov" % pid, "value": round(n_p582 / len(sr), 4) if sr else ""})
            extra_rows.append({"key": "sample_%s_p813_cov" % pid, "value": round(n_p813 / len(sr), 4) if sr else ""})
            print("     sample %s: %d rows anchored=%d P582cov=%.2f P813cov=%.2f"
                  % (pid, len(sr), anchored,
                     (n_p582 / len(sr)) if sr else -1, (n_p813 / len(sr)) if sr else -1), flush=True)
            time.sleep(4)
            try:
                yr = rows2d(sparql(Q_P813_YEAR % pid))
                for x in yr:
                    p813_rows.append({"pid": pid, "year": x.get("yr", ""), "n": int(x["n"])})
            except Exception as e:
                print("    [WARN] %s p813year: %s" % (pid, e), flush=True)
            time.sleep(4)
            try:
                c582 = int(rows2d(sparql(Q_P582_CNT % pid))[0]["n"])
                extra_rows.append({"key": "p582_with_reason_%s" % pid, "value": c582})
            except Exception as e:
                print("    [WARN] %s p582cnt: %s" % (pid, e), flush=True)
        time.sleep(3)
        _dump_all()

    # 原因 QID 标签
    qids = sorted({r["reason"].rsplit("/", 1)[-1] for r in reason_rows})
    labels = {}
    for i in range(0, len(qids), 40):
        vals = " ".join("wd:" + q for q in qids[i:i + 40])
        q = ("SELECT ?x ?en ?zh WHERE { VALUES ?x { " + vals + " } "
             "OPTIONAL { ?x rdfs:label ?en FILTER(lang(?en)='en') } "
             "OPTIONAL { ?x rdfs:label ?zh FILTER(lang(?zh)='zh') } }")
        try:
            for x in rows2d(sparql(q, timeout=60)):
                labels[x["x"]] = (x.get("en", ""), x.get("zh", ""))
        except Exception as e:
            print("[WARN] labels chunk %d: %s" % (i, e), flush=True)
        time.sleep(2)
    with open(BASE / "E8_原因标签.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)
    _dump_all()
    print("E8 profiling DONE (%d props, %d reasons, %d samples)"
          % (len(prop_rows), len(reason_rows), len(sample_rows)), flush=True)


if __name__ == "__main__":
    main()
