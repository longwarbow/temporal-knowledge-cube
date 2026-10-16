# -*- coding: utf-8 -*-
"""E9 机制核验：裁定那一刻到底做了什么？

论文 §7.6 finding 1 断言「Wikidata 的 deprecated 多为 REVISE（同一编辑里
引入更正后的新值 + 把旧值降级），而非独立 RETRACT」。原先只有 5 条 pilot 证据。

本脚本对每条已定位 flip 的声明，取 **flip 修订** 与其 **父修订** 的实体内容，
比较目标属性 P 的**全部声明集合**，分类：

  REVISE_NEW_VALUE    flip 版新增了一条「值不同」的声明（rank != deprecated）
  RANK_ONLY           声明集合完全不变，只有目标声明 rank → deprecated
  REMOVE_CLAIM        flip 版该属性声明数减少（删掉了别的声明）
  CREATE_DEPRECATED   flip 版就是该声明的创建版（出生即降级）
  UNCLEAR             其它

通道同 E9_extract（curl --noproxy '*' + 全局节流）。
"""
import csv, json, os, subprocess, sys, threading, time, urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
API = "https://www.wikidata.org/w/api.php"
UA = "Scene1-TemporalKnowledgeCube-E9/0.1 (research; contact: longwarbow@126.com)"
SRC = BASE / "E9_裁定时刻.csv"
OUT = BASE / "E9_机制核验.csv"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 100
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

_PACE_LOCK = threading.Lock()
_LAST_CALL = [0.0]
PACE = float(os.environ.get("E9_PACE", "0.3"))


def _pace():
    with _PACE_LOCK:
        dt = time.time() - _LAST_CALL[0]
        if dt < PACE:
            time.sleep(PACE - dt)
        _LAST_CALL[0] = time.time()


def api(params, timeout=30, retries=6):
    """MediaWiki API —— **必须 GET，不能 POST**（POST 实测 12/12 被 429，GET 10/12 通过）。"""
    last = None
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        _pace()
        cmd = ["curl", "-s", "--noproxy", "*", "--connect-timeout", "8",
               "-m", str(timeout), "--speed-limit", "500", "--speed-time", "20",
               "-A", UA, "-w", "\n%{http_code}", url]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout + 10)
            body, _, code = proc.stdout.decode("utf-8", "replace").rpartition("\n")
            if code.strip() == "200" and body.strip():
                return json.loads(body)
            last = "HTTP=%s %r" % (code.strip() or proc.returncode, body[:100])
            if code.strip() == "429":
                time.sleep(min(5 * (attempt + 1), 20))
                continue
        except Exception as e:                                   # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, e)
        time.sleep(min(1 + attempt, 4))
    raise RuntimeError("api fail: %s" % last)


def value_key(st):
    dv = (st.get("mainsnak") or {}).get("datavalue") or {}
    v = dv.get("value")
    if isinstance(v, dict) and "id" in v:
        return "E:" + v["id"]
    if isinstance(v, dict) and "time" in v:
        return "T:" + v["time"].lstrip("+")
    if isinstance(v, dict) and "amount" in v:
        return "N:" + str(v["amount"])
    if v is None:
        return "?"
    return "S:" + str(v)[:40]


def claims_of(ent, pid):
    out = []
    for st in (ent or {}).get("claims", {}).get(pid, []) or []:
        out.append((st.get("id"), value_key(st), st.get("rank", "normal")))
    return out


def _times_from_snaks(snaks):
    """从 snak 列表里取 P813(retrieved) / P582(end time) 的 time 值。"""
    got = {}
    for q in snaks or []:
        if q.get("snaktype") != "value":
            continue
        p = q.get("property")
        dv = (q.get("datavalue") or {})
        v = dv.get("value")
        if p in ("P813", "P582") and isinstance(v, dict) and "time" in v:
            got.setdefault(p, []).append(v["time"].lstrip("+"))
    return got


def quals_of(ent, pid, sid):
    """取目标声明的 P813/P582 —— 限定符 + 来源（references）两处。

    ★ 这一步把「P813 是否为裁定时刻上界」的可比对样本从 E8 样本文件里的 5 条
      扩展到**全部**条目：因为 flip 版实体 JSON 本来就已经取到手了，零额外请求。
    """
    for st in (ent or {}).get("claims", {}).get(pid, []) or []:
        if st.get("id") != sid:
            continue
        out = {"q_P813": "", "q_P582": "", "r_P813": ""}
        q = _times_from_snaks(st.get("qualifiers", {}).get("P813"))
        if q.get("P813"):
            out["q_P813"] = min(q["P813"])          # 限定符里的 retrieved（取最早）
        qq = _times_from_snaks(st.get("qualifiers", {}).get("P582"))
        if qq.get("P582"):
            out["q_P582"] = min(qq["P582"])         # 计划性失效时刻
        refs = []
        for ref in st.get("references", []) or []:
            r = _times_from_snaks((ref.get("snaks") or {}).get("P813"))
            refs += r.get("P813", [])
        if refs:
            out["r_P813"] = min(refs)               # 来源核查时刻（取最早，作上界候选）
        return out
    return {"q_P813": "", "q_P582": "", "r_P813": ""}


def run(row):
    qid, pid, sid, flip_rev = row["item"], row["pid"], row["stmt_id"], row["flip_rev"]
    res = {"item": qid, "pid": pid, "flip_rev": flip_rev, "kind": "", "detail": "",
           "n_claims_before": "", "n_claims_after": "", "n_active_after": "",
           "active_values_after": "", "n_new_values": "",
           "prev_rank_of_sid": "", "q_P813": "", "q_P582": "", "r_P813": "",
           "flip_ts": row.get("flip_ts", ""), "e8_retrieved": row.get("e8_retrieved", ""),
           "e8_endtime": row.get("e8_endtime", ""),
           "flip_comment": row.get("flip_comment", "")}
    try:
        d = api({"action": "query", "prop": "revisions", "titles": qid,
                 "rvstartid": str(flip_rev), "rvdir": "older", "rvlimit": "2",
                 "rvprop": "ids|content|timestamp", "rvslots": "main",
                 "format": "json", "formatversion": "2"})
        pages = d.get("query", {}).get("pages", [])
        revs = (pages[0].get("revisions", []) if pages else []) or []
        if len(revs) < 2:
            res["kind"] = "NO_PARENT"
            return res
        ent_flip = json.loads(revs[0]["slots"]["main"]["content"])
        ent_prev = json.loads(revs[1]["slots"]["main"]["content"])
    except Exception as e:                                       # noqa: BLE001
        res["kind"] = "ERROR: %s" % str(e)[:60]
        return res

    res.update(quals_of(ent_flip, pid, sid))
    cb = claims_of(ent_prev, pid)
    ca = claims_of(ent_flip, pid)
    res["n_claims_before"] = len(cb)
    res["n_claims_after"] = len(ca)
    # ★ 该属性在 flip 版里**仍有多少条 active（rank != deprecated）声明**：
    #   >0 ⇒ 系统保留了替代值（旧值就地撤回，替代值已在别处）；
    #   =0 ⇒ 撤回后该属性暂时没有可用值（纯撤回，等待后续补值）。
    act_after = [c for c in ca if c[2] != "deprecated"]
    res["n_active_after"] = len(act_after)
    res["active_values_after"] = "|".join(c[1] for c in act_after[:4])
    ids_b = {c[0] for c in cb}
    # 目标声明在父版里的 rank
    res["prev_rank_of_sid"] = next((c[2] for c in cb if c[0] == sid), "(absent)")
    # flip 版里「新出现且非 deprecated」的声明数（= 更正后的新值）
    new_ok = [c for c in ca if c[0] not in ids_b and c[2] != "deprecated"]
    res["n_new_values"] = len(new_ok)
    # 目标声明在 flip 版里的 rank（应为 deprecated）
    after_rank = next((c[2] for c in ca if c[0] == sid), "(absent)")
    res["detail"] = "sid_after=%s; new_values=%s" % (
        after_rank, "|".join("%s(%s)" % (c[1], c[2]) for c in new_ok[:3]))

    if res["prev_rank_of_sid"] == "(absent)":
        res["kind"] = "CREATE_DEPRECATED"
    elif len(new_ok) >= 1:
        res["kind"] = "REVISE_NEW_VALUE"
    elif len(ca) < len(cb):
        res["kind"] = "REMOVE_CLAIM"
    elif len(ca) == len(cb):
        res["kind"] = "RANK_ONLY"
    else:
        res["kind"] = "UNCLEAR"
    return res


def main():
    rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8-sig"))
            if (r.get("status") or "").startswith("ok") and r.get("flip_rev")]
    rows = rows[:LIMIT]
    print("[INFO] 核验 %d 条 flip 机制" % len(rows), flush=True)
    cols = ["item", "pid", "flip_rev", "kind", "prev_rank_of_sid",
            "n_claims_before", "n_claims_after", "n_active_after", "active_values_after",
            "n_new_values", "detail",
            "flip_ts", "q_P813", "r_P813", "q_P582", "e8_retrieved", "e8_endtime", "flip_comment"]
    out = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run, r): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            out.append(res)
            print("[%3d/%3d] %s %s %s ret=%s (%s)" % (i, len(rows), res["item"], res["pid"],
                                                      res["kind"], (res["q_P813"] or res["r_P813"] or "-"),
                                                      res["detail"][:36]), flush=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print("\n== 机制分布 ==", flush=True)
    for k, v in Counter(r["kind"] for r in out).most_common():
        print("   %-20s %3d (%.1f%%)" % (k, v, 100 * v / max(1, len(out))))
    _valid = [r for r in out if not str(r["kind"]).startswith("ERROR")]
    _va = Counter(str(r["n_active_after"]) for r in _valid)
    print("  flip 版里该属性仍存在的 active 声明数分布：%s"
          % dict(sorted(_va.items(), key=lambda x: (x[0] != "0", x[0]))), flush=True)
    _zero = sum(1 for r in _valid if str(r["n_active_after"]) == "0")
    print("  ⇒ 撤回后该属性**无**可用值（纯撤回）：%d/%d (%.1f%%)"
          % (_zero, len(_valid), 100 * _zero / max(1, len(_valid))), flush=True)

    # ★ P813(retrieved) 是否为裁定时刻的上界（样本 = 全部条目，而非 E8 样本文件里的 5 条）
    def pd(s):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:                                        # noqa: BLE001
            return None
    cmp_ = []
    for r in out:
        f_, ret = pd(r.get("flip_ts", "")), pd(r.get("q_P813") or r.get("r_P813") or "")
        if f_ and ret:
            cmp_.append((r, f_, ret))
    print("\n== P813(retrieved) vs 裁定时刻 flip ==", flush=True)
    if cmp_:
        lt = [x for x in cmp_ if x[2] < x[1]]
        print("  可比对 %d 条：retrieved < flip %d (%.1f%%) ⇒ P813 %s上界"
              % (len(cmp_), len(lt), 100 * len(lt) / len(cmp_), "**不是**" if lt else "是"))
        for r, f_, ret in sorted(lt, key=lambda x: (x[2] - x[1]).days)[:5]:
            print("     ★反例 %-12s %-5s flip=%s retrieved=%s（差 %+d 天）"
                  % (r["item"], r["pid"], f_.date(), ret.date(), (f_ - ret).days))
    else:
        print("  无可比对条目")
    print("[SAVE] %s" % OUT.name, flush=True)


if __name__ == "__main__":
    main()
