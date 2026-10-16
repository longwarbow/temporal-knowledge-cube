# -*- coding: utf-8 -*-
"""E9：MediaWiki 修订史 API 抽「裁定时刻」（把 E8 的时刻代理升级为真实时刻）。

思路（E8 遗留问题）：Wikidata 的 `rank=deprecated` 在 RDF 里**没有时间戳**，
但**每次 rank 变更都对应一次页面修订** ⇒ 用修订史二分搜索定位
「目标声明首次变为 deprecated」的那一版修订，其 revision timestamp 即**裁定时刻**。

精度说明（必须如实标注）：MediaWiki `timestamp` 为**秒级** ISO8601，非毫秒级。

流程（每条声明）：
  1) 取页面修订列表（revid + timestamp + comment，oldest→newest，分页）；
  2) 在最新版内容里定位目标声明的 **statement id**（`Qxx$UUID`，rank 变更不改 id）；
  3) 二分搜索 `f(rev) = 该 id 存在且 rank==deprecated` 的最早版本 → flip_rev / flip_ts；
  4) 二分搜索 `g(rev) = 该 id 存在` 的最早版本 → birth_rev / birth_ts（→ 声明存活时长）；
  5) **边界核验**：确认 f(flip_rev)=True 且 f(flip_rev 前一版)=False（方法论② 自查）。

通道：`--noproxy '*'` 直连 + curl 子进程（见 E8 报告 §2.1）。
"""
import csv, json, os, subprocess, sys, threading, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent
API = "https://www.wikidata.org/w/api.php"
UA = "Scene1-TemporalKnowledgeCube-E9/0.1 (research; contact: longwarbow@126.com)"

E8_SAMPLE = BASE.parent / "E8_Wikidata声明级撤回" / "E8_声明级样本.csv"
MAX_ITEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
REV_CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 2000   # 修订数超此值则跳过（控制成本）
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8

# ★ 全局节流：改 GET 之后 429 显著减少（GET 实测 ~10/12 成功，POST 0/12），
#   但**持续**高负载仍会触发限流 ⇒ 用「全局最小间隔 + 全局冷却」双保险。
#   默认 PACE=1.0s；并发只用于重叠网络延迟，不用来提高速率。
_PACE_LOCK = threading.Lock()
_LAST_CALL = [0.0]
_BLOCK_UNTIL = [0.0]          # ★ 全局冷却截止时刻：一次 429 让**所有**线程一起停
PACE = float(os.environ.get("E9_PACE", "1.0"))


def _pace():
    with _PACE_LOCK:
        now = time.time()
        wait = max(_BLOCK_UNTIL[0] - now, PACE - (now - _LAST_CALL[0]))
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL[0] = time.time()


def _cooldown(seconds):
    """全局冷却：把「撞 429 后各线程各自重试」变成「全体一起等」，
    否则重试本身会持续消耗限流额度，形成自我加剧的风暴。"""
    with _PACE_LOCK:
        _BLOCK_UNTIL[0] = max(_BLOCK_UNTIL[0], time.time() + seconds)


def api(params, timeout=30, retries=8):
    """MediaWiki API **GET**（curl 直连 + 全局节流 + 快速重试）。

    ★★ 最关键的一条实测结论：**必须用 GET，绝不能用 POST。**
      用 `curl --data-urlencode` 会隐式变成 POST，而 Wikimedia 对 POST 的限流极严 ——
      实测 12 次 POST 得到 **12 次 429**，同样 12 次 GET 得到 **10 次 200**。
      早期版本"慢"的根因就是这个：每次请求都 429，再叠加长退避 ⇒ 雪崩。
      ⇒ 参数一律拼进 URL query string。

    另两条次要但有用的防护：
      - 约 8% 的请求会 `HTTP 000`（连接中断）；`--connect-timeout 8 -m 30` 把失败成本压到 ~10s。
      - `--speed-limit/--speed-time` 用于掐断已停住的传输。
    """
    last = None
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        _pace()
        cmd = ["curl", "-s", "--noproxy", "*", "--connect-timeout", "8",
               "-m", str(timeout), "--speed-limit", "500", "--speed-time", "20",
               "-A", UA, "-w", "\n%{http_code}", url]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout + 10)
            out = proc.stdout.decode("utf-8", "replace")
            body, _, code = out.rpartition("\n")
            if code.strip() == "200" and body.strip():
                return json.loads(body)
            last = "HTTP=%s rc=%s %r" % (code.strip() or "-", proc.returncode, body[:80])
            STAT.retries += 1
            STAT.last_err = last[:140]
            if os.environ.get("E9_VERBOSE"):
                sys.stderr.write("   [api-retry a%d] %s\n" % (attempt, last[:120]))
                sys.stderr.flush()
            if code.strip() == "429":
                _cooldown(15)                            # 全体冷却，避免重试风暴
                continue
        except Exception as e:                                  # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, e)
            STAT.retries += 1
            STAT.last_err = last[:140]
        time.sleep(min(1 + attempt, 4))             # 网络抖动：快速重试
    raise RuntimeError("api fail: %s" % last)


class Stat:
    def __init__(self):
        self.calls = 0
        self.retries = 0
        self.last_err = ""


STAT = Stat()


def rev_list(qid):
    """返回 [(revid, ts, comment)]，oldest→newest；超 REV_CAP 返回 None。"""
    out, cont = [], None
    while True:
        p = {"action": "query", "prop": "revisions", "titles": qid, "rvdir": "newer",
             "rvlimit": "500", "rvprop": "ids|timestamp|comment", "format": "json",
             "formatversion": "2"}
        if cont:
            p["rvcontinue"] = cont
        d = api(p)
        STAT.calls += 1
        pages = d.get("query", {}).get("pages", [])
        if not pages or "missing" in pages[0]:
            return None
        for r in pages[0].get("revisions", []):
            out.append((r["revid"], r.get("timestamp", ""), r.get("comment", "")))
        cont = d.get("continue", {}).get("rvcontinue")
        if not cont or len(out) > REV_CAP:
            break
    return out if len(out) <= REV_CAP else None


def rev_content(revid):
    """取某修订的 entity JSON（含 claims 与 rank）。"""
    d = api({"action": "query", "prop": "revisions", "revids": str(revid),
             "rvprop": "content|timestamp", "rvslots": "main",
             "format": "json", "formatversion": "2"})
    STAT.calls += 1
    pages = d.get("query", {}).get("pages", [])
    if not pages or "revisions" not in pages[0]:
        return None, ""
    rev = pages[0]["revisions"][0]
    try:
        content = rev["slots"]["main"]["content"]
    except Exception:                                            # noqa: BLE001
        return None, rev.get("timestamp", "")
    try:
        return json.loads(content), rev.get("timestamp", "")
    except Exception:                                            # noqa: BLE001
        return None, rev.get("timestamp", "")


def norm_value(pid, raw):
    """把 E8 样本里的 value 字段归一成与 entity JSON 可比较的形式。"""
    if raw.startswith("http") and "/entity/" in raw:
        return ("entityid", raw.rsplit("/", 1)[-1])
    t = raw.replace("+", "")
    return ("time", t)


def _parse_time(t):
    """'+1976-04-05T00:00:00Z' / '-2200-00-00T00:00:00Z' → (sign, y, m|None, d|None)。

    ★ 精度坑：WDQS 把「精度只到年」的日期补成 01-01（如 -2200-01-01），
      而 entity JSON 保留 00-00（-2200-00-00）⇒ 必须做**精度感知**比较，
      把 0 视为「未知/通配」。
    """
    t = t.replace("+", "")
    neg = t.startswith("-")
    core = t[1:] if neg else t
    try:
        y, m, d = core.split("T")[0].split("-")
    except Exception:                                            # noqa: BLE001
        return None
    try:
        yi = int(y)
        mi = int(m) or None
        di = int(d) or None
    except Exception:                                            # noqa: BLE001
        return None
    return (-yi if neg else yi, mi, di)


def _time_match(a, b):
    """精度感知时间比较：任一侧为 None(未知) 即视为该分量匹配。"""
    if not a or not b:
        return False
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        if x != y:
            return False
    return True


def _stmt_value(st):
    dv = (st.get("mainsnak") or {}).get("datavalue")
    if not dv:
        return None
    v = dv.get("value")
    if isinstance(v, dict) and "id" in v:
        return ("entityid", v["id"])
    if isinstance(v, dict) and "time" in v:
        return ("time", _parse_time(v["time"]))
    return None


def find_stmt_id(entity, pid, value):
    """返回 (statement_id, matched_by)。matched_by ∈ exact / fallback_unique / none。"""
    kind, val = norm_value(pid, value)
    target = _parse_time(val) if kind == "time" else val
    dep_sts = [st for st in (entity.get("claims", {}) or {}).get(pid, [])
               if st.get("rank") == "deprecated"]
    for st in dep_sts:
        got = _stmt_value(st)
        if got is None:
            continue
        if kind == "time" and got[0] == "time" and _time_match(got[1], target):
            return st.get("id"), "exact"
        if kind == "entityid" and got == ("entityid", target):
            return st.get("id"), "exact"
    if len(dep_sts) == 1 and dep_sts[0].get("id"):
        return dep_sts[0]["id"], "fallback_unique"      # 该属性仅有一条降级声明
    return None, "none"


def stmt_state(entity, sid):
    """返回 statement 状态: ('absent'|'normal'|'preferred'|'deprecated')。"""
    if not entity:
        return "absent"
    for pid, sts in (entity.get("claims", {}) or {}).items():
        for st in sts:
            if st.get("id") == sid:
                return st.get("rank", "normal")
    return "absent"


def bisect_first(revs, pred, cache):
    """在 revs(oldest→newest) 上二分找 pred=True 的最早下标；无则 None。"""
    lo, hi, ans = 0, len(revs) - 1, None
    if not pred(revs[hi][0]):
        return None
    while lo <= hi:
        mid = (lo + hi) // 2
        if pred(revs[mid][0]):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans


def rev_contents_chunk(qid, revs, chunk=150):
    """一次请求取 chunk 个修订的**内容**（rvlimit 带 content），返回 {revid: entity_json}。

    ★ 这是本实验的关键提速：逐版二分搜索每条要 ~30 次请求（会被 API 限流），
      而分块取全史只需 ceil(n/chunk) 次（100 版的条目 = 1 次）。
    """
    out, i = {}, 0
    while i < len(revs):
        batch = revs[i:i + chunk]
        d = api({"action": "query", "prop": "revisions", "titles": qid,
                 "rvstartid": str(batch[0][0]), "rvdir": "newer",
                 "rvlimit": str(len(batch)),
                 "rvprop": "content|ids|timestamp", "rvslots": "main",
                 "format": "json", "formatversion": "2"}, timeout=60)
        STAT.calls += 1
        pages = d.get("query", {}).get("pages", [])
        got = (pages[0].get("revisions", []) if pages else []) or []
        for r in got:
            try:
                out[r["revid"]] = json.loads(r["slots"]["main"]["content"])
            except Exception:                                    # noqa: BLE001
                out[r["revid"]] = None
        if not got:                       # 内容不可得（如修订被隐藏）→ 缩批跳过
            chunk = max(1, chunk // 2)
            if chunk == 1:
                i += 1
            continue
        last = got[-1]["revid"]
        idx = next((k for k, (rv, _, _) in enumerate(revs) if rv == last), None)
        i = (idx + 1) if idx is not None else i + len(batch)
        if len(got) < len(batch):         # 服务端截断 → 自适应缩小批量
            chunk = max(1, len(got))
    return out


def process_chunked(row, qid, pid, revs, res):
    """分块全史线性扫描：定位 flip（首次 deprecated）与 birth（首次存在）。"""
    cont = rev_contents_chunk(qid, revs)
    # HEAD 内容缺失时回退到最近的可用版本
    ent_head = None
    for revid, _, _ in reversed(revs):
        if cont.get(revid):
            ent_head = cont[revid]
            break
    if ent_head is None:
        res["status"] = "content_unavailable"
        return res
    sid, matched_by = find_stmt_id(ent_head, pid, row["value"])
    res["stmt_id"] = sid or ""
    res["matched_by"] = matched_by
    if not sid:
        res["status"] = "stmt_not_found"
        return res
    flip_i = birth_i = None
    prev_states = []
    for idx, (revid, ts, comment) in enumerate(revs):
        if revid not in cont:            # 内容不可得的修订：跳过（不当作 absent）
            prev_states.append("unknown")
            continue
        st_state = stmt_state(cont.get(revid), sid)
        prev_states.append(st_state)
        if st_state != "absent" and birth_i is None:
            birth_i = idx
        if st_state == "deprecated" and flip_i is None:
            flip_i = idx
            break                                   # 首次 deprecated 即裁定时刻
    if flip_i is None:
        res["status"] = "never_deprecated_in_history"
        return res
    res["flip_rev"] = revs[flip_i][0]
    res["flip_ts"] = revs[flip_i][1]
    res["flip_comment"] = (revs[flip_i][2] or "")[:120]
    res["flip_prev_state"] = prev_states[flip_i - 1] if flip_i > 0 else "(first revision)"
    ok_left = flip_i == 0 or prev_states[flip_i - 1] != "deprecated"
    if birth_i is not None:
        res["birth_rev"] = revs[birth_i][0]
        res["birth_ts"] = revs[birth_i][1]
        try:
            from datetime import datetime
            t0 = datetime.fromisoformat(res["birth_ts"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(res["flip_ts"].replace("Z", "+00:00"))
            res["lifespan_days"] = round((t1 - t0).total_seconds() / 86400, 3)
        except Exception:                                        # noqa: BLE001
            pass
    res["status"] = "ok" if ok_left else "ok_boundary_unchecked"
    return res


def process(row):
    qid = row["item"].rsplit("/", 1)[-1]
    pid = row["pid"]
    print("    [START] %s %s" % (qid, pid), flush=True)
    res = {"item": qid, "pid": pid, "value": row["value"], "gloss": row.get("gloss", ""),
           "stmt_id": "", "matched_by": "",
           "n_revs": "", "birth_ts": "", "birth_rev": "", "flip_ts": "", "flip_rev": "",
           "flip_comment": "", "lifespan_days": "", "n_api_calls": "", "elapsed_s": "",
           "n_retries": "", "last_err": "", "status": "",
           "e8_endtime": row.get("endtime", ""), "e8_retrieved": row.get("retrieved", ""),
           "e8_reason": row.get("reason", "").rsplit("/", 1)[-1]}
    before = STAT.calls
    _retries_before = STAT.retries
    _t_start = time.time()
    try:
        revs = rev_list(qid)
        if revs is None:
            res["status"] = "too_many_revisions"
            return res
        if not revs:
            res["status"] = "no_revisions"
            return res
        res["n_revs"] = len(revs)
        # ★ 策略选择（实测教训）：分块取「全史内容」在**小页面**上只要 1 次请求；
        #   但在**大页面**上服务端按响应体大小截断 ⇒ 批量自适应缩小 ⇒ 累计下载数百 MB。
        #   故阈值定得很低（E9_LINEAR_MAX 默认 12）：小页面线性全扫（精确、1 次请求），
        #   大页面二分定位（~log2(n) 次单版请求，每次只取一版内容）。
        if len(revs) <= int(os.environ.get("E9_LINEAR_MAX", "12")):
            return process_chunked(row, qid, pid, revs, res)
        # 定位 statement id（用最新版内容）
        ent, _ = rev_content(revs[-1][0])
        cache = {revs[-1][0]: ent}          # 复用 HEAD 内容，避免重复请求
        sid, matched_by = find_stmt_id(ent, pid, row["value"])
        res["stmt_id"] = sid or ""
        res["matched_by"] = matched_by
        if not sid:
            res["status"] = "stmt_not_found"
            return res
        def get(revid):
            if revid not in cache:
                cache[revid] = rev_content(revid)[0]
            return cache[revid]

        def is_dep(revid):
            return stmt_state(get(revid), sid) == "deprecated"

        def exists(revid):
            return stmt_state(get(revid), sid) != "absent"

        fi = bisect_first(revs, is_dep, cache)
        if fi is None:
            res["status"] = "never_deprecated_in_history"
            return res
        res["flip_rev"] = revs[fi][0]
        res["flip_ts"] = revs[fi][1]
        res["flip_comment"] = (revs[fi][2] or "")[:120]
        # 边界核验
        ok_left = (fi == 0) or (not is_dep(revs[fi - 1][0]))
        bi = bisect_first(revs, exists, cache)
        if bi is not None:
            res["birth_rev"] = revs[bi][0]
            res["birth_ts"] = revs[bi][1]
            try:
                from datetime import datetime
                t0 = datetime.fromisoformat(res["birth_ts"].replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(res["flip_ts"].replace("Z", "+00:00"))
                res["lifespan_days"] = round((t1 - t0).total_seconds() / 86400, 3)
            except Exception:                                    # noqa: BLE001
                pass
        res["status"] = "ok" if ok_left else "ok_boundary_unchecked"
        if fi > 0:
            res["flip_prev_state"] = stmt_state(get(revs[fi - 1][0]), sid)
        return res
    except Exception as e:                                       # noqa: BLE001
        # ★ 部分结果也要保留：若已定位 flip 而在后续（取 birth / 核验）失败，
        #   丢弃 flip 等于白跑几十次请求。标记为 ok_partial_flip_only（仍以 ok 开头，
        #   续跑时会被视为已完成；分析脚本单独统计这类行的 birth 缺失）。
        if res.get("flip_ts"):
            res["status"] = "ok_partial_flip_only"
        else:
            res["status"] = "error: %s" % e
        return res
    finally:
        res["n_api_calls"] = STAT.calls - before
        res["elapsed_s"] = round(time.time() - _t_start, 1)
        res["n_retries"] = STAT.retries - _retries_before
        res["last_err"] = STAT.last_err if res["n_retries"] else ""


def main():
    rows = list(csv.DictReader(open(E8_SAMPLE, encoding="utf-8-sig")))
    # 每 item 取 1 条声明；按属性轮转取样，避免结果被单一属性主导
    buckets = {}
    for r in rows:
        buckets.setdefault(r["pid"], []).append(r)
    picked, seen_item = [], set()
    while len(picked) < MAX_ITEMS and any(buckets.values()):
        progressed = False
        for pid in list(buckets.keys()):
            b = buckets[pid]
            while b:
                r = b.pop(0)
                q = r["item"].rsplit("/", 1)[-1]
                if q in seen_item:
                    continue
                seen_item.add(q)
                picked.append(r)
                progressed = True
                break
            if len(picked) >= MAX_ITEMS:
                break
        if not progressed:
            break
    print("[INFO] 候选 %d 条声明（来自 %d 条 E8 样本，属性分布 %s）"
          % (len(picked), len(rows), sorted({r["pid"] for r in picked})), flush=True)

    out, done = [], 0
    out_path = BASE / "E9_裁定时刻.csv"
    cols = ["item", "pid", "gloss", "value", "stmt_id", "matched_by", "n_revs", "birth_ts", "birth_rev",
            "flip_ts", "flip_rev", "flip_comment", "flip_prev_state", "lifespan_days",
            "e8_endtime", "e8_retrieved", "e8_reason", "n_api_calls", "elapsed_s",
            "n_retries", "last_err", "status"]

    # ★ 断点续跑：保留已有 CSV 里 status=ok* 的行（含 ok_partial_flip_only），
    #   以及任何**已拿到 flip_ts** 的行（历史版本丢失 flip 的浪费不再重演）。
    if out_path.exists() and os.environ.get("E9_FRESH") != "1":
        keep = {}
        for r in csv.DictReader(open(out_path, encoding="utf-8-sig")):
            if r.get("item") and ((r.get("status") or "").startswith("ok") or r.get("flip_ts")):
                keep[r["item"]] = r
        if keep:
            out = list(keep.values())
            picked = [r for r in picked if r["item"].rsplit("/", 1)[-1] not in keep]
            print("[RESUME] 已完成 %d 条，本次重跑 %d 条" % (len(keep), len(picked)), flush=True)

    total = len(picked) + len(out)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process, r): r for r in picked}
        for fut in as_completed(futs):
            res = fut.result()
            out.append(res)
            done += 1
            print("[%3d/%3d] %s %s revs=%s flip=%s status=%s calls=%s retries=%s %ss"
                  % (done, len(picked), res["item"], res["pid"], res["n_revs"],
                     res["flip_ts"] or "-", res["status"], res["n_api_calls"],
                     res["n_retries"], res["elapsed_s"]), flush=True)
            if done % 5 == 0:
                with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(out)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    ok = sum(1 for r in out if r["status"].startswith("ok"))
    print("[DONE] total=%d ok=%d api_calls=%d -> %s" % (len(out), ok, STAT.calls, out_path.name), flush=True)


if __name__ == "__main__":
    main()
