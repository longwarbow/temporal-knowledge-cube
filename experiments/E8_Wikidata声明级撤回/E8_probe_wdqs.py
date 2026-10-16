# -*- coding: utf-8 -*-
"""E8 步骤0：探测 Wikidata Query Service (WDQS) 通道。

方法论⑩：每次先测通道再定路径。本脚本只做三件事：
  1) 连通性 + 延迟（1 行 trivial 查询）
  2) deprecated rank 是否可查（P569 birth date 单属性小查询）
  3) 单属性 count 查询的耗时基线（决定全量画像的批策略）
"""
import sys, time, json, subprocess, urllib.parse

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "Scene1-TemporalKnowledgeCube-E8/0.1 (research; contact: longwarbow@126.com)"

# 本机 Python TLS（OpenSSL）被拦而 curl（schannel）可通 → 统一走 curl 子进程
CURL = ["curl", "-s", "-m", "{timeout}", "-A", UA,
        "-H", "Accept: application/sparql-results+json",
        "--data-urlencode", "query={query}",
        "--data-urlencode", "format=json",
        ENDPOINT]


def sparql(query: str, timeout: int = 90) -> tuple:
    """返回 (rows, elapsed_seconds)。rows 为 JSON bindings 数组。"""
    cmd = [c.format(timeout=timeout, query=query) if "{" in c and "}" in c and c not in (ENDPOINT,) else c
           for c in CURL]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout + 10)
    if proc.returncode != 0:
        raise RuntimeError(f"curl exit {proc.returncode}: {proc.stderr[:200]!r}")
    payload = json.loads(proc.stdout.decode("utf-8"))
    return payload["results"]["bindings"], time.time() - t0


def run(name: str, query: str, timeout: int = 90) -> list:
    try:
        rows, dt = sparql(query, timeout)
        print(f"[OK] {name}: {len(rows)} rows, {dt:.2f}s")
        for r in rows[:5]:
            print("     ", json.dumps(r, ensure_ascii=False)[:220])
        return rows
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return []


if __name__ == "__main__":
    # 1) 连通性
    run("trivial", "SELECT (1 AS ?x) WHERE {}")
    # 2) deprecated rank 可查性：P569（出生日期，历史上大量因纠正而降级）
    run("deprecated_count_P569", """
SELECT (COUNT(*) AS ?n) WHERE {
  ?item p:P569 ?st . ?st wikibase:rank wikibase:DeprecatedRank .
}""")
    # 3) P2241 原因分布（P569 上带原因的子集）
    run("reason_dist_P569", """
SELECT ?reason (COUNT(*) AS ?n) WHERE {
  ?item p:P569 ?st .
  ?st wikibase:rank wikibase:DeprecatedRank ; pq:P2241 ?reason .
} GROUP BY ?reason ORDER BY DESC(?n)""")
