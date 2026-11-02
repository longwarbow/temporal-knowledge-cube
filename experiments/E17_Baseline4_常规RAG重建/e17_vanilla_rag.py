# -*- coding: utf-8 -*-
"""
E17 —— Baseline #4：常规实践基线「RAG + 定期全量重建」

## 为什么做（P0-11 Baseline 补强）
E14（EraRAG）是"最接近的已发表系统"，但它是被修过的开源实现，审稿人真正会问的是
更朴素的一问：**普通工程团队今天会怎么做？** 答案：分块 + 向量索引 + 夜间全量重建。
本实验把这条路径按**最有利于它**的方式实现（不设陷阱），测三件事：

  Q1 成本：每万文档的 token 与墙钟（建索引 / 全量重建 / 增量），对照我们的增量路径。
  Q2 时效：一条声明被撤回后，直到下一个重建周期才生效 —— 窗口内被答成
     "仍然成立"的问答比例，跨 10% / 1% / 0.1% 三档重建预算。
  Q3 引用可溯：撤回后"答案仍引用已撤回文档"只可能发生在无引用轨迹的设置里；
     在引用可追溯的朴素 RAG 里，构建期过滤 + 零 token 重建使它不会发生。

## 诚实性约束
- 语料真实：PHEME 贴文，按 snowflake 反解的真实时间顺序。
- 成本真实：token 全部取自 API usage，墙钟实测；嵌入 CPU 上跑。
- 重建真实：naive 变体每晚全量重嵌 + 重建索引（夜间预算的真实构成）。
- 三方同机、同回次、同语料、同嵌入模型（MiniLM 384-d）、同 LLM（GLM-4-Flash）。

## 用法
  python e17_vanilla_rag.py --run probe --n_docs 300
  python e17_vanilla_rag.py --run cost  --n_docs 300
  python e17_vanilla_rag.py --run stale --n_docs 300
  python e17_vanilla_rag.py --run all   --n_docs 300
"""

from __future__ import annotations
# --- path resolution (released artifact) -----------------------------------
# No absolute path is baked in.  KB_ROOT points at the directory holding the
# experiment folders; it defaults to this repository's root and can be
# overridden with an environment variable.
import os as _os
_KB_ROOT = _os.environ.get(
    "KB_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)


def _KB(*parts):
    """Join *parts* onto the knowledge-base root."""
    return _os.path.join(_KB_ROOT, *parts)
# ---------------------------------------------------------------------------

import os, sys, json, time, re, argparse, threading
from pathlib import Path
from collections import defaultdict

ROOT = Path(_KB('experiments', 'E17_Baseline4_常规RAG重建')).resolve()
E14 = Path(_KB('experiments', 'E14_Baseline2_EraRAG'))
OUT = ROOT / "raw"
OUT.mkdir(parents=True, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

sys.path.insert(0, str(E14))
from e14_adapter import PhemeAdapter  # 只读复用，不修改

# ---------------- 凭据（与 E14 同 key / 同模型 / 同 90s 超时） ----------------
ENV_FILE = Path(_KB('.env_e14'))
_env = {}
for _line in ENV_FILE.read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if not _line or _line.startswith("#") or "=" not in _line:
        continue
    _k, _v = _line.split("=", 1)
    _env[_k.strip()] = _v.strip().strip('"').strip("'")
KEY, BASE, MODEL = _env["E14_LLM_API_KEY"], _env["E14_LLM_BASE_URL"], _env["E14_LLM_MODEL"]


def start_watchdog(seconds: int):
    def _kill():
        sys.stderr.write(f"\n[WATCHDOG] exceeded {seconds}s, force-exit\n")
        sys.stderr.flush()
        os._exit(3)
    t = threading.Timer(seconds, _kill)
    t.daemon = True
    t.start()
    return t


# =====================================================================
# LLM
# =====================================================================
from openai import OpenAI

_client = OpenAI(api_key=KEY, base_url=BASE, timeout=90.0, max_retries=3)
_TOK = {"prompt": 0, "completion": 0, "calls": 0}


def llm(system: str, user: str, temperature: float = 0.0, max_tokens: int = 400) -> str:
    r = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature, max_tokens=max_tokens)
    u = r.usage
    _TOK["prompt"] += getattr(u, "prompt_tokens", 0) or 0
    _TOK["completion"] += getattr(u, "completion_tokens", 0) or 0
    _TOK["calls"] += 1
    return (r.choices[0].message.content or "").strip()


def tok_reset():
    _TOK.update({"prompt": 0, "completion": 0, "calls": 0})


def tok_snap():
    return dict(_TOK)


# =====================================================================
# 嵌入（与 E14 同一模型 MiniLM 384-d，保证可比）
# =====================================================================
_MINILM = None


def embed(texts: list) -> np.ndarray:
    global _MINILM
    if _MINILM is None:
        from sentence_transformers import SentenceTransformer
        cand = list(E14.glob("_hf_cache/**/model.safetensors"))
        if not cand:
            raise RuntimeError("MiniLM not found in E14/_hf_cache")
        _MINILM = SentenceTransformer(str(cand[0].parent), device="cpu")
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    v = _MINILM.encode(texts, batch_size=32, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    return v.astype(np.float32)


# =====================================================================
# 语料
# =====================================================================
def load_corpus(n_docs: int):
    pa = PhemeAdapter().load()
    tw = list(pa.iter_tweets(labeled_only=True, min_text_len=10))
    tw.sort(key=lambda x: x["ms"])
    out = []
    for t in tw[:n_docs]:
        d = dict(t)
        d.setdefault("doc_id", t["tweet_id"])          # 统一字段名
        d.setdefault("thread", t["thread_idx"])        # 统一字段名
        d["is_claim"] = bool(t.get("is_source"))       # 源帖 = 声明（PHEME 线程级真值）
        out.append(d)
    return out


# =====================================================================
# 朴素 RAG（最有利实现）
# =====================================================================
SYS_RAG = (
    "You answer questions about a set of social-media posts. Use ONLY the numbered "
    "context items. Cite the items you use with their bracket number, e.g. [2]. "
    "If the context does not contain the answer, say so.")


class VanillaRAG:
    """mode='naive'    ：撤回排队，等下一个全量重建才生效（真实夜班做法）
       mode='history+' ：撤回 = 标记删除（零 token、即时生效），可 as-of 查历史
       两者检索/作答完全相同，只有维护路径不同，比较因此干净。"""

    def __init__(self, mode: str, docs: list):
        assert mode in ("naive", "history+")
        self.mode = mode
        self.docs = [dict(d) for d in docs]
        self.dead = set()
        self.mat = None

    def build(self) -> dict:
        t0 = time.perf_counter()
        self.mat = embed([d["text"] for d in self.docs])
        return {"wall_s": round(time.perf_counter() - t0, 3),
                "n_docs": len(self.docs), "vectors": len(self.docs)}

    def rebuild(self) -> dict:
        """夜间全量重建：naive = 全量重嵌（LLM token 为 0，嵌入算力为全量）；
        history+ = 标记删除已生效（零成本确认）。"""
        t0 = time.perf_counter()
        if self.mode == "naive":
            self.mat = embed([d["text"] for d in self.docs])
            self.dead = set()
        else:
            self.dead = set(self.dead)
        return {"wall_s": round(time.perf_counter() - t0, 3),
                "mode": self.mode, "n_docs": len(self.docs)}

    def upsert(self, new_docs: list) -> dict:
        t0 = time.perf_counter()
        if new_docs:
            v = embed([d["text"] for d in new_docs])
            self.docs.extend(dict(d) for d in new_docs)
            self.mat = v if self.mat is None else np.vstack([self.mat, v])
        return {"wall_s": round(time.perf_counter() - t0, 3), "n_added": len(new_docs)}

    def retract(self, doc_id) -> dict:
        if self.mode == "history+":
            self.dead.add(doc_id)
            return {"applied_immediately": True}
        return {"applied_immediately": False}   # naive：排队等重建

    def retrieve(self, q: str, k: int = 3, asof_ms=None) -> list:
        if self.mat is None or len(self.docs) == 0:
            return []
        qv = embed([q])[0]
        sims = self.mat @ qv
        order = np.argsort(-sims)
        out = []
        for i in order:
            d = self.docs[int(i)]
            if d["doc_id"] in self.dead:
                continue
            if asof_ms is not None and d["ms"] > asof_ms:
                continue
            out.append((float(sims[int(i)]), d))
            if len(out) >= k:
                break
        return out

    def answer(self, q: str, k: int = 3, asof_ms=None) -> dict:
        hits = self.retrieve(q, k=k, asof_ms=asof_ms)
        ctx = "\n".join(f"[{i+1}] {d['text']}" for i, (s, d) in enumerate(hits))
        prompt = f"Context items:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        ans = llm(SYS_RAG, prompt, temperature=0.0, max_tokens=300)
        cited_idx = sorted({int(x) for x in re.findall(r"\[(\d+)\]", ans)})
        cited = [hits[i - 1][1] for i in cited_idx if 1 <= i <= len(hits)]
        return {"query": q, "answer": ans, "n_context": len(hits),
                "context_ids": [d["doc_id"] for _, d in hits],
                "cited_ids": [d["doc_id"] for d in cited],
                "citation_traceable": len(cited) > 0}


# =====================================================================
# 我们的方法（同语料、同嵌入、同 LLM）：增量立方体 + 声明状态 + 血缘
# =====================================================================
class CubeIncremental:
    """cell = (day, thread)；度量 = 计数 + 支持集（可合并）；声明状态另行索引。
    对照三件事：增量成本（Q1）、撤回即时生效（Q2）、撤回后答案不再基于已撤回声明（Q3）。"""

    def __init__(self):
        self.cells = defaultdict(lambda: {"n": 0, "support": set()})
        self.claims = {}
        self.claim_cells = defaultdict(set)
        self.unit_updates = 0

    def ingest(self, docs: list) -> dict:
        t0 = time.perf_counter()
        u0 = self.unit_updates
        for d in docs:
            day = d["iso"][:10]
            key = (day, d["thread"])
            c = self.cells[key]
            c["n"] += 1
            c["support"].add(d["doc_id"])
            self.unit_updates += 1
            if d.get("is_claim"):
                self.claims[d["doc_id"]] = {"text": d["text"], "ms": d["ms"],
                                            "thread": d["thread"], "state": "asserted",
                                            "retracted_at": None}
                self.claim_cells[d["doc_id"]].add(key)
        return {"wall_s": round(time.perf_counter() - t0, 3), "n_docs": len(docs),
                "unit_updates": self.unit_updates - u0, "cells": len(self.cells)}

    def retract(self, claim_id, at_ms: int) -> dict:
        t0 = time.perf_counter()
        cl = self.claims.get(claim_id)
        if cl is None:
            return {"error": "unknown claim"}
        cl["state"] = "retracted"
        cl["retracted_at"] = at_ms
        affected = self.claim_cells.get(claim_id, set())
        for key in affected:
            c = self.cells[key]
            c["support"].discard(claim_id)
            c["n"] = max(0, c["n"] - 1)
        return {"claim": claim_id, "affected_cells": len(affected),
                "recomputed": len(affected), "cells_total": len(self.cells),
                "wall_s": round(time.perf_counter() - t0, 4),
                "applied_immediately": True}

    def answer(self, q: str, k: int = 3, asof_ms=None) -> dict:
        alive = [c for c in self.claims.values()
                 if c["state"] != "retracted"
                 and (asof_ms is None or c["ms"] <= asof_ms)]
        if not alive:
            return {"query": q, "answer": "", "n_context": 0, "context_ids": [],
                    "cited_ids": [], "citation_traceable": False}
        vecs = embed([c["text"] for c in alive])
        qv = embed([q])[0]
        sims = vecs @ qv
        order = np.argsort(-sims)[:k]
        ctx = "\n".join(f"[{i+1}] {alive[j]['text']}" for i, j in enumerate(order))
        prompt = (f"Context items (retracted claims are already excluded by the "
                  f"system):\n{ctx}\n\nQuestion: {q}\nAnswer:")
        ans = llm(SYS_RAG, prompt, temperature=0.0, max_tokens=300)
        cited_idx = sorted({int(x) for x in re.findall(r"\[(\d+)\]", ans)})
        cited = [alive[order[i - 1]] for i in cited_idx if 1 <= i <= len(order)]
        return {"query": q, "answer": ans, "n_context": len(order),
                "context_ids": [alive[j]["ms"] for j in order],
                "cited_ids": [c["ms"] for c in cited],
                "citation_traceable": len(cited) > 0}


# =====================================================================
# Q1 成本：每万文档的 token 与墙钟
# =====================================================================
PER_10K = 10_000


def per_10k(x, n):
    return round(x / n * PER_10K, 1)


def run_cost(n_docs: int) -> dict:
    print(f"\n=== Q1 cost: n_docs={n_docs} ===", flush=True)
    docs = load_corpus(n_docs)

    # --- naive 建索引 ---
    tok_reset(); t0 = time.perf_counter()
    rag = VanillaRAG("naive", docs)
    rep_build = rag.build()
    build_wall = time.perf_counter() - t0
    build_tok = tok_snap()
    print(f"  naive build: wall={build_wall:.1f}s tok={build_tok}", flush=True)

    # --- naive 全量重建（夜间） ---
    tok_reset(); t0 = time.perf_counter()
    rep_rebuild = rag.rebuild()
    rebuild_wall = time.perf_counter() - t0
    rebuild_tok = tok_snap()
    print(f"  naive rebuild: wall={rebuild_wall:.1f}s tok={rebuild_tok}", flush=True)

    # --- history+ 建索引（同语料） ---
    tok_reset(); t0 = time.perf_counter()
    ragh = VanillaRAG("history+", docs)
    rep_buildh = ragh.build()
    buildh_wall = time.perf_counter() - t0
    buildh_tok = tok_snap()

    # --- history+ 撤回（零 token，即时） ---
    victims = [d["doc_id"] for d in docs[:10]]
    tok_reset(); t0 = time.perf_counter()
    for vid in victims:
        ragh.retract(vid)
    up_wall = time.perf_counter() - t0
    up_tok = tok_snap()
    print(f"  history+ retract(10): wall={up_wall:.4f}s tok={up_tok}", flush=True)

    # --- 我们的增量（同语料） ---
    tok_reset(); t0 = time.perf_counter()
    cube = CubeIncremental()
    rep_cube = cube.ingest(docs)
    cube_wall = time.perf_counter() - t0
    cube_tok = tok_snap()
    retracts = []
    for vid in victims:
        if vid in cube.claims:
            retracts.append(cube.retract(vid, at_ms=0))
    print(f"  cube ingest: wall={cube_wall:.2f}s unit_updates={rep_cube['unit_updates']}", flush=True)

    return {
        "n_docs": n_docs,
        "naive_build": {**rep_build, "wall_s": round(build_wall, 3), "tokens": build_tok,
                        "per_10k": {"wall_s": per_10k(build_wall, n_docs),
                                    "prompt_tok": per_10k(build_tok["prompt"], n_docs),
                                    "completion_tok": per_10k(build_tok["completion"], n_docs),
                                    "embed_vectors": per_10k(n_docs, n_docs)}},
        "naive_rebuild": {**rep_rebuild, "wall_s": round(rebuild_wall, 3), "tokens": rebuild_tok,
                          "per_10k": {"wall_s": per_10k(rebuild_wall, n_docs),
                                      "prompt_tok": per_10k(rebuild_tok["prompt"], n_docs),
                                      "completion_tok": per_10k(rebuild_tok["completion"], n_docs),
                                      "embed_vectors": per_10k(n_docs, n_docs)}},
        "history_build": {**rep_buildh, "wall_s": round(buildh_wall, 3), "tokens": buildh_tok},
        "history_retract": {"wall_s": round(up_wall, 4), "tokens": up_tok,
                            "n_retracted": len(victims)},
        "cube_ingest": {**{**rep_cube, "wall_s": round(cube_wall, 4)}, "tokens": cube_tok,
                        "per_10k": {"wall_s": round(cube_wall / n_docs * PER_10K, 3),
                                    "unit_updates": per_10k(rep_cube["unit_updates"], n_docs)}},
        "cube_retracts": retracts,
    }


# =====================================================================
# Q2 时效：撤回 → 下一重建周期之间的陈旧窗口
# =====================================================================
STALE_QUERIES = [
    "What did the original post claim, and do the replies support it?",
    "Summarise the main claim being discussed in these posts.",
    "Is there any post contradicting the source claim?",
    "What is the overall verdict on the rumour in this collection?",
    "Which posts dispute or refute the original claim?",
    "Do the replies confirm the rumour or debunk it?",
    "What is being alleged, and by whom?",
    "Has anyone challenged the accuracy of the original post?",
]


def run_stale(n_docs: int, rebuild_frac: float = 0.10, top_k: int = 3) -> dict:
    """撤回一条仍在被检索的源帖声明；naive 在 (rebuild_frac × half) 条新文档
    到达、下一次全量重建时才生效；history+ 与 ours 即时生效。
    窗口内问 STALE_QUERIES，统计 naive 把已撤回文档当上下文的比例。"""
    print(f"\n=== Q2 staleness: n_docs={n_docs} rebuild_frac={rebuild_frac} ===", flush=True)
    docs = load_corpus(n_docs)
    half = len(docs) // 2

    base_docs = docs[:half]
    later_docs = docs[half:]

    rag = VanillaRAG("naive", base_docs)
    rag.build()

    # 靶标选择：撤回实验的前提是"被撤回的声明正在被检索"。
    # 先探测：哪个源帖声明在 STALE_QUERIES 的 top-k 中出现最频繁，选它做靶标。
    claim_docs = [d for d in base_docs if d.get("is_claim")]
    hitcount = {}
    for d in claim_docs:
        cnt = 0
        for q in STALE_QUERIES:
            hits = rag.retrieve(q, k=top_k)
            if any(h[1]["doc_id"] == d["doc_id"] for h in hits):
                cnt += 1
        hitcount[d["doc_id"]] = cnt
    best_id = max(hitcount, key=hitcount.get)
    if hitcount[best_id] == 0:
        # 兜底：取与查询平均相似度最高的声明（并在报告中如实标注检索率）
        sims = []
        qvs = embed(STALE_QUERIES)
        cvecs = embed([d["text"] for d in claim_docs])
        m = cvecs @ qvs.T
        best_i = int(np.argmax(m.mean(axis=1)))
        victim = claim_docs[best_i]
        print(f"  [warn] no claim reachable in top-{top_k}; "
              f"fallback = highest-mean-sim claim", flush=True)
    else:
        victim = next(d for d in claim_docs if d["doc_id"] == best_id)
    print(f"  victim claim: {victim['doc_id']} ({victim['iso']}) "
          f"top-{top_k} hit {hitcount[victim['doc_id']]}/{len(STALE_QUERIES)} queries :: "
          f"{victim['text'][:60]!r}", flush=True)

    ragh = VanillaRAG("history+", base_docs)
    ragh.build()
    cube = CubeIncremental()
    cube.ingest(base_docs)

    rag.retract(victim["doc_id"])      # naive：排队，等重建
    ragh.retract(victim["doc_id"])     # history+：即时
    cube_ret = None
    if victim["doc_id"] in cube.claims:
        cube_ret = cube.retract(victim["doc_id"], at_ms=0)

    rebuild_after = max(1, int(half * rebuild_frac))

    def probe_all(tag):
        out = []
        for q in STALE_QUERIES:
            a1 = rag.answer(q, k=top_k)
            a2 = ragh.answer(q, k=top_k)
            a3 = cube.answer(q, k=top_k)
            rec = {"tag": tag, "query": q,
                   "naive_ctx_has_retracted": victim["doc_id"] in a1["context_ids"],
                   "naive_cited_retracted": victim["doc_id"] in a1["cited_ids"],
                   "naive_answer_head": a1["answer"][:200],
                   "hist_ctx_has_retracted": victim["doc_id"] in a2["context_ids"],
                   "hist_answer_head": a2["answer"][:200],
                   "ours_answer_head": a3["answer"][:200],
                   "ours_n_context": a3["n_context"]}
            out.append(rec)
            print(f"    [{tag}] {q[:44]!r} naive_ctx_retracted={rec['naive_ctx_has_retracted']} "
                  f"naive_cited={rec['naive_cited_retracted']}", flush=True)
        return out

    # --- 窗口内（重建前，撤回已排队未生效）---
    window = probe_all("window")

    # --- 新文档到达，naive 才重建 ---
    rag.upsert(later_docs[:rebuild_after])
    ragh.upsert(later_docs[:rebuild_after])
    cube.ingest(later_docs[:rebuild_after])
    rep_re = rag.rebuild()
    print(f"  naive rebuild happens after {rebuild_after} new docs "
          f"(embed_wall={rep_re['wall_s']}s)", flush=True)

    # --- 重建后 ---
    after = probe_all("after_rebuild")

    return {
        "n_docs": n_docs, "half": half, "rebuild_after": rebuild_after,
        "victim": {"doc_id": victim["doc_id"], "iso": victim["iso"],
                   "text_head": victim["text"][:80], "label": victim["label"],
                   "topk_hit_queries": hitcount[victim["doc_id"]],
                   "n_queries": len(STALE_QUERIES)},
        "window_probes": window,
        "after_probes": after,
        "naive_window": {
            "n_queries": len(window),
            "retracted_served": sum(1 for p in window if p["naive_ctx_has_retracted"]),
            "retracted_cited": sum(1 for p in window if p["naive_cited_retracted"]),
        },
        "history_window_clean": all(not p["hist_ctx_has_retracted"] for p in window),
        "cube_retract": cube_ret,
    }


# =====================================================================
# main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="probe", choices=["probe", "cost", "stale", "all"])
    ap.add_argument("--n_docs", type=int, default=300)
    ap.add_argument("--rebuild_frac", type=float, default=0.10)
    ap.add_argument("--watchdog", type=int, default=5400)
    args = ap.parse_args()
    start_watchdog(args.watchdog)

    if args.run in ("probe", "all"):
        docs = load_corpus(min(args.n_docs, 50))
        t0 = time.perf_counter()
        v = embed([d["text"] for d in docs[:50]])
        print(f"[embed] 50 docs in {time.perf_counter()-t0:.2f}s shape={v.shape}", flush=True)
        a = llm("You are a test.", "Reply with the single word OK.")
        print(f"[llm] {a!r} tokens={tok_snap()}", flush=True)

    if args.run in ("cost", "all"):
        r1 = run_cost(args.n_docs)
        (OUT / f"cost_n{args.n_docs}.json").write_text(
            json.dumps(r1, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {OUT / f'cost_n{args.n_docs}.json'}", flush=True)

    if args.run in ("stale", "all"):
        for frac in (args.rebuild_frac, 0.01, 0.001):
            r2 = run_stale(args.n_docs, rebuild_frac=frac)
            tag = str(frac).replace(".", "_")
            (OUT / f"stale_n{args.n_docs}_f{tag}.json").write_text(
                json.dumps(r2, ensure_ascii=False, indent=2), encoding="utf-8")
            nw = r2["naive_window"]
            print(f"[saved] stale f={frac}: naive retracted-served "
                  f"{nw['retracted_served']}/{nw['n_queries']}", flush=True)

    print("\n[E17] done.", flush=True)


if __name__ == "__main__":
    main()
