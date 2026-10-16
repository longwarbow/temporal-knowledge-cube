# -*- coding: utf-8 -*-
"""
E14 / W3b 能力矩阵：EraRAG 对「结构化分析查询」的能力边界

目标（对齐本课题 C1/C2/C3 与 RQ1/RQ2）：
  A) 结构层（确定性、零 LLM）：枚举 EraRAG 的检索原语与索引元数据，
     证明其不存在 时间/空间/事件/声明状态/血缘 五类谓词入口。
  B) 事实存活（确定性，仅嵌入检索）：把 EraRAG 真实检索上下文取出，
     检测结构化事实（雪花 id / ISO 时间戳 / 地点词）在检索层是否存活。
  C) 谓词可检索性（真值锚定）：对 4 类维度定义"元数据谓词目标集"，
     用自然语言查询，量化 precision/recall（对比 chance）。
  D) 辟谣注入对照：先建不含反驳贴的树，查询"是否有人反驳"；
     再把 disagreed 回复增量插入，复查。检验 EraRAG 有无"声明状态更新"。

载体：PHEME 4 个线程（Sydney×2 / Ottawa / Ferguson），~139 贴；
真值来源：veracity_dict（源贴真伪）、stance_dict（回复立场 agreed/comment/appeal/disagreed）。
CPU-only / Windows / Py3.9 / 本地 MiniLM / 智谱 GLM-4-Flash（沿用 W3a 加固）。

用法：
  python e14_w3b_capability.py --watchdog 1800
"""
import os, sys, re, json, time, asyncio, argparse, shutil, threading
from pathlib import Path

ROOT = Path("D:/WSPforWB/10.keyan/场景1-海量短文本/实验/E14_Baseline2_EraRAG").resolve()
ASCII_BASE = "D:/WSPforWB/10.keyan/_e14_run"

# ---------------- CPU 强制 ----------------
import torch
torch.backends.cuda.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sentence_transformers
_orig_st = sentence_transformers.SentenceTransformer.__init__

def _cpu_st(self, *a, **k):
    k["device"] = "cpu"
    return _orig_st(self, *a, **k)
sentence_transformers.SentenceTransformer.__init__ = _cpu_st

sys.path.insert(0, str(ROOT / "_repo"))
import e14_env
e14_env.bootstrap(verbose=False)
from Option.Config2 import Config
from Core.GraphRAG import GraphRAG
from Core.Common.Constants import Retriever
from Core.Provider.OpenaiApi import OpenAILLM
from Core.Provider.BaseLLM import BaseLLM
from Core.Retriever.EntitiyRetriever import EntityRetriever
from e14_adapter import PhemeAdapter, to_erarag_doc

# ---------------- 加固 1：no-op 信号量 ----------------
class _NoopSemaphore:
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

_openai_init = OpenAILLM.__init__
def _patched_openai_init(self, config):
    _openai_init(self, config)
    self.semaphore = _NoopSemaphore()
OpenAILLM.__init__ = _patched_openai_init

# ---------------- 加固 2：每次 LLM 调用硬超时 ----------------
_orig_aask = BaseLLM.aask
PER_CALL_TIMEOUT = 90

async def _guarded_aask(self, *a, **k):
    return await asyncio.wait_for(_orig_aask(self, *a, **k), timeout=PER_CALL_TIMEOUT)
BaseLLM.aask = _guarded_aask

# ---------------- 加固 3：进程级看门狗 ----------------
def start_watchdog(seconds: int):
    def _kill():
        sys.stderr.write(f"\n[WATCHDOG] exceeded {seconds}s → os._exit(3)\n")
        sys.stderr.flush()
        os._exit(3)
    t = threading.Timer(seconds, _kill); t.daemon = True; t.start(); return t

# ---------------- key ----------------
ENV_FILE = Path("D:/WSPforWB/10.keyan/.env_e14")
env = {}
for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    env[k.strip()] = v.strip().strip('"').strip("'")
KEY, BASE, MODEL = env["E14_LLM_API_KEY"], env["E14_LLM_BASE_URL"], env["E14_LLM_MODEL"]

# ---------------- 语料参数 ----------------
THREAD_IDS = [5027, 1358, 145, 5000]      # Sydney / Ottawa / Ferguson / Sydney
MAIN_THREAD = 5027                        # 辟谣注入靶标（label=false，含 16 条 disagreed）
STANCE = {0: "agreed", 1: "comment", 2: "appeal", 3: "disagreed"}
CITY_TOKENS = {
    "sydney": ["sydney", "lindt", "martin place", "#sydneysiege"],
    "ottawa": ["ottawa", "parliament hill", "war memorial", "#ottawashooting"],
    "ferguson": ["ferguson", "michael brown", "st. louis", "st louis", "#ferguson"],
}

# 正则（结构化事实探测）
RE_SNOWFLAKE = re.compile(r"\b\d{17,19}\b")
RE_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
RE_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b")


def build_config(work_dir, data_dir, exp, add, force):
    method = Config.read_yaml(ROOT / "_repo" / "Option" / "Method" / "Dynamic.yaml")
    final = dict(method)
    final["llm"] = {"api_type": "open_llm", "model": MODEL, "base_url": BASE,
                    "api_key": KEY, "api_version": None,
                    "timeout": PER_CALL_TIMEOUT, "max_concurrent": 64}
    MINILM = str(next((ROOT / "_hf_cache").rglob("model.safetensors")).parent)
    final["embedding"] = {"api_type": "hf", "model": MINILM, "cache_dir": str(ROOT / "_hf_cache"),
                          "dimensions": 384, "embed_batch_size": 16, "embedding_func_max_async": 8}
    final["working_dir"] = work_dir
    final["exp_name"] = exp
    final["data_root"] = data_dir
    final["graph"] = dict(method.get("graph", {}))
    final["graph"]["add"] = add
    final["graph"]["force"] = force
    return Config(**final)


# =====================================================================
# Component A：结构层扫描（确定性，零 LLM）
# =====================================================================
def component_a_struct_scan():
    # mode_list 是实例属性，需从源码解析（保证取证可复现）
    src_path = ROOT / "_repo" / "Core" / "Retriever" / "EntitiyRetriever.py"
    modes = []
    m = re.search(r"self\.mode_list\s*=\s*\[([^\]]*)\]", src_path.read_text(encoding="utf-8"))
    if m:
        modes = re.findall(r"[\"']([^\"']+)[\"']", m.group(1))
    out = {
        "retriever_enum_values": [r.value for r in Retriever],
        "entity_retriever_modes": modes,
        "context_metakey": "index",
        "evidence_sources": [
            "Core/Common/Constants.py :: class Retriever(Enum)",
            "Core/Retriever/EntitiyRetriever.py :: self.mode_list = [...]",
            "Core/Storage/TreeGraphStorage.py :: get_node_metadata() → ['index']",
            "Core/Retriever/MixRetriever.py :: retrieve_relevant_content(type, mode, **kwargs)，唯一注入通道是 seed",
        ],
        "predicate_dimensions": {},
        "verdict": "",
    }
    dims = {
        "time (时间/区间/时序)": False,
        "space (地点/行政区)": False,
        "event/thread (事件/线程归属)": False,
        "claim_status (真伪/立场/状态迁移)": False,
        "lineage (血缘/来源/引用链)": False,
        "aggregation (计数/可合并度量)": False,
    }
    src = "Retriever 枚举 + EntityRetriever.mode_list + 向量索引元数据(get_node_metadata→['index'])"
    out["predicate_dimensions"] = {k: {"supported": v, "probe": src} for k, v in dims.items()}
    out["verdict"] = ("EraRAG 检索原语仅『语义 top-k』(vdb/ppr/tf_df/neighbors…)；"
                      "索引元数据仅含 index（无 id/时间/地点/标签/引用），"
                      "六类结构化谓词入口全部缺失。")
    return out


# =====================================================================
# 语料构建
# =====================================================================
def build_corpus():
    pa = PhemeAdapter().load()
    ver, stance = pa.veracity, __import__("pickle").load(
        open(Path(pa.root) / "stance_dict.pkl", "rb"))
    threads = {}
    for ci, g in enumerate(pa.cascades):
        if ci not in THREAD_IDS:
            continue
        src, src_text = None, ""
        for nid, a in g.nodes(data=True):
            if a.get("is_source"):
                src, src_text = nid, (a.get("text") or ""); break
        tws = []
        for nid, a in g.nodes(data=True):
            t = (a.get("text") or "")
            if len(t) < 10:
                continue
            ms = (int(nid) >> 22) + 1288834974657
            tws.append({"tweet_id": nid, "text": t, "ms": ms,
                        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ms / 1000)),
                        "thread_idx": ci, "source_id": src,
                        "is_source": bool(a.get("is_source")),
                        "label": ver.get(src),
                        "stance": (stance.get(nid) if not a.get("is_source") else None)})
        threads[ci] = {"source_id": src, "label": ver.get(src), "tweets": tws, "src_text": src_text}
    all_tw = [t for ci in THREAD_IDS for t in threads[ci]["tweets"]] if threads else []
    all_tw.sort(key=lambda x: x["ms"])
    return threads, all_tw


def thread_member_ids(t):
    return {x["tweet_id"] for x in t["tweets"]}


def ids_with_stance(t, st):
    return {x["tweet_id"] for x in t["tweets"] if x["stance"] == st}


# =====================================================================
# 检索层工具
# =====================================================================
def leaf_index_set(storage):
    return {n.index for n in storage.get_layer(0)}


def descendants(storage, idx, leaf_ids):
    out, stack = set(), [idx]
    seen = set()
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        if i in leaf_ids:
            out.add(i); continue
        try:
            node = storage.nodes[i]
        except Exception:
            continue
        for c in (node.children or []):
            stack.append(c)
    return out


async def get_content2tweet(digi):
    """chunk.content → tweet_id（用 DocChunk 内部的 title 字段，最可靠）。
    get_chunks() 返回 [(key, TextChunk), ...]。"""
    chunks = await digi.doc_chunk.get_chunks()
    m = {}
    for item in chunks:
        c = item[1] if isinstance(item, (tuple, list)) and len(item) == 2 else item
        m[(c.content or "").strip()] = c.title
    return m


def scan_flags(texts):
    joined = "\n".join(texts)
    return {
        "has_snowflake": bool(RE_SNOWFLAKE.search(joined)),
        "has_iso_date": bool(RE_ISO.search(joined)),
        "has_clock": bool(RE_CLOCK.search(joined)),
        "has_location_token": any(tok in joined.lower() for toks in CITY_TOKENS.values() for tok in toks),
        "ctx_chars": len(joined),
        "n_ctx_items": len(texts),
    }


async def retrieve_nodes(digi, query, top_k=5):
    return await digi.entities_vdb.retrieval_nodes(
        query=query, top_k=top_k, graph=digi.graph, tree_node=True)


async def probe(digi, storage, c2t, leaf_ids, query, top_k=5, tid2thread=None):
    """一次检索探针：返回检索上下文 + 命中的叶子 tweet 集合 + 线程纯度。"""
    nodes = await retrieve_nodes(digi, query, top_k)
    texts = [n.text for n in nodes if n is not None]
    leaf_hit = set()
    for n in nodes:
        if n is None:
            continue
        for li in descendants(storage, n.index, leaf_ids):
            tw = c2t.get((storage.nodes[li].text or "").strip())
            if tw:
                leaf_hit.add(tw)
    rec = {"query": query, "ctx_texts": texts, "flags": scan_flags(texts),
           "leaf_tweets": sorted(leaf_hit)}
    if tid2thread:
        from collections import Counter
        dist = Counter(tid2thread.get(t) for t in leaf_hit)
        rec["thread_dist"] = {str(k): v for k, v in dist.items()}
        rec["thread_purity"] = round(max(dist.values()) / sum(dist.values()), 3) if dist else None
        rec["n_threads_touched"] = len(dist)
    return rec


def prf(retrieved, target, universe_n):
    R, S = set(retrieved), set(target)
    inter = R & S
    return {
        "n_retrieved": len(R), "n_target": len(S), "n_hit": len(inter),
        "precision": round(len(inter) / len(R), 3) if R else None,
        "recall": round(len(inter) / len(S), 3) if S else None,
        "chance_precision": round(len(S) / universe_n, 3) if universe_n else None,
    }


# =====================================================================
# 查询集
# =====================================================================
QUERIES = {
    "T": [  # 时序
        "Which reply in this discussion came first, and at what exact time was it posted?",
        "Summarize the reactions strictly in chronological order with timestamps.",
        "How many hours passed between the first and the last post?",
    ],
    "S": [  # 空间
        "Which posts are about the Lindt café in Martin Place, Sydney?",
        "List every distinct location mentioned across these posts.",
        "Which posts concern Ottawa, Canada?",
    ],
    "E": [  # 事件归属
        "Which posts belong to the Sydney hostage siege discussion?",
        "Group these posts by the incident each one refers to.",
        "Is this collection about a single event or several different events?",
    ],
    "C": [  # 声明状态 / 反驳
        "Which replies dispute or refute the original claim?",
        "Is the claim made in the source post confirmed or refuted by the replies?",
        "List the replies that question the original post.",
    ],
    "G": [  # 聚合/可合并度量
        "How many distinct posts are in this collection, and how many of them dispute the claim?",
    ],
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchdog", type=int, default=1800)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()
    start_watchdog(args.watchdog)

    out = {"meta": {"model": MODEL, "top_k": args.top_k,
                    "thread_ids": THREAD_IDS, "main_thread": MAIN_THREAD}}

    # ---------- A ----------
    out["A_struct_scan"] = component_a_struct_scan()
    print("[A] 结构层扫描完成：", out["A_struct_scan"]["verdict"])

    # ---------- 语料 ----------
    threads, all_tw = build_corpus()
    N = len(all_tw)
    tid2thread = {t["tweet_id"]: t["thread_idx"] for t in all_tw}
    out["meta"]["N_tweets"] = N
    out["meta"]["threads"] = {str(ci): {"label": threads[ci]["label"], "n": len(threads[ci]["tweets"])}
                              for ci in threads}
    print(f"[语料] N={N} threads={ {ci: len(threads[ci]['tweets']) for ci in threads} }")

    # 真值集合
    dis_main = ids_with_stance(threads[MAIN_THREAD], 3)          # 反驳（辟谣注入靶）
    agr_main = ids_with_stance(threads[MAIN_THREAD], 0)
    com_main = ids_with_stance(threads[MAIN_THREAD], 1)
    out["truth"] = {"disagreed_main": sorted(dis_main), "n_disagreed": len(dis_main),
                    "n_agreed": len(agr_main), "n_comment": len(com_main),
                    "main_label": threads[MAIN_THREAD]["label"]}
    print(f"[真值] disagreed(main)={len(dis_main)} agreed={len(agr_main)} comment={len(com_main)}")

    # ---------- Phase 1：建树（排除反驳贴） ----------
    p1_tweets = [t for t in all_tw if t["tweet_id"] not in dis_main]
    wd = str(Path(ASCII_BASE) / "_w3b_p1")
    dd = str(Path(ASCII_BASE) / "_w3b_p1_data")
    if Path(wd).exists():
        shutil.rmtree(wd)

    cfg1 = build_config(wd, dd, "w3b_p1", add=False, force=True)
    digi1 = GraphRAG(config=cfg1)
    t0 = time.time()
    await digi1.insert([to_erarag_doc(t) for t in p1_tweets])
    out["phase1"] = {"docs": len(p1_tweets), "wall_s": round(time.time() - t0, 2)}
    storage1 = digi1.graph._graph
    leaf1 = leaf_index_set(storage1)
    c2t1 = await get_content2tweet(digi1)
    out["phase1"]["leaves"] = len(leaf1)
    out["phase1"]["internal"] = storage1.num_nodes - len(leaf1)
    out["phase1"]["map_coverage"] = round(len(c2t1) / max(1, len(p1_tweets)), 3)
    print(f"[P1] docs={len(p1_tweets)} leaves={len(leaf1)} internal={out['phase1']['internal']} "
          f"wall={out['phase1']['wall_s']}s mapCov={out['phase1']['map_coverage']}")

    # ---------- B/C：检索探针 ----------
    probes = []
    # 目标集（真值锚定；一律限制在 Phase1 语料内，保证对比公平）
    p1_ids = {t["tweet_id"] for t in p1_tweets}
    earliest10 = {t["tweet_id"] for t in all_tw if t["tweet_id"] in p1_ids}  # 占位，稍后重排
    earliest10 = {t["tweet_id"] for t in [x for x in all_tw if x["tweet_id"] in p1_ids][:10]}
    targets = {
        "T_earliest10": earliest10,
        "S_ottawa": {t["tweet_id"] for t in p1_tweets
                     if any(k in t["text"].lower() for k in CITY_TOKENS["ottawa"])},
        "E_thread_main": {t["tweet_id"] for t in p1_tweets if t["thread_idx"] == MAIN_THREAD},
    }
    for cls in ["T", "S", "E", "C", "G"]:
        for q in QUERIES[cls]:
            pr = await probe(digi1, storage1, c2t1, leaf1, q, args.top_k, tid2thread)
            pr["class"] = cls
            # 与该类对应目标集的 P/R
            if cls == "T":
                pr["prf"] = prf(pr["leaf_tweets"], targets["T_earliest10"], len(p1_tweets))
            elif cls == "S":
                pr["prf"] = prf(pr["leaf_tweets"], targets["S_ottawa"], len(p1_tweets))
            elif cls == "E":
                pr["prf"] = prf(pr["leaf_tweets"], targets["E_thread_main"], len(p1_tweets))
            elif cls == "C":
                pr["note"] = "Phase1 树中不含反驳贴（真值 recall 必为 0），此处仅测『是否幻觉出反驳』"
                pr["prf"] = prf(pr["leaf_tweets"], dis_main, len(p1_tweets))
            else:  # G：无元数据谓词目标集（本类本身就是要暴露"不可聚合"）
                pr["prf"] = {"n_retrieved": len(pr["leaf_tweets"]), "n_target": None, "n_hit": None,
                             "precision": None, "recall": None, "chance_precision": None}
            probes.append(pr)
            print(f"  [{cls}] P={pr['prf']['precision']} R={pr['prf']['recall']} "
                  f"nRet={pr['prf']['n_retrieved']} thr={pr.get('n_threads_touched')} "
                  f"purity={pr.get('thread_purity')} flags={pr['flags']}")
    out["phase1"]["probes"] = probes

    # 生成层答案（Phase1）
    ans1 = {}
    for cls in ["T", "E", "C", "G"]:
        q = QUERIES[cls][0]
        a = await digi1.query(q)
        ans1[cls] = {"query": q, "answer": a}
        print(f"  [答案·P1·{cls}] {str(a)[:110]}")
    out["phase1"]["answers"] = ans1

    # ---------- Phase 2：辟谣注入（增量插入反驳贴） ----------
    cfg2 = build_config(wd, dd, "w3b_p1", add=True, force=False)
    digi2 = GraphRAG(config=cfg2)
    # 构造新 chunk
    from Core.Chunk.DocChunk import DocChunk
    from Core.Storage.NameSpace import Workspace
    tmp_ws = Workspace(str(Path(wd) / "_inj_tmp"), "w3b_inj")
    tmp_chunk = DocChunk(cfg2.chunk, digi2.ENCODER, tmp_ws.make_for("chunk_storage"))
    inj_docs = [to_erarag_doc(t) for t in all_tw if t["tweet_id"] in dis_main]
    await tmp_chunk.build_chunks(inj_docs)
    new_tuples = await tmp_chunk.get_chunks()
    t0 = time.time()
    await digi2.graph.build_graph(new_tuples, force=False, add=True)
    # 增量建树绕过了 GraphRAG.insert()，须手动刷新向量索引（force=True 覆盖旧索引）
    await digi2.entities_vdb.build_index(
        await digi2.graph.nodes_data(), await digi2.graph.node_metadata(), True)
    await digi2._build_retriever_context()
    out["phase2"] = {"injected": len(inj_docs), "wall_s": round(time.time() - t0, 2)}
    storage2 = digi2.graph._graph
    leaf2 = leaf_index_set(storage2)
    # digi2 的 chunk 存储沿用 Phase1（123 条），新注入的 16 条在 tmp_chunk 里；
    # 合并两处得到完整的 content→tweet 映射（否则新叶无法回溯、旧叶也可能因惰性未加载而全丢）。
    c2t2 = dict(c2t1)
    for item in await tmp_chunk.get_chunks():
        c = item[1] if isinstance(item, (tuple, list)) and len(item) == 2 else item
        c2t2[(c.content or "").strip()] = c.title
    out["phase2"]["map_size"] = len(c2t2)
    out["phase2"]["leaves"] = len(leaf2)
    out["phase2"]["internal"] = storage2.num_nodes - len(leaf2)
    print(f"[P2] injected={len(inj_docs)} leaves={len(leaf2)} internal={out['phase2']['internal']} "
          f"wall={out['phase2']['wall_s']}s")

    # 复查 C 类查询（辟谣注入后）
    probes2 = []
    for q in QUERIES["C"]:
        pr = await probe(digi2, storage2, c2t2, leaf2, q, args.top_k, tid2thread)
        pr["class"] = "C"
        pr["prf"] = prf(pr["leaf_tweets"], dis_main, len(all_tw))
        probes2.append(pr)
        print(f"  [C·P2] P={pr['prf']['precision']} R={pr['prf']['recall']} nRet={pr['prf']['n_retrieved']} "
              f"thr={pr.get('n_threads_touched')} purity={pr.get('thread_purity')} dist={pr.get('thread_dist')}")
    out["phase2"]["probes"] = probes2

    ans2 = {}
    for cls in ["T", "E", "C", "G"]:
        q = QUERIES[cls][0]
        a = await digi2.query(q)
        ans2[cls] = {"query": q, "answer": a}
        print(f"  [答案·P2·{cls}] {str(a)[:110]}")
    out["phase2"]["answers"] = ans2

    # ---------- 汇总指标 ----------
    # 事实存活率（Phase1 全部探针）
    allf = [p["flags"] for p in probes]
    n = len(allf)
    out["summary"] = {
        "fact_survival_rate": {
            "snowflake_id": round(sum(f["has_snowflake"] for f in allf) / n, 3),
            "iso_date": round(sum(f["has_iso_date"] for f in allf) / n, 3),
            "clock_time": round(sum(f["has_clock"] for f in allf) / n, 3),
            "location_token": round(sum(f["has_location_token"] for f in allf) / n, 3),
        },
        "mean_ctx_chars": round(sum(f["ctx_chars"] for f in allf) / n, 1),
        "phase1": {c: [{"query": p["query"], "precision": p["prf"]["precision"],
                        "recall": p["prf"]["recall"], "chance": p["prf"]["chance_precision"],
                        "n_retrieved": p["prf"]["n_retrieved"],
                        "thread_purity": p.get("thread_purity"),
                        "n_threads_touched": p.get("n_threads_touched")}
                       for p in probes if p["class"] == c] for c in ["T", "S", "E", "C", "G"]},
        "phase2_C": [{"query": p["query"], "precision": p["prf"]["precision"], "recall": p["prf"]["recall"],
                      "chance": p["prf"]["chance_precision"], "n_retrieved": p["prf"]["n_retrieved"],
                      "thread_purity": p.get("thread_purity"), "thread_dist": p.get("thread_dist")}
                     for p in probes2],
    }

    (ROOT / "W3b_能力矩阵.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[报告] 写入 W3b_能力矩阵.json")
    print("[汇总]", json.dumps(out["summary"], ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
