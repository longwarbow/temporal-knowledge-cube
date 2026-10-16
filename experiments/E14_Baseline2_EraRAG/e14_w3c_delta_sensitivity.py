# -*- coding: utf-8 -*-
"""
E14 / W3c  Δ 敏感性：批大小对流式增量插入的影响

问题：EraRAG 动态树在「流式到达」场景下，把同一批 N_total 条语料，
      按不同批大小 Δ 分批（add=True）增量插入，其
        ① 成本（token / 墙钟）
        ② 边际成本（每批边际 token / 时间）
        ③ 终态结构（叶子/内部节点）与一致性
      如何随 Δ 变化？

设计（与 W3a 对齐、复用其加固框架）：
  - 固定总语料 N_total（默认 200，PHEME labeled，时间升序）
  - 首批 N0=40 条 force 建树；其余 (N_total-N0) 条按 Δ 分批 add=True 增量
  - Δ ∈ {8,16,32,80,160} → 批次数 20/10/5/2/1
  - 每批记录 {n, leaves, internal, cum_tokens, cum_wall}，差分得边际成本
  - 另跑一次「同 N_total 全量重建」作对照（--with_full）

单 Δ 单进程运行（跨事件循环死锁需进程隔离），结果写 W3c_delta{Δ}.json

用法：
  python e14_w3c_delta_sensitivity.py --delta 32
  python e14_w3c_delta_sensitivity.py --delta 32 --with_full
"""
import os, sys, json, time, asyncio, argparse, shutil, threading
from pathlib import Path

ROOT = Path("D:/WSPforWB/10.keyan/场景1-海量短文本/实验/E14_Baseline2_EraRAG").resolve()
ASCII_BASE = "D:/WSPforWB/10.keyan/_e14_run"

# ---------------- CPU 强制（同 W1/W2/W3a） ----------------
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
from Core.Chunk.DocChunk import DocChunk
from Core.Storage.NameSpace import Workspace
from Core.Provider.OpenaiApi import OpenAILLM
from e14_adapter import PhemeAdapter, to_erarag_doc

# ---------------- 加固 1：loop-无关 no-op 信号量 ----------------
class _NoopSemaphore:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False

_openai_init = OpenAILLM.__init__
def _patched_openai_init(self, config):
    _openai_init(self, config)
    self.semaphore = _NoopSemaphore()
OpenAILLM.__init__ = _patched_openai_init

# ---------------- 加固 2a：根治跨 loop 客户端复用（缺陷 #9 的真正根源） ----------------
# W3a 加固只治了信号量与调用超时，但 AsyncOpenAI(→httpx.AsyncClient) 仍在主 loop
# 创建、被线程池新 loop 复用 —— anyio 内部 Event/Lock 绑定创建 loop，跨 loop 等
# 待即永久挂起（实测 Δ=160 挂死 ~7h：23 时 634 行 → 00-05 时停滞 → 06 时恢复）。
# 根治：每次调用在「当前 loop」新建客户端、用完即毁；cost 计量保留。
from openai import AsyncOpenAI as _FreshAsyncOpenAI

async def _loop_safe_achat(self, messages, timeout=None, max_tokens=None):
    client = _FreshAsyncOpenAI(**self._make_client_kwargs())
    try:
        kwargs = self._cons_kwargs(messages, timeout=self.get_timeout(timeout), max_tokens=max_tokens)
        rsp = await client.chat.completions.create(**kwargs)
        self._update_costs(rsp.usage)
        return rsp
    finally:
        try:
            await client.close()
        except Exception:
            pass
OpenAILLM._achat_completion = _loop_safe_achat

# ---------------- 加固 2b：每次 LLM 调用硬超时（第二道防线） ----------------
from Core.Provider.BaseLLM import BaseLLM
_orig_aask = BaseLLM.aask
PER_CALL_TIMEOUT = 90

async def _guarded_aask(self, *a, **k):
    return await asyncio.wait_for(_orig_aask(self, *a, **k), timeout=PER_CALL_TIMEOUT)
BaseLLM.aask = _guarded_aask

ENV_FILE = Path("D:/WSPforWB/10.keyan/.env_e14")
env = {}
for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    env[k.strip()] = v.strip().strip('"').strip("'")
KEY, BASE, MODEL = env["E14_LLM_API_KEY"], env["E14_LLM_BASE_URL"], env["E14_LLM_MODEL"]


# ---------------- 加固 3：进程级看门狗 ----------------
def start_watchdog(seconds: int):
    def _kill():
        sys.stderr.write(f"\n[WATCHDOG] exceeded {seconds}s, force-exit (os._exit(3))\n")
        sys.stderr.flush()
        os._exit(3)
    t = threading.Timer(seconds, _kill)
    t.daemon = True
    t.start()
    return t


def leaf_count(digimon):
    try:
        return len(digimon.graph._graph.leaf_nodes)
    except Exception:
        return -1


def internal_count(digimon):
    try:
        g = digimon.graph._graph
        return g.num_nodes - len(g.leaf_nodes)
    except Exception:
        return -1


def token_sums(digimon):
    try:
        cm = digimon.llm.cost_manager
        if cm is None:
            return 0, 0, 0
        c = cm.get_costs()
        pp = getattr(c, "total_prompt_tokens", 0) or 0
        cp = getattr(c, "total_completion_tokens", 0) or 0
        return pp, cp, pp + cp
    except Exception:
        return 0, 0, 0


def build_config(work_dir, data_dir, exp, add, force):
    method = Config.read_yaml(ROOT / "_repo" / "Option" / "Method" / "Dynamic.yaml")
    final = dict(method)
    final["llm"] = {
        "api_type": "open_llm", "model": MODEL, "base_url": BASE,
        "api_key": KEY, "api_version": None,
        "timeout": PER_CALL_TIMEOUT, "max_concurrent": 64,
    }
    MINILM = str(next((ROOT / "_hf_cache").rglob("model.safetensors")).parent)
    HF_CACHE = str(ROOT / "_hf_cache")
    final["embedding"] = {
        "api_type": "hf", "model": MINILM, "cache_dir": HF_CACHE,
        "dimensions": 384, "embed_batch_size": 16, "embedding_func_max_async": 8,
    }
    final["working_dir"] = work_dir
    final["exp_name"] = exp
    final["data_root"] = data_dir
    final["graph"] = dict(method.get("graph", {}))
    final["graph"]["add"] = add
    final["graph"]["force"] = force
    return Config(**final)


async def run_delta(N_total, N0, delta, workdir, datadir, tweets):
    exp = f"w3c_d{delta}"
    docs0 = [to_erarag_doc(t) for t in tweets[:N0]]
    cfg = build_config(workdir, datadir, exp, add=False, force=True)
    digi = GraphRAG(config=cfg)

    t0 = time.time()
    await digi.insert(docs0)
    dt0 = time.time() - t0
    pp, cp, tt = token_sums(digi)
    out = {
        "N_total": N_total, "N0": N0, "delta": delta,
        "phase0": {"docs": N0, "leaves": leaf_count(digi), "internal": internal_count(digi),
                   "wall_s": round(dt0, 2), "prompt_tokens": pp, "completion_tokens": cp, "total_tokens": tt},
    }

    rest = tweets[N0:]
    batches = []
    cum_tok = tt
    cum_wall = dt0
    idx = 0
    for i in range(0, len(rest), delta):
        chunk_t = rest[i:i + delta]
        idx += 1
        tmp_ws = Workspace(str(Path(workdir) / f"_c_b{idx}"), f"{exp}_b{idx}")
        tmp_chunk = DocChunk(cfg.chunk, digi.ENCODER, tmp_ws.make_for("chunk_storage"))
        await tmp_chunk.build_chunks([to_erarag_doc(t) for t in chunk_t])
        tuples = await tmp_chunk.get_chunks()

        t0 = time.time()
        await digi.graph.build_graph(tuples, force=False, add=True)
        dt = time.time() - t0

        pp, cp, tt2 = token_sums(digi)
        batch_tok = tt2 - cum_tok
        cum_wall += dt
        cum_tok = tt2
        leaves, internal = leaf_count(digi), internal_count(digi)
        batches.append({
            "b": idx, "n": len(chunk_t), "cum_docs": N0 + (idx * delta if idx * delta <= len(rest) else len(rest)),
            "leaves": leaves, "internal": internal,
            "wall_s": round(dt, 2), "batch_tokens": batch_tok,
            "batch_tokens_per_doc": round(batch_tok / max(len(chunk_t), 1), 2),
            "cum_wall_s": round(cum_wall, 2), "cum_tokens": cum_tok,
        })
        print(f"  batch{idx:>2} n={len(chunk_t):>3} leaves={leaves:>4} internal={internal:>3} "
              f"wall={dt:>6.2f}s tok={batch_tok:>7}")

    out["batches"] = batches
    out["summary"] = {
        "batch_count": len(batches),
        "final_leaves": leaf_count(digi),
        "final_internal": internal_count(digi),
        "total_wall_s": round(cum_wall, 2),
        "total_tokens": cum_tok,
        "increment_tokens": cum_tok - out["phase0"]["total_tokens"],
        "increment_wall_s": round(cum_wall - dt0, 2),
        "tokens_per_doc": round(cum_tok / N_total, 2),
        "wall_per_doc": round(cum_wall / N_total, 3),
        "avg_batch_tokens_per_doc": round(
            (cum_tok - out["phase0"]["total_tokens"]) / max(len(rest), 1), 2),
    }
    return out, digi


async def run_full(N_total, workdir, datadir, tweets):
    exp = f"w3c_full_{N_total}"
    docs = [to_erarag_doc(t) for t in tweets[:N_total]]
    cfg = build_config(workdir, datadir, exp, add=False, force=True)
    digi = GraphRAG(config=cfg)
    t0 = time.time()
    await digi.insert(docs)
    dt = time.time() - t0
    pp, cp, tt = token_sums(digi)
    return {"docs": N_total, "leaves": leaf_count(digi), "internal": internal_count(digi),
            "wall_s": round(dt, 2), "prompt_tokens": pp, "completion_tokens": cp, "total_tokens": tt}


async def mode_init(N_total, N0, delta, workdir, datadir, tweets):
    """phase0：首批 N0 条 force 建树，写 state JSON（供 batch/finalize 模式接续）。"""
    exp = f"w3c_d{delta}"
    docs0 = [to_erarag_doc(t) for t in tweets[:N0]]
    cfg = build_config(workdir, datadir, exp, add=False, force=True)
    digi = GraphRAG(config=cfg)
    t0 = time.time()
    await digi.insert(docs0)
    dt0 = time.time() - t0
    pp, cp, tt = token_sums(digi)
    out = {
        "N_total": N_total, "N0": N0, "delta": delta, "mode": "per_batch_process",
        "phase0": {"docs": N0, "leaves": leaf_count(digi), "internal": internal_count(digi),
                   "wall_s": round(dt0, 2), "prompt_tokens": pp, "completion_tokens": cp, "total_tokens": tt},
    }
    p = ROOT / f"W3c_delta{delta}_state.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[init] phase0 leaves={out['phase0']['leaves']} wall={dt0:.2f}s tok={tt} -> {p.name}")


async def mode_batch(N_total, N0, delta, k, workdir, datadir, tweets):
    """第 k 批：独立进程 load 磁盘树 → refine 该批 → persist → 写批 JSON。
    等价于流式服务的「持久化-恢复」循环；每批 aux 状态全新。"""
    exp = f"w3c_d{delta}"
    rest = tweets[N0:]
    i = (k - 1) * delta
    chunk_t = rest[i:i + delta]
    assert chunk_t, f"batch {k} out of range"

    cfg = build_config(workdir, datadir, exp, add=True, force=False)
    digi = GraphRAG(config=cfg)
    tmp_ws = Workspace(str(Path(workdir) / f"_c_b{k}"), f"{exp}_b{k}")
    tmp_chunk = DocChunk(cfg.chunk, digi.ENCODER, tmp_ws.make_for("chunk_storage"))
    await tmp_chunk.build_chunks([to_erarag_doc(t) for t in chunk_t])
    tuples = await tmp_chunk.get_chunks()

    t0 = time.time()
    await digi.graph.build_graph(tuples, force=False, add=True)
    dt = time.time() - t0
    pp, cp, tt = token_sums(digi)
    out = {
        "N_total": N_total, "N0": N0, "delta": delta, "b": k, "n": len(chunk_t),
        "leaves": leaf_count(digi), "internal": internal_count(digi),
        "wall_s": round(dt, 2), "batch_tokens": tt,
        "batch_tokens_per_doc": round(tt / max(len(chunk_t), 1), 2),
    }
    p = ROOT / f"W3c_delta{delta}_b{k}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[batch{k}] n={out['n']} leaves={out['leaves']} internal={out['internal']} "
          f"wall={out['wall_s']}s tok={out['batch_tokens']}")


def mode_finalize(N_total, N0, delta):
    """聚合 state + 批 JSON → W3c_delta{D}.json（与单进程模式字段兼容）。"""
    state = json.loads((ROOT / f"W3c_delta{delta}_state.json").read_text(encoding="utf-8"))
    k = 1
    batches = []
    while True:
        p = ROOT / f"W3c_delta{delta}_b{k}.json"
        if not p.exists():
            break
        batches.append(json.loads(p.read_text(encoding="utf-8")))
        k += 1
    cum_tok = state["phase0"]["total_tokens"]
    cum_wall = state["phase0"]["wall_s"]
    for b in batches:
        cum_tok += b["batch_tokens"]
        cum_wall += b["wall_s"]
        b["cum_tokens"] = cum_tok
        b["cum_wall_s"] = round(cum_wall, 2)
    out = dict(state)
    out["batches"] = batches
    last = batches[-1] if batches else state["phase0"]
    out["summary"] = {
        "batch_count": len(batches),
        "final_leaves": last["leaves"], "final_internal": last["internal"],
        "total_wall_s": round(cum_wall, 2), "total_tokens": cum_tok,
        "increment_tokens": cum_tok - state["phase0"]["total_tokens"],
        "increment_wall_s": round(cum_wall - state["phase0"]["wall_s"], 2),
        "tokens_per_doc": round(cum_tok / N_total, 2),
        "wall_per_doc": round(cum_wall / N_total, 3),
    }
    full_p = ROOT / "W3c_full_200.json"
    if full_p.exists():
        out["full_rebuild"] = json.loads(full_p.read_text(encoding="utf-8"))
    p_out = ROOT / f"W3c_delta{delta}.json"
    p_out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    s = out["summary"]
    print(f"  ★ Δ={delta}: batches={s['batch_count']} final_leaves={s['final_leaves']} "
          f"final_internal={s['final_internal']} wall={s['total_wall_s']}s tokens={s['total_tokens']}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N_total", type=int, default=200)
    ap.add_argument("--N0", type=int, default=40)
    ap.add_argument("--delta", type=int, required=True)
    ap.add_argument("--mode", choices=["single", "init", "batch", "finalize"], default="single")
    ap.add_argument("--k", type=int, default=None, help="batch 模式的批次号（1-based）")
    ap.add_argument("--suffix", type=str, default="", help="workdir 后缀（绕开旧目录残留）")
    ap.add_argument("--with_full", action="store_true")
    ap.add_argument("--watchdog", type=int, default=1800)
    args = ap.parse_args()

    start_watchdog(args.watchdog)
    N_total, N0, delta = args.N_total, args.N0, args.delta

    pa = PhemeAdapter().load()
    tweets = []
    for t in pa.iter_tweets(labeled_only=True, min_text_len=10):
        tweets.append(t)
        if len(tweets) >= N_total:
            break
    tweets = sorted(tweets, key=lambda x: x["ms"])
    workdir = str(Path(ASCII_BASE) / f"_w3c_d{N_total}_{delta}{args.suffix}")
    datadir = str(Path(ASCII_BASE) / f"_w3c_d{N_total}_{delta}{args.suffix}_data")

    if args.mode == "init":
        if Path(workdir).exists():
            shutil.rmtree(workdir, ignore_errors=True)
        await mode_init(N_total, N0, delta, workdir, datadir, tweets)
        return
    if args.mode == "batch":
        await mode_batch(N_total, N0, delta, args.k, workdir, datadir, tweets)
        return
    if args.mode == "finalize":
        mode_finalize(N_total, N0, delta)
        return

    # ---- 原单进程模式（保留兼容） ----
    print(f"[准备] N_total={N_total} N0={N0} delta={delta} "
          f"batches={-(-(N_total-N0)//delta)} pool={len(tweets)}")
    if Path(workdir).exists():
        shutil.rmtree(workdir)

    out, _ = await run_delta(N_total, N0, delta, workdir, datadir, tweets)
    s = out["summary"]
    print(f"  ★ Δ={delta}: batches={s['batch_count']} final_leaves={s['final_leaves']} "
          f"final_internal={s['final_internal']} wall={s['total_wall_s']}s tokens={s['total_tokens']}")

    if args.with_full:
        wf = str(Path(ASCII_BASE) / f"_w3c_full_{N_total}")
        df = str(Path(ASCII_BASE) / f"_w3c_full_{N_total}_data")
        if Path(wf).exists():
            shutil.rmtree(wf)
        full = await run_full(N_total, wf, df, tweets)
        out["full_rebuild"] = full
        print(f"  FullRebuild(N={N_total}): leaves={full['leaves']} internal={full['internal']} "
              f"wall={full['wall_s']}s tokens={full['total_tokens']}")

    out_path = ROOT / f"W3c_delta{delta}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[报告] 写入 {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
