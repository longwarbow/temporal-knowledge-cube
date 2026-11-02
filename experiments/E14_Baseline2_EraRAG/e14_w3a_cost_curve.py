# -*- coding: utf-8 -*-
"""
E14 / W3a 成本曲线（v2 —— 带超时/看门狗加固版）

背景：v1 在 N=200 时挂死 3.7h（无网络活动、CPU 近 0）——疑似 EraRAG 的
      ThreadPoolExecutor + 每任务新建事件循环 与 主循环创建的 AsyncOpenAI
      客户端 / asyncio.Semaphore 发生跨 loop 死锁。v2 三重加固：
        1) 每次 LLM 调用外包 asyncio.wait_for(per_call_timeout)  —— 硬超时
        2) 用 loop-无关的 no-op 信号量替换 asyncio.Semaphore —— 消除跨 loop future
        3) 进程级看门狗线程 —— 超时直接 os._exit，绝不挂死
      并改为「单规模、独立进程」运行，结果增量落盘。

用法：
  python e14_w3a_cost_curve.py --scale 50            # 只跑 N=50
  python e14_w3a_cost_curve.py --scale 50 --with_full
"""

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
import os, sys, json, time, asyncio, argparse, shutil, threading
from pathlib import Path

ROOT = Path(_KB('experiments', 'E14_Baseline2_EraRAG')).resolve()
ASCII_BASE = _KB('_e14_run')

# ---------------- CPU 强制（同 W1/W2） ----------------
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
    self.semaphore = _NoopSemaphore()   # 消除 asyncio.Semaphore 的跨 loop 绑定
OpenAILLM.__init__ = _patched_openai_init

# ---------------- 加固 2：每次 LLM 调用硬超时 ----------------
from Core.Provider.BaseLLM import BaseLLM
_orig_aask = BaseLLM.aask
PER_CALL_TIMEOUT = 90   # 秒

async def _guarded_aask(self, *a, **k):
    return await asyncio.wait_for(_orig_aask(self, *a, **k), timeout=PER_CALL_TIMEOUT)
BaseLLM.aask = _guarded_aask

# 读 key
ENV_FILE = Path(_KB('.env_e14'))
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
        msg = f"\n[WATCHDOG] exceeded {seconds}s, force-exit (os._exit(3))\n"
        sys.stderr.write(msg); sys.stderr.flush()
        os._exit(3)
    t = threading.Timer(seconds, _kill)
    t.daemon = True
    t.start()
    return t


def leaf_count(digimon):
    g = digimon.graph._graph
    try:
        return len(g.leaf_nodes)
    except Exception:
        return -1


def internal_count(digimon):
    g = digimon.graph._graph
    try:
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


async def run_one_scale(N, M, workdir, datadir, tweets_M):
    out = {"N": N, "M": M}
    pa = PhemeAdapter().load()
    tweets = []
    for t in pa.iter_tweets(labeled_only=True, min_text_len=10):
        tweets.append(t)
        if len(tweets) >= N + M:
            break
    tweets = sorted(tweets, key=lambda x: x["ms"])
    docs_N = [to_erarag_doc(t) for t in tweets[:N]]

    cfgA = build_config(workdir, datadir, f"w3a_N{N}", add=False, force=True)
    digiA = GraphRAG(config=cfgA)
    t0 = time.time()
    await digiA.insert(docs_N)
    dtA = time.time() - t0
    pp, cp, tt = token_sums(digiA)
    out["phaseA"] = {"docs": N, "leaves": leaf_count(digiA), "internal": internal_count(digiA),
                     "wall_s": round(dtA, 2), "prompt_tokens": pp, "completion_tokens": cp, "total_tokens": tt}

    cfgB = build_config(workdir, datadir, f"w3a_N{N}", add=True, force=False)
    digiB = GraphRAG(config=cfgB)
    tmp_ws = Workspace(str(Path(workdir) / "_b_tmp"), f"w3a_N{N}_b")
    tmp_chunk = DocChunk(cfgB.chunk, digiB.ENCODER, tmp_ws.make_for("chunk_storage"))
    await tmp_chunk.build_chunks([to_erarag_doc(t) for t in tweets_M])
    new_tuples = await tmp_chunk.get_chunks()
    t0 = time.time()
    await digiB.graph.build_graph(new_tuples, force=False, add=True)
    dtB = time.time() - t0
    pp, cp, tt = token_sums(digiB)
    out["phaseB"] = {"new_chunks": M, "leaves_after": leaf_count(digiB), "internal_after": internal_count(digiB),
                     "wall_s": round(dtB, 2), "prompt_tokens": pp, "completion_tokens": cp, "total_tokens": tt}
    return out


async def run_full_rebuild(N, M, workdir, datadir, all_tweets):
    docs_all = [to_erarag_doc(t) for t in all_tweets]
    cfg = build_config(workdir, datadir, f"w3a_full_{N+M}", add=False, force=True)
    digi = GraphRAG(config=cfg)
    t0 = time.time()
    await digi.insert(docs_all)
    dt = time.time() - t0
    pp, cp, tt = token_sums(digi)
    return {"docs": N + M, "leaves": leaf_count(digi), "internal": internal_count(digi),
            "wall_s": round(dt, 2), "prompt_tokens": pp, "completion_tokens": cp, "total_tokens": tt}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--M_frac", type=float, default=0.2)
    ap.add_argument("--with_full", action="store_true")
    ap.add_argument("--watchdog", type=int, default=1200, help="进程级看门狗秒数")
    args = ap.parse_args()

    start_watchdog(args.watchdog)
    N = args.scale
    M = int(N * args.M_frac)

    pa = PhemeAdapter().load()
    all_tweets = []
    for t in pa.iter_tweets(labeled_only=True, min_text_len=10):
        all_tweets.append(t)
        if len(all_tweets) >= N + M:
            break
    all_tweets = sorted(all_tweets, key=lambda x: x["ms"])
    print(f"[准备] N={N} M={M} pool={len(all_tweets)}")

    workdir = str(Path(ASCII_BASE) / f"_w3a_N{N}")
    datadir = str(Path(ASCII_BASE) / f"_w3a_N{N}_data")
    if Path(workdir).exists():
        shutil.rmtree(workdir)

    out = await run_one_scale(N, M, workdir, datadir, all_tweets[N:N + M])
    print(f"  PhaseA: docs={out['phaseA']['docs']} leaves={out['phaseA']['leaves']} "
          f"wall={out['phaseA']['wall_s']}s tokens={out['phaseA']['total_tokens']}")
    print(f"  PhaseB: new={out['phaseB']['new_chunks']} leaves={out['phaseB']['leaves_after']} "
          f"wall={out['phaseB']['wall_s']}s tokens={out['phaseB']['total_tokens']}")

    if args.with_full:
        wf = str(Path(ASCII_BASE) / f"_w3a_full_{N+M}")
        df = str(Path(ASCII_BASE) / f"_w3a_full_{N+M}_data")
        if Path(wf).exists():
            shutil.rmtree(wf)
        full = await run_full_rebuild(N, M, wf, df, all_tweets[:N + M])
        out["full_rebuild_NpM"] = full
        ft = full["total_tokens"] or 1
        out["incremental_value"] = {
            "phaseB_tokens_over_full": round(out["phaseB"]["total_tokens"] / ft, 3),
            "phaseB_sec_over_full": round(out["phaseB"]["wall_s"] / (full["wall_s"] or 1), 3),
        }
        print(f"  FullRebuild(N+M={N+M}): leaves={full['leaves']} wall={full['wall_s']}s tokens={full['total_tokens']}")
        print(f"  ★ Incremental value: token_ratio={out['incremental_value']['phaseB_tokens_over_full']} "
              f"sec_ratio={out['incremental_value']['phaseB_sec_over_full']}")

    # 增量落盘
    out_path = ROOT / f"W3a_scale{N}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[报告] 写入 {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
