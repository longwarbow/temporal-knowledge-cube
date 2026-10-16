# -*- coding: utf-8 -*-
"""
E14 / W2 冒烟：PHEME → EraRAG chunk 流适配器

两个模式：
- --mode adapter  纯适配器验证：统计 + 格式 + 时间排序 + EraRAG doc 形态（快）
- --mode erarag   端到端：取前 N 条 PHEME 贴，喂给 EraRAG，验证 chunk 流与
                   chunk 摘要可被 insert 路径消化（沿用 W1 的 CPU/ASCII 保护）
"""
import os, sys, json, time, argparse
from pathlib import Path

# ----- 路径（必须 ASCII 给 faiss，详见 E14_repo_compat.md #8） -----
ROOT = Path("D:/WSPforWB/10.keyan/场景1-海量短文本/实验/E14_Baseline2_EraRAG").resolve()
ASCII_BASE = "D:/WSPforWB/10.keyan/_e14_run"
W2WORK = str(Path(ASCII_BASE) / "_w2work")
W2DATA = str(Path(ASCII_BASE) / "_w2data")

# 让 sys.path 能 import 适配器
sys.path.insert(0, str(ROOT))
from e14_adapter import (
    PhemeAdapter, CedAdapter, to_erarag_doc, batched,
    snowflake_to_ms, ms_to_iso,
)


def mode_adapter(args):
    print("=" * 70)
    print("E14 / W2 适配器冒烟（纯数据层）")
    print("=" * 70)
    pa = PhemeAdapter().load()
    print(f"\n[加载] cascades={pa._thread_count}, "
          f"veracity={len(pa.veracity)}")
    # 枚举分布
    from collections import Counter
    print(f"  veracity 枚举分布：{Counter(pa.veracity.values())}")

    # 线程级统计
    print("\n[线程级统计 labeled]")
    ths = pa.thread_stats(labeled_only=True)
    print(json.dumps(ths, ensure_ascii=False, indent=2))

    # 贴文级统计（限制 max_items 避免扫全 10 万）
    print("\n[贴文级统计 labeled, 前 %d 条]" % args.max_items)
    tws = pa.tweet_stats(labeled_only=True, max_items=args.max_items)
    print(json.dumps(tws, ensure_ascii=False, indent=2))

    # 取前 N 条
    print("\n[前 %d 条样本，含 snowflake/时间/标签]" % args.max_items)
    it = pa.iter_tweets(labeled_only=True, min_text_len=args.min_text_len)
    sample = []
    for i, t in enumerate(it):
        if i >= args.max_items:
            break
        sample.append(t)
    print(f"  实际取到 {len(sample)} 条")
    for t in sample[:3]:
        print(f"   - id={t['tweet_id']} ms={t['ms']} iso={t['iso']} "
              f"label={PhemeAdapter.LABEL_NAME.get(t['label'])} "
              f"src={t['is_source']} len={t['text_len']}")
        print(f"     text: {t['text'][:80]}")

    # 时间排序 + 头尾
    print("\n[时间排序后头/尾各 3 条]")
    sorted_t = sorted(sample, key=lambda x: x["ms"])
    print("  最早：")
    for t in sorted_t[:3]:
        print(f"   {t['iso']} | {t['text'][:60]}")
    print("  最晚：")
    for t in sorted_t[-3:]:
        print(f"   {t['iso']} | {t['text'][:60]}")

    # EraRAG doc 形态
    print("\n[to_erarag_doc 转换验证（前 2 条，纯文本 / 带前缀）]")
    for t in sample[:2]:
        d1 = to_erarag_doc(t, content_prefix_meta=False)
        d2 = to_erarag_doc(t, content_prefix_meta=True)
        print(f"  纯: title={d1['title']} content={d1['content'][:60]}")
        print(f"  带前缀: content={d2['content'][:90]}")

    # batched
    print(f"\n[batched 验证 (n={args.batch_size})]")
    it2 = pa.iter_tweets(labeled_only=True, min_text_len=args.min_text_len)
    batches = list(batched(it2, args.batch_size))
    if not batches:
        print("  (无数据)")
    else:
        # 只取前 3 批
        for i, b in enumerate(batches[:3]):
            ms_span = [t["ms"] for t in b]
            print(f"  batch#{i} size={len(b)} ms_min={ms_to_iso(min(ms_span))} "
                  f"ms_max={ms_to_iso(max(ms_span))}")
        # 总条数
        total = sum(len(b) for b in batches)
        print(f"  total batches={len(batches)} total_tweets={total}")

    # CED 占位
    print("\n[CED 占位]")
    ce = CedAdapter()
    print(f"  exists={ce.exists()} note={ce.note()[:60]}...")

    # 写报告
    out = {
        "thread_stats_labeled": ths,
        "tweet_stats_labeled_maxN": tws,
        "first_n_tweets": sample,
        "n_batches": len(batches) if batches else 0,
        "n_tweets_in_batches": sum(len(b) for b in batches) if batches else 0,
    }
    report = ROOT / "W2_适配器冒烟.json"
    report.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[报告] 写入 {report}")


def mode_erarag(args):
    """小批量 PHEME → EraRAG insert 端到端（沿用 W1 的 CPU/ASCII 保护）"""
    # ---- 与 W1 完全相同的 CPU 强制（防止重导） ----
    import torch  # noqa
    torch.backends.cuda.enabled = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import sentence_transformers  # noqa
    _orig_st = sentence_transformers.SentenceTransformer.__init__

    def _cpu_st(self, *a, **k):
        k["device"] = "cpu"
        return _orig_st(self, *a, **k)
    sentence_transformers.SentenceTransformer.__init__ = _cpu_st

    REPO = ROOT / "_repo"
    HF_CACHE = str(ROOT / "_hf_cache")
    MINILM = str(next((ROOT / "_hf_cache").rglob("model.safetensors")).parent)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE
    sys.path.insert(0, str(REPO))
    import e14_env  # noqa
    e14_env.bootstrap(verbose=False)
    from Option.Config2 import Config
    from Core.GraphRAG import GraphRAG

    # 读 key
    ENV_FILE = Path("D:/WSPforWB/10.keyan/.env_e14")
    env = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    KEY, BASE, MODEL = env["E14_LLM_API_KEY"], env["E14_LLM_BASE_URL"], env["E14_LLM_MODEL"]

    # 取前 N 条 PHEME 贴（按时间排序）→ EraRAG doc 流
    pa = PhemeAdapter().load()
    tweets = []
    for t in pa.iter_tweets(labeled_only=True, min_text_len=args.min_text_len):
        tweets.append(t)
        if len(tweets) >= args.max_items:
            break
    tweets = sorted(tweets, key=lambda x: x["ms"])
    docs = [to_erarag_doc(t, content_prefix_meta=False) for t in tweets]
    print(f"[erarag] 准备插入 {len(docs)} 条 PHEME 贴（时间排序）")

    # 构 Config
    method = Config.read_yaml(REPO / "Option" / "Method" / "Dynamic.yaml")
    final = dict(method)
    final["llm"] = {
        "api_type": "open_llm", "model": MODEL, "base_url": BASE, "api_key": KEY, "api_version": None,
    }
    final["embedding"] = {
        "api_type": "hf", "model": MINILM, "cache_dir": HF_CACHE,
        "dimensions": 384, "embed_batch_size": 16, "embedding_func_max_async": 8,
    }
    final["working_dir"] = W2WORK
    final["exp_name"] = "w2erarag"
    final["data_root"] = W2DATA
    final["graph"] = dict(method.get("graph", {}))
    final["graph"]["add"] = False
    final["graph"]["force"] = True
    cfg = Config(**final)

    import asyncio
    async def go():
        d = GraphRAG(config=cfg)
        t0 = time.time()
        await d.insert(docs)
        dt = time.time() - t0
        g = d.graph._graph
        try:
            leaves = len(g.leaf_nodes)
        except Exception:
            leaves = -1
        print(f"[erarag] insert 完成: leaves={leaves} seconds={dt:.1f}")
        return leaves, dt
    leaves, dt = asyncio.run(go())
    out = {"docs_in": len(docs), "leaves_out": leaves, "seconds": dt}
    (ROOT / "W2_erarag_小批.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["adapter", "erarag"], default="adapter")
    ap.add_argument("--max_items", type=int, default=200)
    ap.add_argument("--min_text_len", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    Path(ASCII_BASE).mkdir(parents=True, exist_ok=True)

    if args.mode == "adapter":
        mode_adapter(args)
    else:
        mode_erarag(args)


if __name__ == "__main__":
    main()
