# -*- coding: utf-8 -*-
"""
E14 / W2 适配器：PHEME (主) / CED (辅) → EraRAG 文档流

只读、惰性、不修改原数据文件。提供：
- snowflake → 毫秒时间轴反解
- PhemeAdapter.iter_tweets(...)   每条贴一条记录
- to_erarag_doc(tweet)            → {"title":<snowflake>,"content":<text>}
- batched(iter, n)                迷你批

约定：EraRAG 的 DocChunk 只识别 {title, content}，其余字段（meta）作为字典
      透传但仅在 chunk 节点的 metadata 上保留一份轻量索引。

设计原则（与论文方法论对齐）：
- 只读原数据，绝不修改
- 标签枚举：0=true(非谣) / 1=false(谣) / 2=unverified（来自 veracity_dict.pkl 分布）
- 过滤策略（默认）：仅保留有标签线程里的、有文本的、且能找到源推的贴
"""
from __future__ import annotations
import os, json, pickle, time
from pathlib import Path
from collections import Counter
from typing import Iterator, Iterable, Optional, Tuple, List, Dict, Any

# Twitter snowflake epoch (2010-11-04 01:42:54.657 UTC, in ms)
SNOWFLAKE_EPOCH_MS = 1288834974657


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def snowflake_to_ms(sid: str) -> int:
    """Twitter snowflake id → 毫秒时间戳（UTC）。"""
    return (int(sid) >> 22) + SNOWFLAKE_EPOCH_MS


def ms_to_iso(ms: int) -> str:
    """毫秒时间戳 → ISO8601 (UTC, 秒精度)。"""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ms / 1000.0))


def batched(iterable: Iterable, n: int) -> Iterator[list]:
    """把迭代器切成 n 大小的小批。"""
    batch: list = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# PHEME 适配器（主载体）
# ---------------------------------------------------------------------------
class PhemeAdapter:
    """PHEME DiGraph list → (tweet, meta) 流。"""

    # veracity 枚举（来自 _provenance + 1972 线程实测分布）
    LABEL_TRUE = 0
    LABEL_FALSE = 1
    LABEL_UNVERIFIED = 2
    LABEL_NAME = {0: "true", 1: "false", 2: "unverified"}

    def __init__(self, root: str = "D:/WSPforWB/10.keyan/场景1-海量短文本/数据/PHEME"):
        self.root = Path(root)
        self._cascades = None
        self._veracity = None
        self._thread_count = 0

    # ---------- 加载 ----------
    def load(self) -> "PhemeAdapter":
        cas_path = self.root / "pheme_cascades.gpickle"
        ver_path = self.root / "veracity_dict.pkl"
        with open(cas_path, "rb") as f:
            self._cascades = pickle.load(f)
        with open(ver_path, "rb") as f:
            self._veracity = pickle.load(f)
        self._thread_count = len(self._cascades)
        return self

    @property
    def cascades(self):
        if self._cascades is None:
            self.load()
        return self._cascades

    @property
    def veracity(self) -> Dict[str, int]:
        if self._veracity is None:
            self.load()
        return self._veracity

    # ---------- 线程级 ----------
    def iter_threads(self, labeled_only: bool = True) -> Iterator[Dict]:
        """每个线程一条记录：thread_idx, source_id, label, n_nodes, n_edges, n_text,
        ms_min/max, iso_min/max。"""
        ver = self.veracity
        for ci, g in enumerate(self.cascades):
            src = None
            for nid, a in g.nodes(data=True):
                if a.get("is_source"):
                    src = nid
                    break
            if labeled_only and src not in ver:
                continue
            tids = [nid for nid, a in g.nodes(data=True) if (a.get("text") or "")]
            if not tids:
                continue
            ms_list = [snowflake_to_ms(t) for t in tids]
            yield {
                "thread_idx": ci,
                "source_id": src,
                "label": ver.get(src),
                "n_nodes": g.number_of_nodes(),
                "n_edges": g.number_of_edges(),
                "n_text": len(tids),
                "ms_min": min(ms_list),
                "ms_max": max(ms_list),
                "iso_min": ms_to_iso(min(ms_list)),
                "iso_max": ms_to_iso(max(ms_list)),
            }

    # ---------- 贴文级 ----------
    def iter_tweets(
        self,
        labeled_only: bool = True,
        min_text_len: int = 1,
        require_source: bool = True,
        include_replies: bool = True,
    ) -> Iterator[Dict]:
        """每条贴一条记录（含源贴与回复）。
        字段：tweet_id, text, ms, iso, source_id, thread_idx, is_source, label, text_len。
        """
        ver = self.veracity
        valid_sources = set(ver.keys()) if labeled_only else None
        for ci, g in enumerate(self.cascades):
            src_id = None
            for nid, a in g.nodes(data=True):
                if a.get("is_source"):
                    src_id = nid
                    break
            if require_source and src_id is None:
                continue
            if labeled_only and (valid_sources is not None and src_id not in valid_sources):
                continue
            label = ver.get(src_id) if src_id else None
            for nid, a in g.nodes(data=True):
                if not include_replies and not a.get("is_source"):
                    continue
                t = a.get("text") or ""
                if len(t) < min_text_len:
                    continue
                ms = snowflake_to_ms(nid)
                yield {
                    "tweet_id": nid,
                    "text": t,
                    "ms": ms,
                    "iso": ms_to_iso(ms),
                    "source_id": src_id,
                    "thread_idx": ci,
                    "is_source": bool(a.get("is_source")),
                    "label": label,
                    "text_len": len(t),
                }

    # ---------- 统计 ----------
    def thread_stats(self, labeled_only: bool = True) -> Dict:
        c_label = Counter()
        c_nodes = []
        c_text = []
        for th in self.iter_threads(labeled_only=labeled_only):
            c_label[th["label"]] += 1
            c_nodes.append(th["n_nodes"])
            c_text.append(th["n_text"])
        return {
            "n_threads": sum(c_label.values()),
            "label_dist": {self.LABEL_NAME.get(k, str(k)): v for k, v in c_label.items()},
            "nodes_per_thread": {
                "min": min(c_nodes) if c_nodes else 0,
                "max": max(c_nodes) if c_nodes else 0,
                "mean": round(sum(c_nodes) / len(c_nodes), 2) if c_nodes else 0,
            },
            "text_per_thread": {
                "min": min(c_text) if c_text else 0,
                "max": max(c_text) if c_text else 0,
                "mean": round(sum(c_text) / len(c_text), 2) if c_text else 0,
            },
        }

    def tweet_stats(
        self,
        labeled_only: bool = True,
        min_text_len: int = 1,
        max_items: Optional[int] = None,
    ) -> Dict:
        c_label = Counter()
        c_len = []
        c_src = Counter()
        n = 0
        ms_first = None
        ms_last = None
        for tw in self.iter_tweets(labeled_only=labeled_only, min_text_len=min_text_len):
            c_label[tw["label"]] += 1
            c_len.append(tw["text_len"])
            c_src[tw["is_source"]] += 1
            if ms_first is None or tw["ms"] < ms_first:
                ms_first = tw["ms"]
            if ms_last is None or tw["ms"] > ms_last:
                ms_last = tw["ms"]
            n += 1
            if max_items and n >= max_items:
                break
        return {
            "n_tweets": n,
            "label_dist": {self.LABEL_NAME.get(k, str(k)): v for k, v in c_label.items()},
            "text_len": {
                "min": min(c_len) if c_len else 0,
                "max": max(c_len) if c_len else 0,
                "mean": round(sum(c_len) / len(c_len), 2) if c_len else 0,
                "median": sorted(c_len)[len(c_len) // 2] if c_len else 0,
            },
            "is_source": {str(k): v for k, v in c_src.items()},
            "ms_span": {
                "first_ms": ms_first,
                "last_ms": ms_last,
                "first_iso": ms_to_iso(ms_first) if ms_first else None,
                "last_iso": ms_to_iso(ms_last) if ms_last else None,
            },
        }


# ---------------------------------------------------------------------------
# 转换：tweet dict → EraRAG 文档
# ---------------------------------------------------------------------------
def to_erarag_doc(t: Dict, content_prefix_meta: bool = False) -> Dict:
    """把 tweet 字典转成 EraRAG 期望的 {title, content} 形态。
    title 用 snowflake (全局唯一 id)，content 用原始文本。
    若 content_prefix_meta=True，则把 meta 信息拼到 content 前面，便于检索词
    捕获事件/标签上下文；否则纯文本。
    """
    if content_prefix_meta:
        prefix = f"[{t['iso']}][label={PhemeAdapter.LABEL_NAME.get(t['label'], 'NA')}] "
        return {"title": t["tweet_id"], "content": prefix + t["text"]}
    return {"title": t["tweet_id"], "content": t["text"]}


# ---------------------------------------------------------------------------
# CED 适配器（占位，CED 转发文本多为空，主要作到达流，不直接进 EraRAG 索引）
# ---------------------------------------------------------------------------
class CedAdapter:
    """CED 转发流以"到达流"为主（计数即可），本类先提供结构占位。"""

    def __init__(self, root: str = "D:/WSPforWB/10.keyan/场景1-海量短文本/数据/CED"):
        self.root = Path(root)

    def exists(self) -> bool:
        return self.root.exists()

    def note(self) -> str:
        return (
            "CED 转发文本多为空，作到达流；EraRAG 适配以 PHEME 为主载体。"
            "如需 CED 文本接入，请在此实现 iter_tweets；当前阶段跳过。"
        )
