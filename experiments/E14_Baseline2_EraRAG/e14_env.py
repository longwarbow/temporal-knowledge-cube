"""E14 运行环境引导（CPU-only / Windows / Python 3.9）。

作用：把「官方仓库目录」与「fail-loud 依赖桩目录」前插到 `sys.path`，
使 `EraRAG-Official` 的 `tree_graph_dynamic`（建树 + 增量插入）与 `faiss` 检索
可以在**不安装** ColBERT / PCST / graspologic / UMAP 等未使用重依赖的前提下导入。

用法：
    import e14_env; e14_env.bootstrap()
    import Core.Graph.TreeGraphDynamic  # now importable
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

E14_DIR = Path(__file__).resolve().parent
REPO_DIR = E14_DIR / "_repo"
SHIMS_DIR = E14_DIR / "e14_shims"


def bootstrap(verbose: bool = True) -> dict:
    for p in (str(SHIMS_DIR), str(REPO_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("E14_REPO_DIR", str(REPO_DIR))

    info = {
        "python": sys.version.split()[0],
        "repo_dir": str(REPO_DIR),
        "shims_dir": str(SHIMS_DIR),
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
    }
    if verbose:
        print("[e14_env] python=%s" % info["python"])
        print("[e14_env] repo =%s" % info["repo_dir"])
        print("[e14_env] shims=%s" % info["shims_dir"])
    return info


def probe_import() -> list:
    """返回 (模块名, 是否可导入, 错误) 列表，用于记录环境指纹。"""
    import importlib

    mods = [
        "Core.GraphRAG",
        "Core.Graph.TreeGraphDynamic",
        "Core.Index.FaissIndex",
        "Core.Query.BasicQuery",
        "llama_index.core",
        "llama_index.embeddings.huggingface",
        "faiss",
        "torch",
        "sentence_transformers",
        "openai",
    ]
    out = []
    for m in mods:
        try:
            importlib.import_module(m)
            out.append((m, True, ""))
        except Exception as e:  # noqa: BLE001
            out.append((m, False, "%s: %s" % (type(e).__name__, e)))
    return out


if __name__ == "__main__":
    bootstrap()
    for name, ok, err in probe_import():
        print(("  OK  " if ok else " FAIL ") + name + ("" if ok else "  <- " + err))
