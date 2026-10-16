"""E14 fail-loud stub for `colbert`.

只满足 `Core/Index/ColBertIndex.py` 的顶层导入；我们的实验使用 `vdb_type: faiss`。
一旦被真正实例化，立即抛错（绝不静默返回假值）。
"""


class _Stub:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "E14 stub: ColBERT path is not exercised in this experiment (vdb_type=faiss)."
        )


Indexer = _Stub
Searcher = _Stub
