"""E14 fail-loud stub for `umap` (RAPTOR/静态树用).

只满足 `Core/Graph/TreeGraph.py` 的顶层导入；动态树走自定义超平面 LSH。
"""


class UMAP:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "E14 stub: UMAP is not used by the tree_graph_dynamic path (custom hyperplane LSH instead)."
        )
