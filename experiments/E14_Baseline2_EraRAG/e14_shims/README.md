# E14 fail-loud 依赖桩（shims）

**目的**：让 `EraRAG-Official` 在 **CPU-only / Windows / Python 3.9** 上跑通它**真正被我们使用的那条路径**
（`tree_graph_dynamic` 建树 + 增量插入 + `faiss` 检索 + `basic` 查询），而**不必安装**它库里
那些只在**未被使用的其他方法**（ColBERT 索引、HippoRAG 的 PCST、Leiden 社区、RAPTOR 的 UMAP、
论文评测脚本的 MAUVE/ROUGE）里才用到的重依赖。

**纪律**：
1. 每个桩都是 **fail-loud** —— 一旦被真正调用就 `NotImplementedError`，**绝不静默返回假值**。
2. 桩只覆盖 `import` 期（顶层 `from X import Y`）的需求，不改变任何被使用路径的语义。
3. 官方仓库代码 **除 `Core/Common/Utils.py` 一处 PEP604 兼容补丁外未做任何改动**（见 `../E14_repo_compat.patch`）。

| 桩 | 为什么需要 | 是否会被真正调用 |
|---|---|---|
| `colbert` | `Core/Index/ColBertIndex.py` 顶层 `from colbert import Indexer, Searcher` | ❌ 我们用 `vdb_type: faiss` |
| `pcst_fast` | `Core/Query/GRQuery.py` 顶层 `from pcst_fast import pcst_fast` | ❌ 我们用 `query_type: basic` |
| `graspologic` | `Core/Community/LeidenCommunity.py` 顶层 `hierarchical_leiden` | ❌ 动态树不用社区检测（`use_community` 未开） |
| `umap` | `Core/Graph/TreeGraph.py` 顶层 `from umap import UMAP` | ❌ 动态树走自定义超平面 LSH，不用 UMAP |
| `mauve` / `rouge_score` | `Core/Utils/Evaluation.py` 顶层导入 | ❌ 我们不做 QA 评测（只做维护代价 + 能力矩阵） |

> 安装方式：运行时把 `e14_shims/` **前插**到 `sys.path`（见 `e14_env.py`），
> 因此不必、也**不要**污染官方仓库目录。
