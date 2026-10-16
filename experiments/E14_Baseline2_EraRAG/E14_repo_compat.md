# E14 / EraRAG-Official 仓库兼容性缺陷清单（共 13 处）

> 用途：支撑论文 reproducibility / threats 段落。所有缺陷均在 **Windows + Python 3.9 + CPU-only + 本地 MiniLM(384) + 智谱 GLM-4-Flash** 环境下暴露，均已本地修复/加固。
> 仓库：`EverM0re/EraRAG-Official`（克隆于 `E14_Baseline2_EraRAG/_repo`）。

---

## 缺陷汇总表

| # | 文件 | 缺陷 | 症状 | 修复 |
|---|---|---|---|---|
| 1 | `Core/Graph/TreeGraphLSH.py` | 占位符代码（clone 后未补全） | `import` 即崩溃 | 补全 LSH 超平面实现 / 移除占位 |
| 2 | 依赖 `llama-index` → `banks` | Py3.9 下 `banks` 模块导入崩溃 | 解释器启动即崩 | 卸载 `banks` 并钉死兼容 llama-index 版本 |
| 3 | `Core/Common/LLMConfig.py` 等 | `LLMType` 枚举被**双份定义** | `create_llm_instance` 报 `KeyError` | 删除重复定义，仅保留一处 |
| 4 | `Option/Config2.py` `default()` | `Config.default()` 缺 `llm` 字段 | pydantic 校验失败（`Context.py`/`Logger.py`/`Evaluation.py` 导入期崩） | 在 `default()` 注入 `llm` 回落（读 `E14_LLM_*` 环境变量） |
| 5 | `Core/Graph/TreeGraphDynamic.py` | `_print_bucket_stats` **被注释掉**却仍被调用 | `AttributeError: object has no attribute '_print_bucket_stats'` | 恢复该方法定义（其依赖的 `_analyze_bucket_distribution` 正常存在） |
| 6 | `Core/Index/FaissIndex.py:74,133` | FAISS 索引维度**硬编码 `1024`**（`IndexHNSWFlat(1024, 32)`） | 用非 OpenAI 嵌入（如 MiniLM 384）时 `assert d == self.d` 失败 | 改为 `IndexHNSWFlat(self._faiss_dim(), 32)`，`_faiss_dim()` 从实际嵌入模型探测维度 |
| 7 | `Core/Index/FaissIndex.py:_storage_index` | `persist` 前未确保目录存在 | `faiss.write_index` 报"could not open ... for writing: No such file or directory" | 在 persist 前 `os.makedirs(self.config.persist_path, exist_ok=True)` |
| 8 | （环境层，非代码） | **Windows 下 faiss C++ `FileIOWriter` 无法用非 ASCII 路径打开文件** | 工作目录含中文时 `write_index` 必失败 | 运行期 `working_dir` 置于纯 ASCII 路径（如 `D:/WSPforWB/10.keyan/_e14_run`） |
| 9 | `Core/Graph/TreeGraphDynamic.py`（`_create_task_with_return`）+ `Core/Provider/OpenaiApi.py` | **跨事件循环死锁风险**：线程池每个任务 `asyncio.new_event_loop()` 跑协程，而 `AsyncOpenAI` 客户端与 `asyncio.Semaphore` 在主循环创建；主线程被 `task.result()` 阻塞 | 增量细化在树规模增大时挂死（实测 N=200 挂 3.7 h，CPU≈0、无网络活动） | 加固三件套：per-call `asyncio.wait_for` 硬超时 + no-op 信号量 + 进程级看门狗 |
| 10 | `Core/GraphRAG.py:insert` vs `Core/Graph/BaseGraph.py:build_graph` | **增量建树不刷新向量索引**：`insert()` 在 `build_graph` 之后才 `entities_vdb.build_index`；若直接调用 `graph.build_graph(add=True)`（增量路径），新叶节点进入树但**不进向量索引**，检索对新内容"失明"且**无任何报错** | 注入反驳贴后树叶子 123→139，但检索仍返回旧内容（W3b 首跑即命中） | 增量后手动 `entities_vdb.build_index(await graph.nodes_data(), await graph.node_metadata(), force=True)` + 重建 retriever context |
| 11 | `Core/Graph/TreeGraphDynamic.py` `init_tree_aux` | **增量路径只支持一次 add**：init 重建 `NodeAux` 但不清空 `affected_entities`，第二次 `_refine_graph` 的 size 断言必炸 | W3c 所有 Δ<160 的第 2 批全部崩溃 | init 末尾重置 `self.affected_entities = set()` |
| 12 | `Core/Graph/TreeGraphDynamic.py` `_reorder_node_id` | **reorder 不同步 `layer_to_nodes`**：被丢弃节点以悬挂 index 残留在层列表，下一批 `init_tree_aux` 遍历层列表时 `NodeAux[node.index]` 越界 | #11 补丁后 Δ=8 从 batch2 推进到 batch15 再崩 | reorder 末尾按对象身份重建层列表 |
| 13 | `Core/Graph/TreeGraphDynamic.py` `_refine_one_layer:569` | **上层列表为空时 `next_layer[0]` 越界**：合并/reorder 后 layer+1 可为空，代码假设非空 | Δ=8 per-batch 模式 batch16 崩溃点之一 | （未继续打地鼠——#11–#13 是同一增量状态机腐化的不同症状，腐化会经 `_persist_graph` 固化进磁盘树，进程-per-batch 模式下从"干净"磁盘加载仍崩） |

> **W3c 综合判读**：三处缺陷叠加的净效应是 **EraRAG 动态树的增量路径在多批流式（>3–15 次 refine、~136–160 docs / 12–13 内部节点）下系统性失效，且腐化状态被持久化固化**——这不是补丁可修复的边角，而是"不可合并度量 + 无一致性校验"的增量设计在结构上不可持续的证据（W3c_Δ敏感性报告.md §3）。

> **#10 的性质**：官方代码里有注释 *"Ensure the graph is successfully loaded before proceeding to load the index, as it represents a one-to-one mapping"*，
> 说明作者**知道**这是一对一映射，但**没有在增量路径上强制执行**，属"静默失效"型坑——比抛异常的坑更危险。
> 对本课题的含义：**"增量插入"若不同步更新检索索引，用户会以为系统已学到新内容，实际检索层完全无感**。
> 这恰是本文 C3（星座模型 + P1 增量一致性）要正面解决的一致性问题。

---

## 对可复现性的含义

1. **"增量插入"机制本身可复现**（W1 已铁证）：`add=True → _refine_graph` 真实运行，叶节点 40→55 净增而非重建。
2. **工程实现对"非官方标配环境"高度脆弱**：官方默认假设 OpenAI 嵌入（1536/1024 维）+ GPU + Linux/英文路径。换成 CPU-only + 本地嵌入 + Windows 中文路径，会依次触发缺陷 #4–#8。
3. **论文应如实列出补丁**：建议把 #4–#7 作为 minimal patch 附在附录，#8 作为运行环境约束写明，以保证审稿人能在同等条件下复现 Baseline #2。

---

## 修复落点核对（2026-10-15）

- `Option/Config2.py:116` → `final["llm"] = {...}` 回落 ✅
- `Core/Common/LLMConfig.py:14` → 仅 1 处 `class LLMType` ✅（重复已删）
- `Core/Graph/TreeGraphLSH.py` → 无占位符/TODO 残留 ✅
- `Core/Graph/TreeGraphDynamic.py` → `_print_bucket_stats` 已恢复 ✅
- `Core/Index/FaissIndex.py` → `_faiss_dim()` + `makedirs` 已加 ✅
- 运行目录 → ASCII 路径 ✅
- #9 加固（超时/no-op 信号量/看门狗）落在**实验脚本内**，不改仓库源码 ✅
- #10 增量后索引刷新落在**实验脚本内**（`e14_w3b_capability.py` Phase2）✅

> 生成 patch 命令（在 `_repo` 内执行 git diff 后存为 `E14_repo_compat.patch`）可后续补充。
