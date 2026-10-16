# -*- coding: utf-8 -*-
"""
E14 / W1 冒烟：EraRAG 动态树路径在 CPU-only + 智谱 GLM-4-Flash + 本地 MiniLM 下跑通。
- Phase A: 初始建树 (graph.add=False, graph.force=True)  -> 应触发 _build_graph (重建)
- Phase B: 增量补充 (graph.add=True,  graph.force=False) -> 应触发 _refine_graph (增量插入)
本脚本直接构造 Config 对象（绕过文件 merge 的占位符覆盖问题），仍走真实 GraphRAG(config) 代码路径。
"""
import os, sys, asyncio, json, time
from pathlib import Path

# ---------- 0. 强制 CPU（最关键）：本机无 NVIDIA 驱动，SentenceTransformer 默认判定 device=cuda 会崩 ----------
# 用 torch.backends.cuda.enabled=False 彻底禁用 CUDA：这会让 torch.cuda.is_available() 永远返回 False，
# 即使 sentence-transformers 内部用 `from torch.cuda import is_available` 绑定了原引用也生效。
import torch  # noqa: E402
torch.backends.cuda.enabled = False  # type: ignore
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 绝对强制 CPU：monkeypatch SentenceTransformer.__init__，无论其内部如何判定 device 都走 cpu。
# （本机无 NVIDIA 驱动，sentence-transformers 默认会尝试 CUDA 而崩溃。）
import sentence_transformers  # noqa: E402
_orig_st_init = sentence_transformers.SentenceTransformer.__init__
def _force_cpu_st_init(self, *a, **k):
    k["device"] = "cpu"
    return _orig_st_init(self, *a, **k)
sentence_transformers.SentenceTransformer.__init__ = _force_cpu_st_init

# ---------- 1. 环境：HF 镜像 + 本地缓存（避免联网下载 MiniLM） ----------
REPO = Path("D:/WSPforWB/10.keyan/场景1-海量短文本/实验/E14_Baseline2_EraRAG/_repo").resolve()
ROOT = Path("D:/WSPforWB/10.keyan/场景1-海量短文本/实验/E14_Baseline2_EraRAG").resolve()
HF_CACHE = str(ROOT / "_hf_cache")
MINILM = str(next((ROOT / "_hf_cache").rglob("model.safetensors")).parent)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = HF_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------- 2. 读取 .env_e14 里的智谱 key（位于项目根目录 D:/WSPforWB/10.keyan/） ----------
ENV_FILE = Path("D:/WSPforWB/10.keyan/.env_e14")
env = {}
for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    env[k.strip()] = v.strip().strip('"').strip("'")
KEY = env["E14_LLM_API_KEY"]; BASE = env["E14_LLM_BASE_URL"]; MODEL = env["E14_LLM_MODEL"]

# ---------- 3. bootstrap（shims / sys.path） ----------
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))
import e14_env  # noqa: E402
e14_env.bootstrap(verbose=False)

from Option.Config2 import Config                                   # noqa: E402
from Core.GraphRAG import GraphRAG                                  # noqa: E402
from Core.Chunk.DocChunk import DocChunk                            # noqa: E402
from Core.Storage.NameSpace import Workspace                        # noqa: E402

# 运行期数据目录必须放在纯 ASCII 路径下：faiss 的 C++ 层 FileIOWriter 在
# Windows 上无法用非 ASCII（中文）路径打开文件，否则 write_index 必失败。
# 因此 _w1work/_w1tmp/_w1data 全部放到 D:/WSPforWB/10.keyan/_e14_run（无中文）。
ASCII_BASE = "D:/WSPforWB/10.keyan/_e14_run"
W1WORK = str(Path(ASCII_BASE) / "_w1work")
W1TMP = str(Path(ASCII_BASE) / "_w1tmp")
W1DATA = str(Path(ASCII_BASE) / "_w1data")

# ---------- 4. 语料：程序化生成短文本（足够多话题以触发 LSH 聚类） ----------
TOPICS_A = [
    "A wildfire near the coast forced the evacuation of two thousand residents on Monday.",
    "The central bank raised interest rates by half a percentage point to curb inflation.",
    "A tech startup launched an open-source model that runs on a single consumer GPU.",
    "Researchers reported a new vaccine showing eighty percent efficacy in early trials.",
    "The local football team won the championship after a penalty shootout.",
    "A earthquake of magnitude six struck the region, damaging several bridges.",
    "The government announced a subsidy for households installing solar panels.",
    "A major retailer admitted a data breach affecting three million customers.",
    "Scientists detected a previously unknown species in the deep ocean trench.",
    "The film festival awarded its top prize to a documentary about climate change.",
    "A cyberattack disrupted rail services across the northern corridor for six hours.",
    "The mayor pledged to plant one million trees before the next election.",
    "A drought reduced the river flow to a record low, threatening fish stocks.",
    "The airline canceled two hundred flights due to a strike by ground crew.",
    "A study found that walking thirty minutes daily lowers the risk of heart disease.",
    "The port handled a record volume of container traffic this quarter.",
    "A musician released an album composed entirely with artificial intelligence tools.",
    "The court ruled that the streaming service must pay higher royalties to artists.",
    "A heatwave pushed electricity demand to its peak and triggered rolling outages.",
    "The university expanded its scholarship program for first-generation students.",
    "A factory recall affected four models of electric scooters over battery faults.",
    "The satellite returned high-resolution images of the polar ice sheet.",
    "A protest drew ten thousand people demanding stronger air-quality standards.",
    "The energy company retired its last coal plant and switched to wind power.",
    "A blockchain project lost funds after a flaw in its smart contract was exploited.",
    "The hospital opened a new wing funded by a private donation.",
    "A rare transit of the planet was visible from the southern hemisphere.",
    "The committee approved a plan to pedestrianize the historic city center.",
    "A fishing ban was extended to help depleted stocks recover.",
    "The telecom operator launched fifth-generation service in rural counties.",
    "A hiker was rescued after being lost for three days in the mountain range.",
    "The museum returned a looted artifact to its country of origin.",
    "A new law requires labels on food products containing added sugars.",
    "The rover collected samples suggesting ancient microbial life may have existed.",
    "A storm surge flooded low-lying neighborhoods and cut off the main road.",
    "The startup raised fifty million dollars to scale its battery recycling process.",
    "A survey showed public trust in institutions declining for the third year.",
    "The bridge reopened after a twelve-month renovation project.",
    "A language model passed a medical licensing exam with a high score.",
    "The farm cooperative adopted drought-resistant crops to secure yields.",
]
TOPICS_B = [
    "A second wildfire broke out down the valley, stretching firefighting resources thin.",
    "The central bank signaled further rate moves if inflation stays above target.",
    "An open-source model update improved inference speed on consumer hardware.",
    "A follow-up study confirmed the vaccine efficacy held across age groups.",
    "The football league announced a new salary cap for the coming season.",
    "Aftershocks rattled the region a day after the initial earthquake.",
    "Solar installations doubled after the subsidy took effect.",
    "The retailer offered free credit monitoring to breach-affected customers.",
    "A second deep-sea expedition found similar organisms at a different site.",
    "The documentary director announced a sequel on ocean acidification.",
    "Rail services resumed after engineers patched the compromised system.",
    "Tree-planting volunteers exceeded the mayor's target within two weeks.",
    "The river authority imposed emergency water restrictions during the drought.",
    "The airline reached a tentative agreement with the ground crew union.",
    "A larger trial began to confirm the daily-walk heart benefit in older adults.",
]

def make_docs(texts):
    return [{"title": f"doc{i}", "content": t} for i, t in enumerate(texts)]

# ---------- 5. 构造 Config（动态树方法 + 真 key + 本地 MiniLM） ----------
def build_config(add: bool, force: bool) -> Config:
    method = Config.read_yaml(REPO / "Option" / "Method" / "Dynamic.yaml")
    final = dict(method)
    final["llm"] = {
        "api_type": "open_llm",
        "model": MODEL,
        "base_url": BASE,
        "api_key": KEY,
        "api_version": None,
    }
    final["embedding"] = {
        "api_type": "hf",
        "model": MINILM,
        "cache_dir": HF_CACHE,
        "dimensions": 384,
        "embed_batch_size": 16,
        "embedding_func_max_async": 8,
    }
    final["working_dir"] = W1WORK
    final["exp_name"] = "w1smoke"
    final["data_root"] = W1DATA
    final["dataset_name"] = "w1smoke"
    final["graph"] = dict(method.get("graph", {}))
    final["graph"]["add"] = add
    final["graph"]["force"] = force
    return Config(**final)

def leaf_count(digimon):
    g = digimon.graph._graph
    try:
        return len(g.leaf_nodes) if hasattr(g, "leaf_nodes") else len(g.get_layer(g.num_layers - 1))
    except Exception:
        return -1

def token_report(digimon, tag):
    cm = digimon.llm._cost_manager if hasattr(digimon.llm, "_cost_manager") else None
    if cm is None:
        return
    for attr in ("total_prompt_tokens", "total_completion_tokens", "total_tokens"):
        v = getattr(cm, attr, None)
        if v is not None:
            print(f"  [{tag}] cost.{attr} = {v}")

async def main():
    log = {"phase": [], "routes": {}}
    # ===== Phase A: 初始建树 =====
    t0 = time.time()
    cfgA = build_config(add=False, force=True)
    # 调用桩：记录 build_graph 路由到哪个方法
    routes = {"build": 0, "refine": 0, "build_n": 0, "refine_n": 0}
    import Core.Graph.TreeGraphDynamic as TGD
    real_build = TGD.TreeGraphDynamic._build_graph
    real_refine = TGD.TreeGraphDynamic._refine_graph
    async def wrap_build(self, chunks):
        routes["build"] += 1; routes["build_n"] = len(chunks)
        return await real_build(self, chunks)
    async def wrap_refine(self, chunks):
        routes["refine"] += 1; routes["refine_n"] = len(chunks)
        return await real_refine(self, chunks)
    TGD.TreeGraphDynamic._build_graph = wrap_build
    TGD.TreeGraphDynamic._refine_graph = wrap_refine

    digiA = GraphRAG(config=cfgA)
    docsA = make_docs(TOPICS_A)
    await digiA.insert(docsA)
    leaves_A = leaf_count(digiA)
    token_report(digiA, "PhaseA")
    log["phase"].append({"name": "A_initial_build", "add": False, "force": True,
                         "seconds": round(time.time() - t0, 1), "leaf_nodes": leaves_A})
    print(f"Phase A done: leaves={leaves_A}, routes={routes}, {time.time()-t0:.1f}s")

    # ===== Phase B: 增量补充（只传 15 条新 docs，构造干净元组） =====
    t1 = time.time()
    cfgB = build_config(add=True, force=False)
    digiB = GraphRAG(config=cfgB)  # 同一 working_dir -> 加载 Phase A 的树
    # 用独立临时 chunk 存储构造 (key, TextChunk) 元组，避免与主存储合并
    tmp_chunk = DocChunk(cfgB.chunk, digiB.ENCODER,
                         Workspace(W1TMP, "w1smoke").make_for("chunk_storage"))
    await tmp_chunk.build_chunks(make_docs(TOPICS_B))
    new_tuples = await tmp_chunk.get_chunks()      # 恰好 15 个 (key, TextChunk)
    # 经路由器 build_graph(add=True) 验证走 _refine_graph
    await digiB.graph.build_graph(new_tuples, force=False, add=True)
    leaves_B = leaf_count(digiB)
    token_report(digiB, "PhaseB")
    log["phase"].append({"name": "B_incremental_add", "add": True, "force": False,
                         "new_chunks": len(new_tuples), "seconds": round(time.time() - t1, 1),
                         "leaf_nodes_after": leaves_B})
    log["routes"] = routes
    print(f"Phase B done: new_tuples={len(new_tuples)}, leaves_after={leaves_B}, routes={routes}, {time.time()-t1:.1f}s")

    # ===== 判定 =====
    ok_A = routes["build"] >= 1 and leaves_A > 5
    ok_B = routes["refine"] >= 1 and leaves_B > leaves_A
    log["verdict"] = {
        "PhaseA_build_path": ok_A,
        "PhaseB_refine_path": ok_B,
        "leaves_grew": leaves_B > leaves_A,
        "W1_PASS": ok_A and ok_B,
    }
    print("\n===== W1 SMOKE VERDICT =====")
    print(json.dumps(log["verdict"], ensure_ascii=False, indent=2))
    (ROOT / "W1_冒烟记录.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    # 还原猴子补丁（若后续复用）
    TGD.TreeGraphDynamic._build_graph = real_build
    TGD.TreeGraphDynamic._refine_graph = real_refine

if __name__ == "__main__":
    asyncio.run(main())
