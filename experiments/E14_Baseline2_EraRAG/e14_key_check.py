"""E14 —— LLM key 端到端可用性检查（走 EraRAG 自己的 provider 工厂）。

用途：确认 `.env_e14` 里的 base_url / api_key / model 能被 EraRAG-Official
的 `create_llm_instance()`（api_type=open_llm → OpenAILLM）直接消费。

用法：
    cd 实验/E14_Baseline2_EraRAG
    ${KB_ROOT}/.venv_e14/Scripts/python.exe e14_key_check.py

安全性：本脚本只打印 key 的前 8 位与长度，不打印明文，不写入任何日志文件。
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

E14_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = E14_DIR.parents[2]            # ${KB_ROOT}
ENV_FILE = PROJECT_ROOT / ".env_e14"


def load_env(path: Path) -> dict:
    """读取 .env_e14（跳过注释与空行，去掉 CRLF）。"""
    vals = {}
    if not path.exists():
        return vals
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if v.strip():
            vals[k.strip()] = v.strip()
    return vals


def main() -> int:
    vals = load_env(ENV_FILE)
    need = ["E14_LLM_BASE_URL", "E14_LLM_API_KEY", "E14_LLM_MODEL"]
    missing = [k for k in need if not vals.get(k)]
    if missing:
        print("[FAIL] .env_e14 缺少：%s" % missing)
        return 2

    os.environ.update(vals)
    os.environ.setdefault("METAGPT_PROJECT_ROOT", str(E14_DIR / "_cfgroot"))

    sys.path.insert(0, str(E14_DIR))
    import e14_env

    e14_env.bootstrap(verbose=False)

    # 注意：仓库内存在两份 LLMType 枚举（Config/LLMConfig.py 与 Core/Common/LLMConfig.py）。
    # provider 注册表按 Config.LLMConfig.LLMType 建键，因此必须从这里导入，
    # 否则 create_llm_instance() 会抛 KeyError。正常 yaml→Config 路径不受影响。
    from Config.LLMConfig import LLMConfig
    from Core.Common.CostManager import TokenCostManager
    import Core.Provider.OpenaiApi  # noqa: F401  触发 @register_provider 注册 OPEN_LLM
    from Core.Provider.LLMProviderRegister import create_llm_instance

    cfg = LLMConfig(
        api_type="open_llm",
        model=vals["E14_LLM_MODEL"],
        base_url=vals["E14_LLM_BASE_URL"],
        api_key=vals["E14_LLM_API_KEY"],
        max_token=64,
        temperature=0.0,
        timeout=60,
        max_concurrent=8,
    )
    print("[cfg] api_type = %s" % cfg.api_type)
    print("[cfg] model    = %s" % cfg.model)
    print("[cfg] base_url = %s" % cfg.base_url)
    print("[cfg] api_key  = %s...(%d chars)" % (vals["E14_LLM_API_KEY"][:8], len(vals["E14_LLM_API_KEY"])))

    llm = create_llm_instance(cfg)
    llm.cost_manager = TokenCostManager()
    print("[provider] %s" % type(llm).__name__)

    async def run() -> int:
        t0 = time.time()
        out = await llm.aask("Reply with exactly this token and nothing else: E14-KEY-OK")
        dt = time.time() - t0
        print("[EraRAG LLM] 返回  : %r" % (out.strip()[:200],))
        print("[EraRAG LLM] 耗时  : %.2fs" % dt)
        cm = llm.cost_manager
        for attr in ("total_prompt_tokens", "total_completion_tokens", "total_tokens", "total_cost"):
            v = getattr(cm, attr, None)
            if v is not None:
                print("[cost] %-24s = %s" % (attr, v))
        try:
            await llm.aclient.close()   # 关闭 httpx 连接，避免 Windows 退出期 Proactor 噪音
        except Exception:
            pass
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
