"""E14 —— 全链路检查：Config2.yaml → Config → Context → provider → 真 API。

这是 EraRAG 实际建树/检索时会走的路径。通过即说明配置接线无误。

用法：
    D:/WSPforWB/10.keyan/.venv_e14/Scripts/python.exe e14_e2e_check.py           # 真 API
    D:/WSPforWB/10.keyan/.venv_e14/Scripts/python.exe e14_e2e_check.py --stub    # 本地 stub
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

E14_DIR = Path(__file__).resolve().parent
MODE = "stub" if "--stub" in sys.argv else "real"
os.environ["METAGPT_PROJECT_ROOT"] = str(E14_DIR / ("_cfgroot" if MODE == "stub" else "_cfgroot_real"))

sys.path.insert(0, str(E14_DIR))
import e14_env  # noqa: E402

e14_env.bootstrap(verbose=False)

import Core.Provider.OpenaiApi  # noqa: E402,F401  触发 @register_provider 注册 OPEN_LLM
from Core.Common.Context import Context  # noqa: E402


def main() -> int:
    ctx = Context()
    cfg = ctx.config.llm
    print("[cfg] mode           = %s" % MODE)
    print("[cfg] api_type       = %s" % cfg.api_type)
    print("[cfg] model          = %s" % cfg.model)
    print("[cfg] base_url       = %s" % cfg.base_url)
    print("[cfg] api_key        = %s...(%d chars)" % (cfg.api_key[:8], len(cfg.api_key)))
    print("[cfg] max_concurrent = %s" % cfg.max_concurrent)
    print("[cfg] data_root      = %s" % getattr(ctx.config, "data_root", "?"))

    llm = ctx.llm()
    print("[provider] %s" % type(llm).__name__)

    async def run() -> int:
        t0 = time.time()
        out = await llm.aask("Reply with exactly this token and nothing else: E14-E2E-OK")
        print("[result] %r  (%.2fs)" % (out.strip()[:120], time.time() - t0))
        cm = llm.cost_manager
        print(
            "[cost] prompt=%s completion=%s total_cost=%s"
            % (
                getattr(cm, "total_prompt_tokens", "?"),
                getattr(cm, "total_completion_tokens", "?"),
                getattr(cm, "total_cost", "?"),
            )
        )
        try:
            await llm.aclient.close()
        except Exception:
            pass
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
