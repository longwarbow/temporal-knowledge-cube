"""E14 —— 从 `.env_e14` 生成运行配置根（真 API 版）。

设计原则：
  key 只活在 `.env_e14` 与**运行时生成**的配置文件里，绝不写进任何模板/仓库文件。
  生成目录 `_cfgroot_real/` 已在 .gitignore 中忽略。

用法：
    cd 实验/E14_Baseline2_EraRAG
    ${KB_ROOT}/.venv_e14/Scripts/python.exe e14_make_cfg.py            # 真 API
    ${KB_ROOT}/.venv_e14/Scripts/python.exe e14_make_cfg.py --stub     # 本地 stub

生成后运行 EraRAG 时：
    set METAGPT_PROJECT_ROOT=<E14>/_cfgroot[_real]
    cwd = <E14>/_repo
"""
from __future__ import annotations

import sys
from pathlib import Path

E14_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = E14_DIR.parents[2]
ENV_FILE = PROJECT_ROOT / ".env_e14"

TEMPLATE = """\
# 由 e14_make_cfg.py 自动生成 —— 请勿手工编辑，勿提交（key 在此文件中）。
# mode: {mode}
llm:
  api_type: "open_llm"
  model: "{model}"
  base_url: "{base_url}"
  api_key: "{api_key}"
  max_token: {max_token}
  temperature: 0.0
  timeout: 600
  max_concurrent: {max_concurrent}

embedding:
  api_type: "hf"
  api_key: "sk-local-embed"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  cache_folder: "{hf_cache}"
  dimensions: 384
  embed_batch_size: 64

data_root: "{data_root}"
working_dir: "{working_dir}"
exp_name: "E14"
external_tree_path: ""
"""


def load_env(path: Path) -> dict:
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
    stub = "--stub" in sys.argv
    if stub:
        mode, out_dir = "stub", E14_DIR / "_cfgroot"
        base_url, api_key, model = "http://127.0.0.1:8791/v1", "sk-e14-stub", "stub-local"
    else:
        mode, out_dir = "real-api", E14_DIR / "_cfgroot_real"
        vals = load_env(ENV_FILE)
        missing = [k for k in ("E14_LLM_BASE_URL", "E14_LLM_API_KEY", "E14_LLM_MODEL") if not vals.get(k)]
        if missing:
            print("[FAIL] .env_e14 缺少：%s" % missing)
            return 2
        base_url = vals["E14_LLM_BASE_URL"]
        api_key = vals["E14_LLM_API_KEY"]
        model = vals["E14_LLM_MODEL"]

    body = TEMPLATE.format(
        mode=mode,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_token=2048,
        max_concurrent=8,
        hf_cache=(E14_DIR / "_hf_cache").as_posix(),
        data_root=(E14_DIR / "_data").as_posix(),
        working_dir=(E14_DIR / ("runs" if mode == "stub" else "runs_real")).as_posix(),
    )

    target = out_dir / "Option" / "Config2.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as fh:   # Py3.9 兼容：write_text 无 newline 参数
        fh.write(body)

    print("[ok] mode       = %s" % mode)
    print("[ok] config     = %s" % target)
    print("[ok] model      = %s" % model)
    print("[ok] base_url   = %s" % base_url)
    print("[ok] api_key    = %s...(%d chars)" % (api_key[:8], len(api_key)))
    print("[ok] 运行前请设 METAGPT_PROJECT_ROOT=%s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
