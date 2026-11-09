"""环境自检：一条命令告诉你「装好了没、能不能跑」。

    python check_env.py

只读不写，不联网（除了读配置），可以随便重复跑。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OK, BAD, WARN = "[OK]  ", "[缺失]", "[注意]"

REQUIRED = {                     # pip 包名 → import 名
    "langchain": "langchain",
    "langchain-openai": "langchain_openai",
    "langchain-chroma": "langchain_chroma",
    "langgraph": "langgraph",
    "chromadb": "chromadb",
    "rank-bm25": "rank_bm25",
    "jieba": "jieba",
    "pandas": "pandas",
    "streamlit": "streamlit",
    "python-dotenv": "dotenv",
}


def main() -> int:
    problems = 0

    # ---------- 1. Python 与虚拟环境 ----------
    print("=" * 62)
    print("1) Python 环境")
    print("=" * 62)
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"     解释器: {sys.executable}")
    print(f"     版本  : {sys.version.split()[0]}")
    if in_venv:
        print(f"{OK} 处于虚拟环境（.venv）")
    else:
        print(f"{WARN} 不在虚拟环境里。能用，但依赖会装到全局；建议先激活 .venv")
    if sys.version_info < (3, 10):
        print(f"{BAD} Python 版本过低，需要 3.10+")
        problems += 1

    # ---------- 2. 依赖 ----------
    print("\n" + "=" * 62)
    print("2) 依赖包")
    print("=" * 62)
    missing = []
    for pkg, mod in REQUIRED.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            print(f"{OK} {pkg:<18} {ver}")
        except Exception as exc:
            missing.append(pkg)
            print(f"{BAD} {pkg:<18} {type(exc).__name__}")
    if missing:
        problems += 1
        print(f"\n     缺 {len(missing)} 个包，装它：")
        print("     pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")

    try:
        importlib.import_module("openpyxl")
        print(f"{OK} openpyxl           可选包也已安装（可用 .xlsx 标准表）")
    except Exception:
        print(f"{WARN} openpyxl           未装（只有用 .xlsx 标准表才需要，用 .csv 可忽略）")

    # ---------- 3. 配置文件 ----------
    print("\n" + "=" * 62)
    print("3) 配置文件 .env")
    print("=" * 62)
    env = ROOT / ".env"
    if not env.exists():
        print(f"{BAD} 没有 .env，执行：copy .env.example .env")
        problems += 1
    else:
        print(f"{OK} .env 存在")
        try:
            from dotenv import load_dotenv

            load_dotenv(env)
            import os

            checks = {
                "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY", ""),
                "SILICONFLOW_API_KEY": os.getenv("SILICONFLOW_API_KEY", ""),
            }
            for k, v in checks.items():
                if v.startswith("sk-") and len(v) > 20:
                    print(f"{OK} {k} 已填写（{v[:6]}…{v[-4:]}）")
                elif v:
                    print(f"{WARN} {k} 看起来不像有效 key（应以 sk- 开头）")
                    problems += 1
                else:
                    print(f"{BAD} {k} 是空的")
                    problems += 1
            print(f"     模型: {os.getenv('LLM_MODEL', '(未设置)')} / "
                  f"{os.getenv('EMB_MODEL', '(未设置)')}")
        except Exception as exc:
            print(f"{WARN} 读取失败: {exc}")

    # ---------- 4. 语料 ----------
    print("\n" + "=" * 62)
    print("4) 语料文件")
    print("=" * 62)
    data = ROOT / "data"
    files = [p for p in sorted(data.iterdir())
             if p.suffix.lower() in (".md", ".xlsx", ".csv")] if data.exists() else []
    if not files:
        print(f"{BAD} data/ 下没有语料")
        problems += 1
    else:
        for p in files:
            print(f"{OK} {p.name}")
        try:
            from ingest import chunks_of

            total = sum(len(chunks_of(p)) for p in files)
            print(f"\n     预计切出 {total} 个 chunk")
            if total != 45:
                print(f"{WARN} 与预期 45 不一致：语料被改过？")
        except Exception as exc:
            print(f"{WARN} 无法预估 chunk 数（缺依赖？）: {exc}")

    # ---------- 5. 向量库 ----------
    print("\n" + "=" * 62)
    print("5) 向量库")
    print("=" * 62)
    chroma_dir = ROOT / "storage" / "chroma"
    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        print(f"{OK} 已建库（storage/chroma 有数据）")
    else:
        print(f"{WARN} 还没建库，跑：python ingest.py")

    # ---------- 6. 框架版本与 Agent 入口 ----------
    print("\n" + "=" * 62)
    print("6) 框架版本与 Agent 入口（决定代码走哪条兼容分支）")
    print("=" * 62)
    try:
        from importlib.metadata import PackageNotFoundError, version

        for pkg in ("langchain", "langchain-core", "langchain-openai",
                    "langchain-chroma", "langgraph", "chromadb"):
            try:
                print(f"{OK} {pkg:<18} {version(pkg)}")
            except PackageNotFoundError:
                print(f"{BAD} {pkg:<18} 未安装")
    except Exception as exc:
        print(f"{WARN} 无法读取版本: {exc}")

    legacy = new = False
    try:
        from langgraph.prebuilt import create_react_agent  # noqa: F401
        legacy = True
    except Exception:
        pass
    try:
        from langchain.agents import create_agent  # noqa: F401
        new = True
    except Exception:
        pass

    if legacy:
        print(f"{OK} Agent 入口：langgraph.prebuilt.create_react_agent（0.3 世代）")
    elif new:
        print(f"{OK} Agent 入口：langchain.agents.create_agent（1.x 世代）")
    else:
        print(f"{BAD} 两条入口都不可用，检查 langchain / langgraph 是否装好")
        problems += 1
    if legacy and new:
        print("     两个入口都在，代码优先用 0.3 世代那条（等价，性能无差异）")

    # ---------- 结论 ----------
    print("\n" + "=" * 62)
    if missing or problems:
        print("结论：还不能跑，按上面 [缺失] / [注意] 逐条处理")
        print("     完整排查见《零基础跑通指南.md》的报错对照表")
        return 1
    print("结论：环境就绪，可以 python ingest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
