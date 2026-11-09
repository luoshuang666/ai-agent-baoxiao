"""全局配置：所有可调参数都在这里。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).parent


def _chroma_dir() -> Path:
    """向量库目录。默认在项目内，可用 .env 的 CHROMA_DIR 改到别处。

    如果 Windows 上反复出现「unable to open database file」，
    把 CHROMA_DIR 指向一个纯英文、无空格、非同步盘的路径（如 F:\\chroma_db）。
    """
    raw = os.getenv("CHROMA_DIR", "").strip().strip('"').strip("'")
    return Path(raw) if raw else ROOT / "storage" / "chroma"


@dataclass
class Settings:
    # ---------- LLM：DeepSeek（OpenAI 兼容协议） ----------
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-flash")
    deepseek_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # ---------- Embedding：硅基流动 Qwen3-Embedding ----------
    emb_model: str = os.getenv("EMB_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    emb_dim: int = int(os.getenv("EMB_DIM", "1024"))
    sf_key: str = os.getenv("SILICONFLOW_API_KEY", "")
    sf_base: str = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

    # ---------- 向量库：Chroma 本地持久化 ----------
    collection: str = "policy_kb"
    data_dir: Path = ROOT / "data"
    chroma_dir: Path = field(default_factory=_chroma_dir)

    # ---------- 检索参数 ----------
    k_vec: int = 10          # 向量召回条数
    k_kw: int = 10           # 关键词召回条数
    k_final: int = 6         # 最终给模型的片段数（消融：4→6 时 Recall 70%→85%）
    rrf_k: int = 60          # RRF 常数
    relevance_th: float = 0.35   # 低于此相关性 → 触发一次 query 重写
    use_bm25: bool = os.getenv("USE_BM25", "1") == "1"

    # ---------- 多轮记忆 ----------
    history_turns: int = int(os.getenv("HISTORY_TURNS", "6"))   # 送进模型的最近轮数（1 轮 = 一问一答）


CFG = Settings()
