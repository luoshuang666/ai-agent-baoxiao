"""模型工厂：DeepSeek 与硅基流动都走 OpenAI 兼容协议，所以统一用 langchain-openai。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from config import CFG

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_llm(temperature: float = 0.2) -> "ChatOpenAI":
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=CFG.llm_model,
        api_key=CFG.deepseek_key,
        base_url=CFG.deepseek_base,
        temperature=temperature,
        timeout=90,
        max_retries=2,
    )


def get_embeddings() -> "OpenAIEmbeddings":
    from langchain_openai import OpenAIEmbeddings

    # check_embedding_ctx_length=False 很关键：
    # 默认会用 tiktoken 按 OpenAI 的切词规则预处理，换成非 OpenAI 端点容易报错。
    return OpenAIEmbeddings(
        model=CFG.emb_model,
        api_key=CFG.sf_key,
        base_url=CFG.sf_base,
        dimensions=CFG.emb_dim,
        check_embedding_ctx_length=False,
    )
