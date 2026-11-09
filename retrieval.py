"""检索层：Chroma 向量召回 + BM25 关键词召回，RRF 融合，取前 k。

- 向量负责语义相近的表述，BM25 负责条款号、金额、专有名词这类精确 token；
- 融合用 RRF：余弦分数 0~1 与 BM25 分数 0~20+ 量纲不同，加权需要归一化和调参；
- 相关性低于阈值时只做一次 query 重写（在 agent 的工具里调用）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from config import CFG
from llm import get_llm

if TYPE_CHECKING:                      # 仅为类型检查，运行时不强制安装
    from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """中文分词（jieba 延迟导入，方便不装依赖时也能跑单元测试）。"""
    import jieba

    return [w for w in jieba.lcut(text) if w.strip()]


TERMS = (
    "出差住宿→住宿费标准；报销多久到账→报销结算时限；能报多少→费用标准/限额；"
    "发票→增值税专用发票；打车→市内交通费；吃饭→餐费补贴"
)
REWRITE_PROMPT = f"""把用户的口语问题改写为适合检索公司制度的正式表述。
要求：保留原意，只换成制度用语并补全隐含主体，输出一行，不要解释。
术语参考：{TERMS}

问题：{{q}}
改写："""


def rewrite_query(q: str) -> str:
    """检索得分太低时用一次 LLM 重写；失败就退回原问题。"""
    try:
        out = get_llm(0.0).invoke(REWRITE_PROMPT.format(q=q)).content
        return str(out).strip().splitlines()[0][:200] or q
    except Exception:
        return q


def rrf_fuse(vec: list[tuple[str, dict, float]],
             kw: list[tuple[str, dict, float]],
             rrf_k: int = 60) -> list[tuple[str, dict, float]]:
    """Reciprocal Rank Fusion：只用排名不用分数，规避余弦与 BM25 的量纲差异。"""
    acc: dict[str, list] = {}
    for ranked in (vec, kw):
        for rank, (text, meta, _) in enumerate(ranked):
            acc.setdefault(text, [0.0, meta])[0] += 1.0 / (rrf_k + rank + 1)
    return sorted(((t, v[1], v[0]) for t, v in acc.items()), key=lambda x: x[2], reverse=True)


class Retriever:
    def __init__(self) -> None:
        from langchain_chroma import Chroma

        from llm import get_embeddings

        self.vs = Chroma(
            collection_name=CFG.collection,
            persist_directory=str(CFG.chroma_dir),
            embedding_function=get_embeddings(),
        )
        data = self.vs.get(include=["documents", "metadatas"])
        self.texts: list[str] = data["documents"] or []
        self.metas: list[dict] = data["metadatas"] or []
        self._bm25 = None

        if not self.texts:
            raise RuntimeError("向量库是空的，先跑 python ingest.py")

    # ---------- 关键词一路 ----------
    @property
    def bm25(self):
        if self._bm25 is None and CFG.use_bm25 and self.texts:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        return self._bm25

    def _keyword(self, query: str, k: int, version: str | None):
        bm = self.bm25
        if bm is None:
            return []
        scores = bm.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            if version and self.metas[i].get("version") not in (version, "all"):
                continue
            out.append((self.texts[i], self.metas[i], float(scores[i])))
            if len(out) >= k:
                break
        return out

    # ---------- 融合 ----------
    def search(self, query: str, version: str | None = None,
               k: int | None = None) -> tuple[list[tuple[str, dict, float]], float]:
        """返回 (top-k 片段, 向量最高相关性)；相关性用于阈值判定是否重写。"""
        k = k or CFG.k_final
        where = {"version": version} if version else None

        pairs = self.vs.similarity_search_with_relevance_scores(
            query, k=CFG.k_vec, filter=where
        )
        vec = [(d.page_content, d.metadata, float(s)) for d, s in pairs]
        best = max((s for _, _, s in vec), default=0.0)

        kw = self._keyword(query, CFG.k_kw, version)
        if not kw:
            return vec[:k], best

        return rrf_fuse(vec, kw, CFG.rrf_k)[:k], best
