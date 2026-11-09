"""消融实验：只测检索（不花钱调对话模型），对比不同检索配置的 Recall@k。

用法：
    python ablation.py

会依次跑：
    A. 向量单路，k=4
    B. 向量 + BM25（RRF），k=4
    C. 向量 + BM25（RRF），k=6
输出表格并写 storage/ablation.md，可直接贴进文档/简历。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from config import CFG
from retrieval import Retriever

SET = Path(__file__).parent / "eval_set.jsonl"
OUT = Path(__file__).parent / "storage" / "ablation.md"


def load_items() -> list[dict]:
    return [json.loads(x) for x in SET.read_text(encoding="utf-8").splitlines() if x.strip()]


def recall(r: Retriever, items: list[dict], k: int, version_of=None) -> tuple[int, int, float]:
    hit = total = 0
    lat: list[float] = []
    for it in items:
        if not it.get("answerable", True):
            continue
        t0 = time.time()
        docs, _ = r.search(it["question"], version=it.get("version"), k=k)
        lat.append(time.time() - t0)
        total += 1
        gold = it.get("gold_clause", "")
        if gold and any(gold in (m.get("clause_path") or "") for _, m, _ in docs):
            hit += 1
    return hit, total, sum(lat) / max(len(lat), 1)


def main() -> None:
    items = load_items()
    r = Retriever()
    rows = []

    configs = [
        ("A. 向量单路 k=4", False, 4),
        ("B. 向量+BM25 k=4", True, 4),
        ("C. 向量+BM25 k=6", True, 6),
    ]
    try:
        for name, use_bm25, k in configs:
            CFG.use_bm25 = use_bm25
            r._bm25 = None                      # 让 BM25 按新开关重建
            hit, total, lat = recall(r, items, k)
            rows.append({"配置": name, "Recall": f"{hit}/{total} = {hit / total:.1%}",
                         "平均延迟": f"{lat:.2f}s"})
            print(f"{name:<20} {hit}/{total} = {hit / total:.1%}   {lat:.2f}s")
    finally:
        CFG.use_bm25 = True

    header = "| 配置 | Recall | 平均延迟 |\n|---|---|---|\n"
    body = "\n".join(f"| {x['配置']} | {x['Recall']} | {x['平均延迟']} |" for x in rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(header + body + "\n", encoding="utf-8")
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
