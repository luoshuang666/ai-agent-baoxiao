"""评测脚本。

    python eval.py            # 只测检索（Recall + 延迟，不花 LLM 钱）
    python eval.py --full     # 端到端：额外测「拒答正确率」与「答案关键数字命中率」

eval_set.jsonl 每行字段：
    id            编号（q=可答，u=不可答）
    question      用户原话
    answerable    知识库里到底有没有答案（false 用于测拒答）
    doc           该题依赖的语料文件名（语料没入库的题会被跳过，避免误判）
    gold_clause   命中即可算对的条款路径（与 ingest.py 生成的 clause_path 做子串匹配）
    must_contain  答案里必须出现的字符串（--full 时检查，用于验算金额是否算对）
    version       测试时使用的制度版本（可选，版本题必填）

语料未入库的题会先跳过：否则跑出来一堆「未命中」，分不清是检索差还是文档没放进来。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from config import CFG
from retrieval import Retriever

SET = Path(__file__).parent / "eval_set.jsonl"
GIVE_UP = ("没有写明", "未写明", "没有规定", "未规定", "找不到", "联系 HR", "建议咨询", "咨询 HR")


def load_set() -> list[dict]:
    return [json.loads(x) for x in SET.read_text(encoding="utf-8").splitlines() if x.strip()]


def indexed_sources(r: Retriever) -> set[str]:
    data = r.vs.get(include=["metadatas"])
    return {m.get("source") for m in (data["metadatas"] or []) if m}


def main(full: bool = False) -> None:
    items = load_set()
    r = Retriever()
    have = indexed_sources(r)

    runnable, pending = [], []
    for it in items:
        need = it.get("doc")
        (pending if (need and need not in have) else runnable).append(it)

    print(f"\n== 语料覆盖 ==")
    print(f"已入库 {len(have)} 个文件：{', '.join(sorted(x for x in have if x)) or '（空）'}")
    print(f"评测集 {len(items)} 条 → 可跑 {len(runnable)} 条，待补语料 {len(pending)} 条")
    missing = sorted({it["doc"] for it in pending})
    if missing:
        print(f"缺少语料：{', '.join(missing)}")
        print("（补上这些文件后重新 python ingest.py，对应题目会自动纳入评测）")

    if not runnable:
        print("\n[!] 没有可跑的题，先补语料或先建库")
        return

    # ---------- 检索指标 ----------
    hit = total = 0
    lat: list[float] = []
    misses: list[str] = []

    for it in runnable:
        if not it.get("answerable", True):
            continue
        t0 = time.time()
        docs, best = r.search(it["question"], version=it.get("version"))
        lat.append(time.time() - t0)

        total += 1
        gold = it.get("gold_clause", "")
        if gold and any(gold in (m.get("clause_path") or "") for _, m, _ in docs):
            hit += 1
        else:
            misses.append(f"{it['id']} {it['question']}（gold={gold}，最高相关性 {best:.2f}）")

    print(f"\n== 检索 ==")
    print(f"Recall@{CFG.k_final}: {hit}/{total} = {hit / max(total, 1):.1%}")
    print(f"平均检索延迟: {sum(lat) / max(len(lat), 1):.2f}s")
    if misses:
        print("未命中：")
        for m in misses:
            print("  -", m)

    if not full:
        return

    # ---------- 端到端指标 ----------
    from agent import answer   # 延迟导入：只测检索时不初始化 Agent

    right = wrong = 0
    ans_ok = ans_total = 0
    t_start = time.time()

    for it in runnable:
        text, _, _ = answer(it["question"], version=it.get("version"))

        if not it.get("answerable", True):
            if any(k in text for k in GIVE_UP):
                right += 1
            else:
                wrong += 1
                print("  [拒答失败]", it["id"], it["question"], "→", text[:60].replace("\n", " "))
            continue

        must = it.get("must_contain") or []
        if must:
            ans_total += 1
            if all(k in text for k in must):
                ans_ok += 1
            else:
                print("  [数字不符]", it["id"], "期望含", must, "→", text[:60].replace("\n", " "))

    print(f"\n== 端到端 ==")
    print(f"不可答题拒答正确率: {right}/{right + wrong} = {right / max(right + wrong, 1):.1%}")
    print(f"可答题关键数字命中率: {ans_ok}/{ans_total} = {ans_ok / max(ans_total, 1):.1%}")
    print(f"平均耗时: {(time.time() - t_start) / max(len(runnable), 1):.2f}s/条")


if __name__ == "__main__":
    main(full="--full" in sys.argv)
