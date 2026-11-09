"""一键冒烟测试：把项目所有功能跑一遍，输出 PASS/FAIL。

用法（在 skeleton 目录下）：
    python smoke_test.py

会花几毛钱：只调 5 次左右对话模型 + 一次检索评测；建库走增量，不会重复调 embedding。
UI 相关的「刷新恢复」「新会话」只能在浏览器里点，见文末手动清单。
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

import agent  # noqa: E402
from config import CFG  # noqa: E402
from ingest import chunks_of, split_md  # noqa: E402
from obs import tail as log_tail  # noqa: E402
from retrieval import Retriever  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASSED if cond else FAILED).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def main() -> int:
    t_all = time.time()
    py = sys.executable

    # ---------- 1. 环境 ----------
    print("\n== 1. 环境 ==")
    check("虚拟环境", sys.prefix != sys.base_prefix, sys.prefix)
    check("两个 API Key 已配置", bool(CFG.deepseek_key and CFG.sf_key))
    check("记忆窗口可配", CFG.history_turns >= 1, f"HISTORY_TURNS={CFG.history_turns}")

    # ---------- 2. 建库 + 增量 ----------
    print("\n== 2. 建库与增量 ==")
    r1 = run([py, "ingest.py"])
    r2 = run([py, "ingest.py"])
    out1, out2 = r1.stdout + r1.stderr, r2.stdout + r2.stderr
    check("首次运行成功（exit=0）", r1.returncode == 0, f"exit={r1.returncode}")
    check("集合共 45 个 chunk", "当前集合共 45 个" in out1, out1.strip().splitlines()[-1] if out1.strip() else "无输出")
    check("第二次全部 skip（增量生效）",
          "本次写入 0 个 chunk" in out2 and out2.count("skip") == 5,
          f"skip={out2.count('skip')}")

    # ---------- 3. 切分 ----------
    print("\n== 3. 条款级切分 ==")
    expect = {
        "差旅费管理制度_v2025.md": 13, "差旅费管理制度_v2024.md": 13,
        "发票管理规定.md": 5, "费用报销流程.md": 5, "报销标准速查表_v2025.csv": 9,
    }
    total = 0
    per_file: dict[str, int] = {}
    for p in sorted(CFG.data_dir.glob("*")):
        if p.suffix.lower() not in (".md", ".csv", ".xlsx"):
            continue
        n = len(chunks_of(p))
        per_file[p.name] = n
        total += n
    check("五份语料都在", set(per_file) == set(expect), str(per_file))
    check("单文件块数与预期一致", per_file == expect, f"实际={per_file}")
    check("合计 45 块", total == 45, f"total={total}")

    md = (CFG.data_dir / "差旅费管理制度_v2025.md").read_text(encoding="utf-8")
    chunks = split_md(md, "差旅费管理制度_v2025.md", "2025")
    check("章节路径注入到 chunk 首行", all(c["text"].startswith("【") for c in chunks))
    check("条款路径含层级", any(" > " in c["metadata"]["clause_path"] for c in chunks))
    check("版本号进 metadata", all(c["metadata"]["version"] == "2025" for c in chunks))

    lv = agent.match_level("高级测试工程师")
    check("岗位→职级映射（精确）", bool(lv) and lv[0]["职级档位"] == "P7及以上", str(lv))
    lv2 = agent.match_level("高级测试工程师（上海）")
    check("长岗位名不被短名吃掉", bool(lv2) and lv2[0]["岗位名称"] == "高级测试工程师", str(lv2))
    lv3 = agent.match_level("工程师")
    check("泛称岗位返回多个候选", len(lv3) >= 2, f"{len(lv3)} 个候选")
    check("未知岗位不硬猜", agent.match_level("外星人") == [])

    # ---------- 4. 检索与版本过滤 ----------
    print("\n== 4. 检索与版本过滤 ==")
    r = Retriever()
    d25, best25 = r.search("上海出差住宿费标准", version="2025")
    d24, _ = r.search("上海出差住宿费标准", version="2024")
    t25 = " ".join(t for t, _, _ in d25)
    t24 = " ".join(t for t, _, _ in d24)
    check("检索返回 k_final 条", len(d25) == CFG.k_final, f"k={len(d25)}")
    check("2025 结果含 600（P5-P6 一线）", "600" in t25)
    check("2024 结果含 500、不含 600", "500" in t24 and "600" not in t24)
    check("版本过滤干净（无跨版本 chunk）",
          all(m.get("version") in ("2025", "all") for _, m, _ in d25))
    check("相关性分数可用", 0.0 <= best25 <= 1.0, f"best={best25:.2f}")

    # ---------- 5. 端到端：工具 + 引用 ----------
    print("\n== 5. 工具调用与引用（会调模型）==")
    t, cites, _ = agent.answer("我是 P6，去上海出差，住宿费能报多少？", thread_id="smoke-tool")
    trace = agent.get_last_trace()
    check("答案含 600", "600" in t, t[:60].replace("\n", " "))
    check("引用非空", len(cites) > 0, f"{len(cites)} 条")
    check("引用带出处与条款", all(c.get("source") and c.get("clause") for c in cites))
    check("引用条款与住宿相关", any("住宿" in (c["clause"] or "") or "住宿" in c["quote"] for c in cites))
    check("回答分四段（直接回答/怎么操作/依据/确认）",
          all(k in t for k in ("直接回答", "怎么操作", "依据", "确认")),
          t[:80].replace("\n", " "))
    check("回答不冗长（≤550 字）", len(t) <= 550, f"{len(t)} 字")

    calc = agent.calc_reimbursement.invoke(
        {"level": "P6", "city_tier": "一线", "nights": 3, "actual_per_night": 750})
    check("算账工具：上限 1800、超支 450", "1800" in calc and "450" in calc,
          calc.replace("\n", " ")[:80])

    t_calc, _, _ = agent.answer(
        "我是 P6，去上海出差 3 晚，酒店 750 一晚，能报多少？", thread_id="smoke-calc")
    check("端到端给出算好的结论（1800/450）",
          "1800" in t_calc and "450" in t_calc, t_calc[:80].replace("\n", " "))
    check("记录了决策轨迹", len(trace) >= 2, f"{len(trace)} 步")
    check("问答写了日志（qa_ok）",
          any(l.get("event") == "qa_ok" and l.get("rid") == agent.get_last_rid()
              for l in log_tail(50)), agent.get_last_rid())
    rec = next((l for l in log_tail(80)
                if l.get("rid") == agent.get_last_rid() and l.get("event") == "qa_ok"), {})
    check("日志记录证据门控结果", rec.get("grounded") is True, str(rec.get("missing")))

    agent.SESSION["grounds"] = ["住宿上限=600 元；3 晚上限=1800 元"]
    check("门控：有依据的数字通过", agent._ungrounded("600 元、1800 元 [1]") == [])
    check("门控：无依据数字被揪出", agent._ungrounded("上限 999 元 [1]") == ["999"])
    check("轨迹含工具调用与耗时",
          any(s.get("tool") in ("search_policy", "lookup_standard") for s in trace)
          and all(isinstance(s.get("ms"), int) for s in trace),
          str([s.get("tool") for s in trace]))
    check("轨迹带检索相关性分数",
          any(s.get("best") is not None for s in trace),
          str([s.get("best") for s in trace]))

    t_lv, _, _ = agent.answer("我是高级测试工程师，去上海出差，住宿费能报多少？", thread_id="smoke-level")
    check("不知道职级时按岗位映射到 P7（800）", "800" in t_lv, t_lv[:60].replace("\n", " "))

    # ---------- 6. 版本切换（同一问题） ----------
    print("\n== 6. 版本切换 ==")
    t24, _, _ = agent.answer("我是 P6，去上海出差，住宿费能报多少？", version="2024", thread_id="smoke-v24")
    conflict = any(k in t24 for k in ("数据不一致", "无法确认最终标准"))
    check("2024 版本给出 500 且不报冲突", "500" in t24 and not conflict,
          t24[:60].replace("\n", " "))

    # ---------- 7. 拒答 ----------
    print("\n== 7. 无依据拒答 ==")
    t_refuse, _, _ = agent.answer("出差可以坐飞机头等舱吗？", thread_id="smoke-refuse")
    give_up = ("没有写明", "未写明", "没有规定", "未规定", "找不到", "联系 HR", "建议咨询", "咨询 HR")
    check("明确说制度没写，不编造", any(k in t_refuse for k in give_up), t_refuse[:60].replace("\n", " "))

    # ---------- 8. 多轮记忆 ----------
    print("\n== 8. 多轮记忆（会调模型）==")
    agent.answer("我下周去上海出差，住宿费能报多少？", thread_id="smoke-mem")
    t2, _, _ = agent.answer("那 P7 呢？", thread_id="smoke-mem")
    check("同会话追问带上「上海」", "800" in t2 and ("上海" in t2 or "一线" in t2),
          t2[:60].replace("\n", " "))
    check("同会话追问给出 800（P7 一线）", "800" in t2, t2[:60].replace("\n", " "))
    t3, _, _ = agent.answer("那 P7 呢？", thread_id="smoke-other")
    check("新会话槽位不串（没有继承上海）",
          agent.get_slots("smoke-other").get("city") is None,
          str(agent.get_slots("smoke-other")))
    check("新会话会反问城市", "城市" in t3, t3[:60].replace("\n", " "))

    # 槽位记忆：窗口调到 1 轮，仍记得第 1 轮说过的职级和城市
    s = agent.update_slots("smoke-slot", "我是 P7，下周去上海出差三天，酒店 750 元")
    check("槽位抽取（职级/城市/天数/金额）",
          s.get("level") == "P7" and s.get("city") == "上海"
          and s.get("days") == 3 and s.get("amount") == 750, str(s))
    check("槽位提示能注入模型", "职级=P7" in (agent._slots_system("smoke-slot") or ""),
          str(agent._slots_system("smoke-slot")))

    saved_turns = CFG.history_turns
    CFG.history_turns = 1
    agent.answer("我是 P7，下周去上海出差", thread_id="smoke-slot-e2e")
    t_slot, _, _ = agent.answer("住宿费能报多少？", thread_id="smoke-slot-e2e")
    CFG.history_turns = saved_turns
    check("窗口=1 也靠槽位答对（800）", "800" in t_slot, t_slot[:60].replace("\n", " "))

    # ---------- 9. 窗口裁剪（不调模型） ----------
    print("\n== 9. 窗口裁剪 ==")
    msgs = []
    for i in range(10):
        msgs.append(HumanMessage(f"问题{i}"))
        msgs.append(AIMessage(f"回答{i}"))
    trimmed = agent._trim_history({"messages": msgs})["llm_input_messages"]
    body = [m for m in trimmed if getattr(m, "type", "") != "system"]
    check("只送最近 N 轮", len(body) <= CFG.history_turns * 2, f"len={len(body)}")
    check("裁剪从 human 处开始", body[0].type == "human")

    tc = {"name": "search_policy", "args": {"query": "x"}, "id": "c1"}
    toy = [HumanMessage("q0"), AIMessage("", tool_calls=[tc]),
           ToolMessage("工具结果", tool_call_id="c1"), AIMessage("a0"),
           HumanMessage("q1"), AIMessage("", tool_calls=[tc]),
           ToolMessage("工具结果2", tool_call_id="c1"), AIMessage("a1")]
    saved = CFG.history_turns
    CFG.history_turns = 1
    cut = agent._trim_history({"messages": toy})["llm_input_messages"]
    CFG.history_turns = saved
    check("窗口=1 时不拆散工具对", cut[0].type == "human" and len(cut) >= 2,
          f"types={[m.type for m in cut]}")

    # ---------- 10. 历史恢复 ----------
    print("\n== 10. 刷新恢复 ==")
    hist = agent.load_history("smoke-mem")
    check("能读回历史", len(hist) >= 4, f"{len(hist)} 条")
    check("角色成对出现", [m["role"] for m in hist][:4] == ["user", "assistant", "user", "assistant"])

    # ---------- 11. 错误兜底 ----------
    print("\n== 11. 错误兜底 ==")
    real = agent._AGENT

    class Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("模拟模型故障")

    agent._AGENT = Boom()
    t_err, _, _ = agent.answer("随便问", thread_id="smoke-err")
    agent._AGENT = real
    rid_err = agent.get_last_rid()
    check("模型故障不崩、给出提示", t_err.startswith("调用模型失败"), t_err[:40])
    check("错误信息带错误号", "错误号" in t_err and rid_err in t_err, rid_err)
    check("错误日志含堆栈",
          any(l.get("event") == "qa_error" and l.get("rid") == rid_err and l.get("stack")
              for l in log_tail(50)))

    class BadRetr:
        def search(self, *a, **k):
            raise RuntimeError("模拟检索故障")

    saved_r = agent._RETRIEVER
    agent._RETRIEVER = BadRetr()
    fallback = agent.search_policy.func("测试查询")
    agent._RETRIEVER = saved_r
    check("检索失败有兜底文案", "暂时不可用" in fallback, fallback[:40])

    # ---------- 12. 报销材料预检（只读，无副作用） ----------
    print("\n== 12. 报销材料预检 ==")
    bad = agent.check_claim_materials.func("P5-P6", "一线", 3, 2250, "普票", True, False)
    check("预检：普票判退回", "专用发票" in bad and "退回" in bad, bad[:60].replace("\n", " "))
    check("预检：超标提示审批", "超标" in bad and "审批" in bad)
    good = agent.check_claim_materials.func("P5-P6", "一线", 3, 1800, "专票", True, False)
    check("预检：材料齐可报", "可以报销" in good or "可报" in good, good[:60].replace("\n", " "))
    check("预检：只读，没有提交类工具",
          not any(t.name in ("draft_claim", "check_claim_status", "submit_claim")
                  for t in agent.TOOLS), str([t.name for t in agent.TOOLS]))

    # ---------- 12. 评测回归 ----------
    print("\n== 13. 评测回归 ==")
    r3 = run([py, "eval.py"])
    m = re.search(r"Recall@(\d+): (\d+)/(\d+) = ([\d.]+)%", r3.stdout or "")
    recall = float(m.group(4)) if m else -1
    check("eval.py 可跑出结果", bool(m), (m.group(0) if m else "无输出"))
    check("Recall ≥ 60%（当前 k）", recall >= 60, f"{recall}%")

    # ---------- 汇总 ----------
    dt = time.time() - t_all
    print("\n" + "=" * 56)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项，耗时 {dt:.1f}s")
    if FAILED:
        print("失败项：")
        for f in FAILED:
            print("  -", f)
    print("=" * 56)
    print("""
浏览器手动清单（脚本测不了）：
  1. 问「我下周去上海出差，住宿费能报多少？」→ 反问职级
  2. 问「那 P7 呢？」→ 800 元/晚，且知道是上海
  3. 按 F5 刷新 → 聊天记录还在，URL 里 ?tid= 不变，继续追问接得上
  4. 侧栏切「2024」再问第 1 题 → 500
  5. 问「出差可以坐飞机头等舱吗？」→ 明确说没写、建议联系 HR
  6. 点「新会话（清空记忆）」→ 新 tid，问「那 P7 呢」会反问你城市
""")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
