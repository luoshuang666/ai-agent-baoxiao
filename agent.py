"""制度与报销 Agent：ReAct 循环、5 个只读工具、引用整理与证据门控。"""
from __future__ import annotations

import contextvars
import re
import time
import traceback
import uuid
from typing import Any

import pandas as pd
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from config import CFG
from llm import get_llm
from obs import log
from retrieval import Retriever, rewrite_query

# 单次会话的临时状态（Streamlit 单用户演示够用；要并发就改成 contextvar）
SESSION: dict[str, Any] = {"version": None, "cited": [], "rewritten": [], "trace": []}


def _record(step: dict) -> None:
    """记录一次工具调用，供界面轨迹和日志使用。"""
    SESSION.setdefault("trace", []).append(step)


def get_last_trace() -> list[dict]:
    """最近一次 answer() 的决策轨迹（空列表表示本轮没有调工具）。"""
    return list(SESSION.get("trace") or [])


def get_last_rid() -> str:
    """最近一次问答的请求号；报错时用它在日志里定位。"""
    return str(SESSION.get("rid") or "")


def _ground(text: str) -> str:
    """记录工具返回的原文，供证据门控核对答案里的数字。"""
    SESSION.setdefault("grounds", []).append(text)
    return text


_NUM_RE = re.compile(r"\d{2,}")          # 只查两位数以上的数字，避开 [1] 这类引用编号


def _ungrounded(text: str) -> list[str]:
    """返回答案里「工具没给过」的数字；空列表表示全部有依据。"""
    grounds = "\n".join(SESSION.get("grounds") or [])
    missing = []
    for num in _NUM_RE.findall(text):
        if re.fullmatch(r"20\d{2}", num):      # 年份不算事实数字
            continue
        if num not in grounds:
            missing.append(num)
    return missing


def _repair(question: str, text: str, missing: list[str]) -> str:
    """按证据重写一次：只允许使用工具返回的数字。"""
    grounds = "\n\n".join(SESSION.get("grounds") or [])[:4000]
    prompt = (
        "下面这份回答里出现了工具没有提供过的数字：{0}。\n"
        "请只依据【工具返回】重写回答，保留四段格式（直接回答/怎么操作/依据/需要你确认），"
        "所有金额和数字必须来自工具返回，逐条标 [n]；依据里没有的信息就写「制度中没有写明」。\n\n"
        "问题：{1}\n\n【工具返回】\n{2}\n\n原回答：\n{3}\n\n重写："
    ).format("、".join(missing), question, grounds, text)
    try:
        return str(get_llm(0.1).invoke(prompt).content)
    except Exception:
        return text


# ---------- 槽位记忆：记住「职级 / 城市 / 天数 / 金额 / 发票」而不是原始对话 ----------
# 好处：不占上下文、不受窗口限制；用户第 1 轮说职级、第 9 轮问金额也记得住。
_SLOTS: dict[str, dict] = {}          # thread_id -> 槽位
_CURRENT_TID: contextvars.ContextVar[str] = contextvars.ContextVar("current_tid", default="default")

_CITY_TIER = {
    "北京": "一线", "上海": "一线", "广州": "一线", "深圳": "一线",
    "杭州": "新一线", "成都": "新一线", "南京": "新一线", "武汉": "新一线",
    "西安": "新一线", "苏州": "新一线", "重庆": "新一线", "天津": "新一线",
    "合肥": "新一线", "长沙": "新一线", "青岛": "新一线", "宁波": "新一线",
    "郑州": "新一线", "东莞": "新一线", "佛山": "新一线",
}
_CITY_RE = re.compile(
    r"(北京|上海|广州|深圳|杭州|成都|南京|武汉|西安|苏州|重庆|天津|合肥|长沙|青岛|宁波|"
    r"郑州|东莞|佛山|无锡|厦门|福州|济南|大连|昆明|南昌|贵阳|南宁|海口|珠海|中山|惠州|"
    r"温州|常州|嘉兴|绍兴|金华|台州|泉州|烟台|徐州|扬州|淄博|洛阳|唐山|保定|廊坊|汕头|"
    r"湛江|桂林|柳州|三亚|绵阳|宜昌|襄阳|株洲|湘潭|赣州|芜湖)"
)
_LEVEL_RE = re.compile(r"(?<![A-Za-z0-9])(P[3-9]|M[1-3])(?![A-Za-z0-9])")   # 中文紧挨着也能匹配
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_DAYS_RE = re.compile(r"(\d+|[一二两三四五六七八九十])\s*(天|晚)")
_AMOUNT_RE = re.compile(r"(\d{2,5})\s*元")


def update_slots(thread_id: str, question: str) -> dict:
    """从本轮提问里抽取槽位，覆盖旧值（用户改口以最新为准）。返回当前槽位。"""
    s = _SLOTS.setdefault(thread_id, {})
    m = _LEVEL_RE.search(question)
    if m:
        s["level"] = m.group(1)
    elif "M序列" in question:
        s["level"] = "M序列"
    c = _CITY_RE.search(question)
    if c:
        s["city"] = c.group(1)
        s["city_tier"] = _CITY_TIER.get(s["city"], "")
    d = _DAYS_RE.search(question)
    if d:
        raw = d.group(1)
        s["days"] = int(raw) if raw.isdigit() else _CN_NUM.get(raw, 0)
        s["unit"] = d.group(2)
    a = _AMOUNT_RE.search(question)
    if a:
        s["amount"] = int(a.group(1))
    for k in ("专票", "普票"):
        if k in question:
            s["invoice"] = k
    return s


def get_slots(thread_id: str = "default") -> dict:
    """给界面展示「我已经记住了什么」。"""
    return dict(_SLOTS.get(thread_id) or {})


def _slots_system(thread_id: str) -> str | None:
    s = _SLOTS.get(thread_id) or {}
    if not s:
        return None
    parts = []
    if s.get("level"):
        parts.append(f"职级={s['level']}")
    if s.get("city"):
        tier = s.get("city_tier")
        parts.append(f"城市={s['city']}" + (f"（{tier}）" if tier else ""))
    if s.get("days"):
        parts.append(f"住宿={s['days']}{s.get('unit', '天')}")
    if s.get("amount"):
        parts.append(f"实际金额={s['amount']}元")
    if s.get("invoice"):
        parts.append(f"发票={s['invoice']}")
    return ("【已确认信息】" + "；".join(parts) +
            "。用户本轮没提这些时直接沿用，不要再反问；若本轮说法不同，以本轮为准。")


# 会话记忆：按 thread_id 存完整消息（含工具调用），进程内有效；
# 想跨重启保留就换成 langgraph.checkpoint.sqlite.SqliteSaver。
_CHECKPOINTER = MemorySaver()

_RETRIEVER: Retriever | None = None
_STD_DF: pd.DataFrame | None = None
_STD_VERSION: str | None = None      # 标准表文件名里的版本号（如 v2025）
_LEVEL_DF: pd.DataFrame | None = None


def level_df() -> pd.DataFrame:
    """岗位 → 职级档位 对照表（放 data/org/ 下，不进向量库）。"""
    global _LEVEL_DF
    if _LEVEL_DF is None:
        p = CFG.data_dir / "org" / "岗位职级对照表.csv"
        _LEVEL_DF = pd.read_csv(p, encoding="utf-8").fillna("").astype(str) if p.exists() else pd.DataFrame()
    return _LEVEL_DF


def match_level(job_title: str) -> list[dict]:
    """按岗位名称匹配职级档位，匹配不到或有多条就交回给用户确认。

    顺序：精确 → 前缀命中（「测试」→ 测试工程师/测试专家/测试经理）→
    岗位名包含输入（「工程师」→ 最短的测试/研发工程师）→ 输入包含岗位名
    （「高级测试工程师（上海）」→ 最长的那个）。返回多条表示有歧义。
    """
    df = level_df()
    title = (job_title or "").strip()
    if df.empty or not title:
        return []
    exact = df[df["岗位名称"] == title]
    if not exact.empty:
        return exact.to_dict("records")

    prefix = df[df["岗位名称"].apply(lambda s: s.startswith(title))]
    if not prefix.empty:
        return prefix.to_dict("records")

    sup = df[df["岗位名称"].apply(lambda s: title in s)]     # 表里的名字更长
    if not sup.empty:
        shortest = sup["岗位名称"].str.len().min()
        return sup[sup["岗位名称"].str.len() == shortest].to_dict("records")

    sub = df[df["岗位名称"].apply(lambda s: s in title)]     # 用户说的更长
    if not sub.empty:
        longest = sub["岗位名称"].str.len().max()
        return sub[sub["岗位名称"].str.len() == longest].to_dict("records")
    return []


def retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


def std_df() -> pd.DataFrame:
    global _STD_DF, _STD_VERSION
    if _STD_DF is None:
        files = sorted(CFG.data_dir.glob("*.xlsx")) + sorted(CFG.data_dir.glob("*.csv"))
        if not files:
            _STD_DF = pd.DataFrame()
        elif files[0].suffix.lower() == ".csv":
            _STD_DF = pd.read_csv(files[0], encoding="utf-8").fillna("").astype(str)
        else:
            _STD_DF = pd.read_excel(files[0]).fillna("").astype(str)
        import re

        m = re.search(r"_v(\d{4})", files[0].name) if files else None
        _STD_VERSION = m.group(1) if m else None
    return _STD_DF


# ------------------------- 工具 1：制度检索（含一次重写兜底） -------------------------
@tool
def search_policy(query: str) -> str:
    """检索公司制度原文。回答制度、流程、标准、发票类问题前必须先调用本工具。"""
    t0 = time.time()
    version = SESSION.get("version")
    try:
        r = retriever()
        docs, best = r.search(query, version=version)
    except Exception as exc:
        log("tool_error", SESSION.get("rid", ""), tool="search_policy",
            error=type(exc).__name__, detail=str(exc)[:200])
        return "（检索服务暂时不可用，请稍后重试；如持续失败请联系 HR）"
    rewritten = None

    if best < CFG.relevance_th:
        new_q = rewrite_query(query)
        SESSION["rewritten"].append({"原问题": query, "重写后": new_q, "得分": round(best, 3)})
        extra, _ = r.search(new_q, version=version)
        seen = {t for t, _, _ in docs}
        docs += [d for d in extra if d[0] not in seen]
        rewritten = {"from": query, "to": new_q}

    _record({"tool": "search_policy", "args": {"query": query},
             "ms": int((time.time() - t0) * 1000), "hits": len(docs),
             "best": round(best, 3), "rewritten": rewritten})

    if not docs:
        return "（知识库中未检索到相关制度条款）"

    blocks = []
    for text, meta, _ in docs:
        n = len(SESSION["cited"]) + 1
        SESSION["cited"].append({
            "n": n,
            "source": meta.get("source"),
            "clause": meta.get("clause_path"),
            "quote": text[:220],
        })
        blocks.append(f"[{n}] {text}\n（来源：{meta.get('source')} · {meta.get('clause_path')}）")
    return _ground("\n\n".join(blocks))


# ------------------------- 工具 2：标准数值精确查询 -------------------------
@tool
def lookup_standard(level: str, city_tier: str) -> str:
    """查询报销标准数值表，返回住宿上限、餐补、市内交通额度。

    Args:
        level: 职级，如 P5 / P6 / P7 / M1
        city_tier: 城市级别，如 一线 / 新一线 / 二线
    """
    t0 = time.time()
    try:
        df = std_df()
    except Exception as exc:
        log("tool_error", SESSION.get("rid", ""), tool="std_table",
            error=type(exc).__name__, detail=str(exc)[:200])
        return "（标准表暂时读不到，请稍后重试）"
    if df.empty:
        _record({"tool": "lookup_standard", "args": {"level": level, "city_tier": city_tier},
                 "ms": int((time.time() - t0) * 1000), "hits": 0, "best": None,
                 "rewritten": None, "note": "标准表为空"})
        return "（未找到报销标准表）"

    want = SESSION.get("version")
    if want and _STD_VERSION and want != _STD_VERSION:
        # 版本过滤只作用于检索；标准表是单一文件，必须显式拦住，
        # 否则切到 2024 时会拿 2025 的表答出 600，和制度正文的 500 打架。
        _record({"tool": "lookup_standard", "args": {"level": level, "city_tier": city_tier},
                 "ms": int((time.time() - t0) * 1000), "hits": 0, "best": None,
                 "rewritten": None, "note": f"表版本 {_STD_VERSION} ≠ 会话 {want}，已拒绝"})
        return (f"（注意：报销标准速查表只有 {_STD_VERSION} 版，当前会话是 {want} 版。"
                "表中金额不适用于当前版本，请改用 search_policy 检索该版本制度正文后再回答。）")

    mask = df.apply(
        lambda r: level.strip() in r.get("职级", "") and city_tier.strip() in r.get("城市级别", ""),
        axis=1,
    )
    hit = df[mask]
    if hit.empty:
        _record({"tool": "lookup_standard", "args": {"level": level, "city_tier": city_tier},
                 "ms": int((time.time() - t0) * 1000), "hits": 0, "best": None,
                 "rewritten": None, "note": "表里没有该组合"})
        return f"（标准表中没有 职级={level} 城市={city_tier} 的记录，可用值见标准表）"

    _record({"tool": "lookup_standard", "args": {"level": level, "city_tier": city_tier},
             "ms": int((time.time() - t0) * 1000), "hits": len(hit),
             "best": None, "rewritten": None, "note": "命中标准表"})
    rows = ["；".join(f"{c}={v}" for c, v in row.items()) for _, row in hit.iterrows()]
    return _ground("报销标准（精确查表）：\n" + "\n".join(rows))


TOOLS = [search_policy, lookup_standard]


# ------------------------- 工具 3：算账（住宿可报 / 超支） -------------------------
@tool
def calc_reimbursement(level: str, city_tier: str, nights: int,
                       actual_per_night: int = 0) -> str:
    """按标准表算住宿可报金额和超支：用户给了天数或金额时调用，金额不要自己心算。

    Args:
        level: 职级档位，如 P5-P6 / P7及以上 / M序列 / P4及以下
        city_tier: 一线 / 新一线 / 二线
        nights: 住宿晚数
        actual_per_night: 实际每晚价格（元），不知道就传 0
    """
    t0 = time.time()
    try:
        df = std_df()
    except Exception as exc:
        log("tool_error", SESSION.get("rid", ""), tool="std_table",
            error=type(exc).__name__, detail=str(exc)[:200])
        return "（标准表暂时读不到，请稍后重试）"
    if df.empty:
        return "（未找到报销标准表）"
    want = SESSION.get("version")
    if want and _STD_VERSION and want != _STD_VERSION:
        return (f"（标准速查表只有 {_STD_VERSION} 版，当前会话是 {want} 版；"
                "请改用 search_policy 查该版本制度正文后再答复。）")

    mask = df.apply(
        lambda r: level.strip() in r.get("职级", "") and city_tier.strip() in r.get("城市级别", ""),
        axis=1,
    )
    hit = df[mask]
    if hit.empty:
        return f"（标准表中没有 职级={level} 城市={city_tier} 的记录）"

    row = hit.iloc[0]
    cap = int(float(row.get("住宿上限", 0) or 0))
    nights = max(1, int(nights))
    limit_total = cap * nights
    lines = [f"住宿上限={cap}元/晚；{nights}晚上限={cap}×{nights}={limit_total}元"]

    if actual_per_night and actual_per_night > 0:
        actual_total = int(actual_per_night) * nights
        over = max(0, int(actual_per_night) - cap) * nights
        lines.append(f"实际={actual_per_night}×{nights}={actual_total}元")
        if over:
            lines.append(f"超支={over}元（自理或事前提交超标申请获批）")
            lines.append(f"结论：可报{limit_total}元，超出{over}元需自理")
        else:
            lines.append("结论：未超标，据实报销")
    else:
        lines.append(f"结论：最多可报{limit_total}元")

    _record({"tool": "calc_reimbursement",
             "args": {"level": level, "city_tier": city_tier, "nights": nights,
                      "actual_per_night": actual_per_night},
             "ms": int((time.time() - t0) * 1000), "hits": 1, "best": None,
             "rewritten": None, "note": f"上限 {limit_total} 元"})
    return _ground("算账结果：\n" + "\n".join(lines))


TOOLS.append(calc_reimbursement)


# ------------------------- 工具 3：岗位 → 职级档位 -------------------------
@tool
def lookup_level(job_title: str) -> str:
    """根据岗位名称查对应的职级档位（P4及以下 / P5-P6 / P7及以上 / M序列）。

    用户不知道自己几级、但说了岗位名称（如「高级测试工程师」）时先用本工具，
    再把得到的档位传给 lookup_standard。禁止自己猜职级。
    """
    t0 = time.time()
    hits = match_level(job_title)
    if not hits:
        _record({"tool": "lookup_level", "args": {"job_title": job_title},
                 "ms": int((time.time() - t0) * 1000), "hits": 0, "best": None,
                 "rewritten": None, "note": "对照表无匹配，需人工确认"})
        return ("（岗位职级对照表里没有匹配项。请提供正式岗位名称，如「测试工程师」；"
                "或直接告诉我职级：P4及以下 / P5-P6 / P7及以上 / M序列）")

    _record({"tool": "lookup_level", "args": {"job_title": job_title},
             "ms": int((time.time() - t0) * 1000), "hits": len(hits), "best": None,
             "rewritten": None, "note": "命中岗位职级对照表"})
    rows = [f"岗位={h['岗位名称']} → 职级档位={h['职级档位']}（{h.get('说明', '')}）" for h in hits]
    return _ground("岗位职级映射：\n" + "\n".join(rows))


TOOLS.append(lookup_level)


# ------------------------- 工具 4：报销材料预检（只读，无副作用） -------------------------
@tool
def check_claim_materials(level: str, city_tier: str, nights: int, amount: float,
                          invoice_type: str = "专票", has_itinerary: bool = True,
                          has_approval: bool = False) -> str:
    """报销材料预检（只读）：判断这笔住宿能不能报、缺什么、会不会被退回。

    不提交、不写库、不改任何状态。用户问「能不能报 / 帮我报」时调用本工具。

    Args:
        level: 职级档位，如 P5-P6 / P7及以上 / M序列
        city_tier: 一线 / 新一线 / 二线
        nights: 住宿晚数
        amount: 住宿总金额（元）
        invoice_type: 发票类型：专票 / 普票 / 无
        has_itinerary: 是否有机票或网约车行程单
        has_approval: 超标是否已有事前审批记录
    """
    t0 = time.time()
    try:
        df = std_df()
    except Exception as exc:
        log("tool_error", SESSION.get("rid", ""), tool="check_claim_materials",
            error=type(exc).__name__, detail=str(exc)[:200])
        return "（标准表暂时读不到，请稍后重试）"

    nights = max(1, int(nights))
    cap = 0
    if not df.empty:
        mask = df.apply(
            lambda r: level.strip() in r.get("职级", "") and city_tier.strip() in r.get("城市级别", ""),
            axis=1,
        )
        hit = df[mask]
        if not hit.empty:
            cap = int(float(hit.iloc[0].get("住宿上限", 0) or 0))

    per_night = amount / nights if nights else 0
    over = max(0, per_night - cap) * nights if cap else 0

    ok, warn, bad = [], [], []
    if cap:
        if over == 0:
            ok.append(f"住宿在限额内（{cap}元/晚 × {nights}晚 = {cap * nights}元）")
        else:
            warn.append(f"住宿超标：{per_night:.0f}元/晚 超上限 {cap}元，共超 {over:.0f}元，需自理")
            if not has_approval:
                bad.append("超标缺事前审批记录，报销时会被退回")
    else:
        warn.append(f"标准表里没有 {level}/{city_tier} 的组合，需人工确认限额")

    if invoice_type != "专票":
        bad.append(f"住宿发票为「{invoice_type}」，住宿必须增值税专用发票，否则验真退回")
    if has_itinerary is False:
        warn.append("若本次含机票或网约车，需补电子发票 + 行程单")

    if bad:
        verdict = "会被退回，先补齐再提交"
    elif over:
        verdict = "可报限额内部分，超出部分自理"
    else:
        verdict = "材料齐、额度内，可以报销"

    lines = [f"报销材料预检（只读，不提交）：{verdict}"]
    lines += [f"满足：{x}" for x in ok]
    lines += [f"风险：{x}" for x in warn]
    lines += [f"缺项：{x}" for x in bad]
    lines.append("下一步：补齐上述材料后，到 OA「费用报销」自行提交（系统不代提交）。")

    _record({"tool": "check_claim_materials",
             "args": {"level": level, "city_tier": city_tier, "nights": nights,
                      "amount": amount, "invoice_type": invoice_type,
                      "has_itinerary": has_itinerary, "has_approval": has_approval},
             "ms": int((time.time() - t0) * 1000), "hits": 1, "best": None,
             "rewritten": None, "note": f"{verdict}"})
    return _ground("\n".join(lines))


TOOLS.append(check_claim_materials)

SYSTEM = """你是公司 HR 制度与报销助手，回答员工关于差旅、报销、发票、审批流程的问题。

必须遵守：
1. 制度类问题先调用 search_policy；金额标准类问题调用 lookup_standard。
   用户给了天数或实际金额时，必须调用 calc_reimbursement 算账，金额不许心算。
2. 只使用工具返回的原文和算式作答，每条结论标注来源编号，如 [1]。
3. 检索结果里没有的内容，直接说「制度中没有写明」，并建议联系 HR 确认，不允许推测。
4. 回答格式固定，用加粗小标题，顺序不能变：
   **直接回答**：用一句带数字的结论回答，必须结合用户给的条件，不要只复述条款。
      例：「你能报 600/晚，3 晚共 1800；实际 750×3=2250，超出 450 元自理（想报需事前审批）。」
      不要只写「住宿上限 600 元/晚 [1]」这种复述条款的话。
   **怎么操作**：≤3 条具体动作，要能直接照做（订什么价位、提前几天申请、开什么票、寄什么材料）。
   **依据**：≤3 条，每条 ≤40 字，只引关键句和 [n]，不要整段抄制度。
   **需要你确认**：≤2 项，只问真正影响金额的信息；没有就写「无」。
   全文 ≤300 字。能确定多少说多少，不要一次把所有档位/条款都倒出来。
5. 有金额时必须给算式，且算式里的数要来自 calc_reimbursement 的返回。
6. 用户没说明职级或城市时，只问缺的那一个，不要猜，也不要顺手把三档全列出来。
7. 用户不知道自己职级、但说了岗位名称（如「我是高级测试工程师」）时：
   先调用 lookup_level 把岗位映射成职级档位，再查表；映射不到或有歧义时只列候选，禁止编。
8. 若消息里带有【已确认信息】（职级/城市/天数/金额），视为用户已提供，直接使用，
   并在结论里点明（如「上海（一线）」）；不要反问，也不要列其他城市/职级的档位。
9. 用户问「能不能报 / 材料够不够 / 帮我报」时，调用 check_claim_materials 做只读预检，
   列出缺项和风险；你没有提交权限，禁止说「已提交 / 已办理」，最后提醒到 OA 自行提交。"""

def _trim_history(state: dict) -> dict:
    """只把最近 N 轮原文送给模型，控制 token；完整历史仍留在 checkpointer 里。

    一律从 human 消息处下刀，保证「AI 发起的 tool_calls」和「工具返回」
    不会被切成两半（OpenAI 兼容接口会报 400：tool 消息缺少对应的 tool_calls）。
    额外在最前面插入【已确认信息】，让窗口外的槽位也能被记住。
    """
    msgs = state.get("messages") or []
    limit = max(1, CFG.history_turns)
    humans = [i for i, m in enumerate(msgs) if getattr(m, "type", "") == "human"]
    trimmed = msgs if (not humans or len(humans) <= limit) else msgs[humans[-limit]:]

    hint = _slots_system(_CURRENT_TID.get())
    if hint:
        trimmed = [SystemMessage(content=hint)] + list(trimmed)
    return {"llm_input_messages": trimmed}


def _build_agent():
    """构建 ReAct Agent，兼容两代 API。

    LangGraph 0.2~0.6：langgraph.prebuilt.create_react_agent(llm, tools, prompt=...)
    LangChain 1.x    ：langchain.agents.create_agent(llm, tools, system_prompt=...)
    其余写法（ChatOpenAI / @tool / Chroma）两代一致，所以只在这一个入口做兼容。
    """
    try:
        from langgraph.prebuilt import create_react_agent

        agent = create_react_agent(
            get_llm(), TOOLS, prompt=SYSTEM,
            checkpointer=_CHECKPOINTER, pre_model_hook=_trim_history,
        )
        print("[agent] 入口：langgraph.prebuilt.create_react_agent（0.3 世代，已挂 checkpointer）")
        return agent
    except Exception as exc:                      # 1.x 里该入口可能已被移除
        from langchain.agents import create_agent

        agent = create_agent(
            get_llm(), TOOLS, system_prompt=SYSTEM,
            checkpointer=_CHECKPOINTER, pre_model_hook=_trim_history,
        )
        print(f"[agent] 入口：langchain.agents.create_agent（1.x 世代，原因 {type(exc).__name__}）")
        return agent


_AGENT = _build_agent()


# ------------------------- 引用整理（finalize） -------------------------
def finalize(text: str, cited: list[dict]) -> tuple[str, list[dict]]:
    """去重引用 + 只保留答案里真正引用到的编号。"""
    used = {i for i in range(1, len(cited) + 1) if f"[{i}]" in text}
    kept = [c for c in cited if c["n"] in used] or cited[:2]
    for i, c in enumerate(kept, 1):
        c["n"] = i
    return text, kept


def answer(question: str, version: str | None = None,
           thread_id: str = "default") -> tuple[str, list[dict], list[dict]]:
    """一次问答：清空引用 → 按 thread_id 带上前几轮 → 跑 Agent → 整理引用。

    多轮记忆由 checkpointer 负责：同一 thread_id 的调用会自动接上历史消息，
    包括上一轮的工具调用与检索原文；CFG.history_turns 控制真正送进模型的轮数。
    """
    SESSION["version"] = version
    SESSION["cited"] = []
    SESSION["rewritten"] = []
    SESSION["trace"] = []
    SESSION["grounds"] = []
    rid = uuid.uuid4().hex[:8]
    SESSION["rid"] = rid
    t0 = time.time()

    update_slots(thread_id, question)            # 先把本轮提到的职级/城市等记下来
    token = _CURRENT_TID.set(thread_id)          # 让 pre_model_hook 知道是哪个会话
    cfg = {"recursion_limit": 12, "configurable": {"thread_id": thread_id}}
    ok = True
    try:
        resp = _AGENT.invoke({"messages": [("user", question)]}, config=cfg)
        text = str(resp["messages"][-1].content)
    except Exception as exc:                      # 演示环境不因一次失败崩掉
        ok = False
        log("qa_error", rid, thread=thread_id, ver=version, q=question[:60],
            error=type(exc).__name__, stack=traceback.format_exc()[-1500:])
        text = f"调用模型失败（错误号 {rid}），请把这个号发给管理员。"
    finally:
        _CURRENT_TID.reset(token)

    text, cites = finalize(text, SESSION["cited"])

    # ---- 证据门控：答案里的两位数以上数字必须来自工具返回 ----
    missing = _ungrounded(text) if ok else []
    repaired = False
    if missing:
        fixed = _repair(question, text, missing)
        fixed, cites2 = finalize(fixed, SESSION["cited"])
        if not _ungrounded(fixed):
            text, cites, repaired = fixed, cites2, True
        else:
            log("grounded_fail", rid, missing=_ungrounded(fixed)[:5])
            text = ("（无法把结论对上制度依据，暂不回答。请补充信息，或联系 HR 确认。）\n\n"
                    f"错误号 {rid}")
            cites = []

    _record({"tool": "generate", "args": {}, "ms": int((time.time() - t0) * 1000),
             "hits": len(cites), "best": None, "rewritten": None, "note": "生成最终回答"})
    log("qa_ok" if ok else "qa_error_done", rid, thread=thread_id, ver=version,
        ms=int((time.time() - t0) * 1000), q_len=len(question),
        tools=[s.get("tool") for s in SESSION["trace"]],
        cites=len(cites), rewrites=len(SESSION["rewritten"]),
        grounded=not bool(_ungrounded(text)) and ok, repaired=repaired,
        missing=_ungrounded(text)[:5] if missing else [])
    return text, cites, SESSION["rewritten"]


def load_history(thread_id: str) -> list[dict]:
    """从 checkpointer 读出某个会话的历史，用于刷新页面后恢复聊天窗。"""
    try:
        snap = _AGENT.get_state({"configurable": {"thread_id": thread_id}})
    except Exception:
        return []
    out: list[dict] = []
    for m in (snap.values.get("messages") or []):
        role = {"human": "user", "ai": "assistant"}.get(getattr(m, "type", ""))
        content = getattr(m, "content", "")
        if role and content and not getattr(m, "tool_calls", None):   # 跳过「边调工具边说话」的中间消息
            out.append({"role": role, "content": str(content)})
    return out
