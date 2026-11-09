"""Streamlit 前端：聊天 + 引用卡片 + 决策轨迹面板。"""
from __future__ import annotations

import uuid

import streamlit as st

from agent import answer, get_last_trace, get_slots, load_history
from config import CFG

st.set_page_config(page_title="制度与报销助手", page_icon=None)
st.title("制度与报销智能问答")
st.caption("Chroma 向量检索 + BM25 关键词召回（RRF 融合），答案标注制度出处")

# 会话 ID 放 URL 里：刷新页面还是同一个 thread_id，记忆不丢
if "thread_id" not in st.session_state:
    st.session_state.thread_id = st.query_params.get("tid") or uuid.uuid4().hex[:8]
    st.query_params["tid"] = st.session_state.thread_id

if "history" not in st.session_state:
    # 刷新后从 checkpointer 恢复聊天记录（只在页面状态为空时读一次）
    st.session_state.history = load_history(st.session_state.thread_id)

with st.sidebar:
    st.subheader("演示控制")
    version = st.radio("适用的制度版本", ["2025", "2024"], index=0,
                       help="切换版本可验证「同名条款不同标准」是否被正确区分")
    show_debug = st.checkbox("显示决策轨迹与检索调试信息", value=True)
    if st.button("新会话（清空记忆）"):
        st.session_state.thread_id = uuid.uuid4().hex[:8]
        st.query_params["tid"] = st.session_state.thread_id
        st.session_state.history = []
        st.rerun()
    st.caption(f"会话 ID：`{st.session_state.thread_id}` · 记忆窗口 {CFG.history_turns} 轮")
    slots = get_slots(st.session_state.thread_id)
    if slots:
        label = {"level": "职级", "city": "城市", "city_tier": "城市级别",
                 "days": "天数", "unit": "单位", "amount": "金额", "invoice": "发票"}
        st.caption("已记住：" + " · ".join(f"{label.get(k, k)}={v}" for k, v in slots.items()))
    st.divider()
    st.caption("语料：差旅制度 / 发票规定 / 报销流程 / 驳回FAQ / 标准速查表")
    st.caption("能力边界：只做制度问答、算账、材料预检；不代提交、不改系统数据。")


def _trace_rows(trace: list[dict]) -> list[dict]:
    rows = []
    for s in trace:
        args = "；".join(f"{k}={v}" for k, v in (s.get("args") or {}).items()) or "—"
        note = s.get("note") or f"命中 {s.get('hits', 0)} 条"
        if s.get("best") is not None:
            note += f"，最高相关 {s['best']}"
        if s.get("rewritten"):
            note += f"，已重写 → {s['rewritten']['to']}"
        rows.append({"步骤": s.get("tool", ""), "参数": args,
                     "耗时(ms)": s.get("ms", 0), "结果": note})
    return rows


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace") and show_debug:
            with st.expander(f"本轮决策轨迹（{len(msg['trace'])} 步）"):
                st.table(_trace_rows(msg["trace"]))


for msg in st.session_state.history:
    _render_message(msg)

question = st.chat_input("例如：我是 P6，去上海出差三天，住宿能报多少？")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("检索制度条款…"):
            text, cites, _ = answer(question, version, st.session_state.thread_id)
            trace = get_last_trace()

        st.markdown(text)

        if cites:
            with st.expander(f"依据片段（{len(cites)} 条）"):
                for c in cites:
                    st.markdown(f"**[{c['n']}] {c['source']} · {c['clause']}**")
                    st.caption(c["quote"])

        if show_debug and trace:
            with st.expander(f"本轮决策轨迹（{len(trace)} 步）"):
                st.table(_trace_rows(trace))

    st.session_state.history.append({"role": "assistant", "content": text, "trace": trace})
