"""可观测性：结构化日志（JSONL）+ 请求号。

每次问答写一条 qa_ok / qa_error；工具内部异常写 tool_error。
用户界面上只暴露 8 位错误号，堆栈只进日志，不发到前端。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from config import CFG

LOCK = threading.Lock()
LOG_PATH: Path = CFG.data_dir.parent / "storage" / "logs" / "app.jsonl"


def log(event: str, rid: str = "", **fields) -> None:
    """追加一条 JSONL 日志；字段按需传，写失败不能影响主流程。"""
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, "rid": rid}
    rec.update(fields)
    try:
        with LOCK:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def tail(n: int = 20) -> list[dict]:
    """读最近 n 条，调试和测试用。"""
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()[-n:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
