"""data/ → 条款级切分（注入章节路径）→ Chroma 入库（MD5 增量）。

用法：
    python ingest.py            # 增量：只重建改动过的文件
    python ingest.py --force    # 全量重建
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from config import CFG

HEAD = re.compile(r"^(#{1,6})\s+(.*)$")                       # Markdown 标题
CLAUSE = re.compile(r"^(\d+(?:\.\d+)+|\d+[、.．])\s*(\S.*)$")  # 4.2 住宿标准 / 1、总则
VERSION = re.compile(r"_v(\d{4})")                            # 文件名里的版本号
MAXLEN = 480
MANIFEST = CFG.data_dir.parent / "storage" / "manifest.json"


def md5_of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _split_long(body: str, limit: int = MAXLEN) -> list[str]:
    """超长条款按句子再切，避免一条 chunk 过胖。"""
    if len(body) <= limit:
        return [body]
    parts, buf = [], ""
    for sent in re.split(r"(?<=[。！？；\n])", body):
        if buf and len(buf) + len(sent) > limit:
            parts.append(buf.strip())
            buf = sent
        else:
            buf += sent
    if buf.strip():
        parts.append(buf.strip())
    return [p for p in parts if p]


def split_md(text: str, source: str, version: str | None,
             doc_type: str = "policy") -> list[dict]:
    """按「标题 → 条款号」切块，并把层级路径写进 chunk 首行。

    路径写入块首行：让 embedding 能区分条款归属，引用时也能显示具体位置。
    """
    chunks: list[dict] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    clause: str | None = None

    def path_now() -> str:
        parts = [t for _, t in stack]
        if clause:
            parts.append(clause)
        return " > ".join(parts) or Path(source).stem

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        prefix = path_now()
        for i, part in enumerate(_split_long(body)):
            chunks.append({
                "text": f"【{prefix}】\n{part}",
                "metadata": {
                    "source": source,
                    "clause_path": prefix,
                    "doc_type": doc_type,
                    "version": version or "all",
                    "part": i,
                },
            })

    for raw in text.splitlines():
        line = raw.rstrip()
        head = HEAD.match(line)
        clause_match = None if head else CLAUSE.match(line.strip())

        if head:
            flush()
            level, title = len(head.group(1)), head.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            clause = None
        elif clause_match:
            flush()
            clause = f"{clause_match.group(1)} {clause_match.group(2).strip()}"
        else:
            buf.append(line)
    flush()
    return chunks


def rows_to_chunks(df: pd.DataFrame, source: str, version: str) -> list[dict]:
    """标准表一行一条 chunk，职级/城市级别单独存进 metadata 供精确查询。"""
    df = df.fillna("").astype(str)
    cols = list(df.columns)
    out = []
    for _, row in df.iterrows():
        kv = "；".join(f"{c}={row[c]}" for c in cols)
        level, city = row.get("职级", ""), row.get("城市级别", "")
        out.append({
            "text": f"【报销标准速查表 > {level} / {city}】\n{kv}",
            "metadata": {
                "source": source,
                "clause_path": "报销标准速查表",
                "doc_type": "table",
                "version": version,
                "level": level,
                "city_tier": city,
                "part": 0,
            },
        })
    return out


def chunks_of(p: Path) -> list[dict]:
    """按后缀分派解析器（重依赖延迟导入，切分逻辑可单独测试）。"""
    import pandas as pd
    v = VERSION.search(p.name)
    version = v.group(1) if v else "all"
    if p.suffix.lower() == ".md":
        return split_md(p.read_text(encoding="utf-8"), p.name, version)
    if p.suffix.lower() == ".xlsx":
        return rows_to_chunks(pd.read_excel(p), p.name, version)
    if p.suffix.lower() == ".csv":
        return rows_to_chunks(pd.read_csv(p, encoding="utf-8"), p.name, version)
    return []


def build_index(force: bool = False) -> None:
    """建库/增量更新（需要 langchain-chroma 与两个 API Key）。"""
    from langchain_chroma import Chroma

    from llm import get_embeddings

    if force and CFG.chroma_dir.exists():
        import shutil

        shutil.rmtree(CFG.chroma_dir, ignore_errors=True)
        print(f"[force] 已清空旧库 {CFG.chroma_dir}")

    CFG.chroma_dir.mkdir(parents=True, exist_ok=True)
    CFG.data_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, str] = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    try:
        vs = Chroma(
            collection_name=CFG.collection,
            persist_directory=str(CFG.chroma_dir),
            embedding_function=get_embeddings(),
        )
    except Exception as exc:
        print(f"[!] 打开向量库失败：{type(exc).__name__}: {exc}")
        print(f"    当前库路径：{CFG.chroma_dir}")
        print("    常见原因与处理：")
        print("    1) 上次运行中断留下了残缺数据库 → 删掉整个 storage 目录后重跑")
        print("    2) 杀毒软件 / OneDrive 正占用 sqlite 文件 → 换个目录：")
        print("       在 .env 里加一行  CHROMA_DIR=F:\\chroma_db  （纯英文、无空格、非同步盘）")
        print("    3) 路径含中文或空格导致 sqlite 打不开 → 同上，用 CHROMA_DIR 指向英文路径")
        raise

    files = sorted(
        list(CFG.data_dir.glob("*.md"))
        + list(CFG.data_dir.glob("*.xlsx"))
        + list(CFG.data_dir.glob("*.csv"))
    )
    if not files:
        print(f"[!] {CFG.data_dir} 里没有 .md / .xlsx / .csv 文件")
        return

    total = 0
    for p in files:
        digest = md5_of(p)
        if not force and manifest.get(p.name) == digest:
            print(f"  skip  {p.name}（未改动）")
            continue

        old_ids = vs.get(where={"source": p.name})["ids"]
        if old_ids:
            vs.delete(ids=old_ids)          # 先删旧 chunk，保证幂等

        chunks = chunks_of(p)
        if not chunks:
            print(f"  warn  {p.name} 解析出 0 个 chunk")
            continue

        vs.add_texts(
            texts=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[f"{p.name}::{i}::{digest[:8]}" for i in range(len(chunks))],
        )
        manifest[p.name] = digest
        total += len(chunks)
        print(f"  index {p.name} → {len(chunks)} chunks")

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] 本次写入 {total} 个 chunk，当前集合共 {vs._collection.count()} 个")


if __name__ == "__main__":
    build_index(force="--force" in sys.argv)
