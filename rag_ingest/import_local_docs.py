"""导入本地排障文档到 Qdrant，并记录 parent 文档映射。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

# 兼容直接从仓库根目录执行：python rag_ingest/import_local_docs.py
_ROOT_DIR = Path(__file__).resolve().parents[1]
_SRC_DIR = _ROOT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from db.db_store import ChatDBStore
from qdrant import QdrantStore

_DEFAULT_SUFFIXES = [".md", ".txt", ".doc", ".docx", ".log", ".rst"]
_DEFAULT_CHUNK_SIZE = 900
_DEFAULT_CHUNK_OVERLAP = 120


# 方法注释（业务）:
# - 入参：`raw`(str)=命令行后缀配置，逗号分隔。
# - 出参：`list[str]`=标准化后缀列表（统一小写且带点前缀）。
# - 方法逻辑：按逗号切分、去空白并补齐前导点，最终得到可用于匹配的后缀集合。
def _normalize_suffixes(raw: str) -> list[str]:
    values = [item.strip().lower() for item in str(raw or "").split(",") if item.strip()]
    return [item if item.startswith(".") else f".{item}" for item in values]


# 方法注释（业务）:
# - 入参：`source_dir`(Path)=文档根目录；`suffixes`(list[str])=允许导入后缀。
# - 出参：`list[Path]`=待导入文件路径列表。
# - 方法逻辑：递归扫描目录，按后缀过滤并按路径排序，保证导入顺序稳定。
def _collect_source_files(source_dir: Path, suffixes: list[str]) -> list[Path]:
    allowed = {item.lower() for item in suffixes}
    rows: list[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        rows.append(path)
    rows.sort(key=lambda item: str(item))
    return rows


# 方法注释（业务）:
# - 入参：`path`(Path)=文档路径。
# - 出参：`str`=读取到的文本；失败或二进制内容返回空字符串。
# - 方法逻辑：优先 utf-8，回退 gb18030；发现 NUL 字节视为二进制跳过。
def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if b"\x00" in raw:
        return ""
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore").strip()


# 方法注释（业务）:
# - 入参：`text`(str)=完整文档文本；`chunk_size`(int)=切片长度；`chunk_overlap`(int)=切片重叠长度。
# - 出参：`list[str]`=按顺序切分后的 chunk 文本列表。
# - 方法逻辑：采用滑动窗口切分文本，确保长文可检索并通过 overlap 保留上下文连续性。
def _split_chunks(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    content = str(text or "").strip()
    if not content:
        return []

    size = max(100, int(chunk_size))
    overlap = max(0, min(int(chunk_overlap), size - 1))
    step = max(1, size - overlap)

    rows: list[str] = []
    start = 0
    length = len(content)
    while start < length:
        end = min(length, start + size)
        chunk = content[start:end].strip()
        if chunk:
            rows.append(chunk)
        if end >= length:
            break
        start += step
    return rows


# 方法注释（业务）:
# - 入参：`doc_path`(Path)=原始文档路径；`text`(str)=文档全文；`chunk_size`/`chunk_overlap`=切分参数。
# - 出参：`tuple[str, list[dict[str, Any]]]`=(parent_id, qdrant_upsert_items)。
# - 方法逻辑：每篇原始文档生成唯一 parent_id，并把 parent_id/path 写入每个 chunk 的 payload。
def _build_upsert_items(
    *,
    doc_path: Path,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[str, list[dict[str, Any]]]:
    parent_id = uuid4().hex
    chunks = _split_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    abs_path = str(doc_path.resolve())

    items: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        items.append(
            {
                "id": uuid4().hex,
                "text": chunk,
                "payload": {
                    "parent_id": parent_id,
                    "path": abs_path,
                    "chunk_index": idx,
                    "file_name": doc_path.name,
                },
            }
        )
    return parent_id, items


# 方法注释（业务）:
# - 入参：`source_dir`(str|Path|None)=文档目录；`suffixes`(list[str]|None)=后缀白名单；`chunk_size`/`chunk_overlap`=切分参数。
# - 出参：`dict[str, Any]`=导入统计结果。
# - 方法逻辑：逐文件读取与切分，先将 parent_id/path upsert 到 rag_document，再将 chunk（带 parent_id）写入 Qdrant。
def run_import(
    *,
    source_dir: str | Path | None = None,
    suffixes: list[str] | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    target_dir = Path(source_dir) if source_dir else (_ROOT_DIR / "rag_ingest" / "source_docs")
    normalized_suffixes = list(suffixes or _DEFAULT_SUFFIXES)
    files = _collect_source_files(target_dir, normalized_suffixes)

    db_store = ChatDBStore()
    qdrant_store = QdrantStore()

    file_count = 0
    chunk_count = 0
    db_rows = 0
    skipped_files: list[str] = []

    for doc_path in files:
        text = _read_text(doc_path)
        if not text:
            skipped_files.append(str(doc_path))
            continue

        parent_id, items = _build_upsert_items(
            doc_path=doc_path,
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not items:
            skipped_files.append(str(doc_path))
            continue

        db_rows += db_store.upsert_rag_document(parent_id=parent_id, path=str(doc_path.resolve()))
        qdrant_store.upsert_texts(items)

        file_count += 1
        chunk_count += len(items)

    return {
        "source_dir": str(target_dir),
        "suffixes": normalized_suffixes,
        "files_found": len(files),
        "files_ingested": file_count,
        "chunks_ingested": chunk_count,
        "rag_document_rows_affected": db_rows,
        "db_enabled": bool(db_store.enabled),
        "qdrant_collection": qdrant_store.config.collection_name,
        "skipped_files": skipped_files,
    }


# 方法注释（业务）:
# - 入参：无（从命令行读取）。
# - 出参：`None`。
# - 方法逻辑：解析参数后执行导入，并将结构化结果打印为 JSON。
def main() -> None:
    parser = argparse.ArgumentParser(description="导入本地文档到 Qdrant，并写入 rag_document 映射表")
    parser.add_argument(
        "--source-dir",
        default="",
        help="文档目录，默认使用 rag_ingest/source_docs",
    )
    parser.add_argument(
        "--suffixes",
        default=",".join(_DEFAULT_SUFFIXES),
        help="允许导入的后缀，逗号分隔",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=_DEFAULT_CHUNK_SIZE,
        help="chunk 字符长度",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=_DEFAULT_CHUNK_OVERLAP,
        help="chunk 重叠字符长度",
    )
    args = parser.parse_args()

    result = run_import(
        source_dir=args.source_dir or None,
        suffixes=_normalize_suffixes(args.suffixes),
        chunk_size=int(args.chunk_size),
        chunk_overlap=int(args.chunk_overlap),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
