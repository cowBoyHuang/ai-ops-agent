"""导入本地排障文档到 Qdrant。"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5
import xml.etree.ElementTree as ET

# 兼容直接从仓库根目录执行：python rag_ingest/import_local_docs.py
_ROOT_DIR = Path(__file__).resolve().parents[1]
_SRC_DIR = _ROOT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pydantic import SecretStr

from qdrant import QdrantStore

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - 可选运行时依赖
    ChatOpenAI = None  # type: ignore[assignment]

_DEFAULT_SUFFIXES = [".md", ".txt", ".text", ".doc", ".docx", ".log", ".rst"]
_DEFAULT_CHUNK_SIZE = 900
_DEFAULT_CHUNK_OVERLAP = 120
_DEFAULT_MULTIMODAL_MODEL = "azure/gpt-4.1"
_DEFAULT_LLM_BASE_URL = "http://llm.api.corp.qunar.com/v1"
_DOCX_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}
_DOCX_REL_ATTR_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_MULTIMODAL_CLIENT: Any | None = None
_MULTIMODAL_INIT_DONE = False
_TITLE_PATTERNS = (
    re.compile(r"^\s{0,3}#{1,6}\s+\S+"),  # markdown headings: #/##/...
    re.compile(r"^[一二三四五六七八九十百]+[、.．]\s*\S+"),  # Chinese ordered title
    re.compile(r"^\d+([.)）]|[.．、])\s*\S+"),  # numeric ordered title
    re.compile(r"^[（(]?[一二三四五六七八九十百\d]+[)）]\s*\S+"),  # (1)/(一)
    re.compile(r"^[A-Za-z][.)]\s+\S+"),  # A. / a)
)


@dataclass
class ImageAsset:
    name: str
    mime_type: str
    data: bytes


# 方法注释（业务）:
# - 入参：`raw`(str)=命令行后缀配置，逗号分隔。
# - 出参：`list[str]`=标准化后缀列表（统一小写且带点前缀）。
# - 方法逻辑：按逗号切分、去空白并补齐前导点，最终得到可用于匹配的后缀集合。
def _normalize_suffixes(raw: str) -> list[str]:
    values = [item.strip().lower() for item in str(raw or "").split(",") if item.strip()]
    return [item if item.startswith(".") else f".{item}" for item in values]


# 方法注释（业务）:
# - 入参：`raw`(str)=命令行绝对路径配置，逗号分隔。
# - 出参：`list[str]`=标准化后的路径列表。
# - 方法逻辑：按逗号切分并去除空白，保持输入顺序。
def _normalize_source_paths(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


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
# - 入参：`source_paths`(list[str]|None)=显式输入路径（文件或目录）；`source_dir`(str|Path|None)=目录入口；`suffixes`=允许后缀。
# - 出参：`tuple[list[Path], str]`=(解析后的文件列表, 来源模式)。
# - 方法逻辑：优先使用 `source_paths`，支持绝对路径文件/目录；否则回退目录扫描。
def _resolve_source_files(
    *,
    source_paths: list[str] | None,
    source_dir: str | Path | None,
    suffixes: list[str],
) -> tuple[list[Path], str]:
    allowed = {item.lower() for item in suffixes}
    if source_paths:
        resolved: list[Path] = []
        for raw in source_paths:
            candidate = Path(str(raw)).expanduser()
            if not candidate.is_absolute():
                candidate = candidate.resolve()
            if candidate.is_dir():
                resolved.extend(_collect_source_files(candidate, suffixes))
                continue
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in allowed:
                continue
            resolved.append(candidate)

        deduped = sorted({path.resolve() for path in resolved}, key=lambda item: str(item))
        return deduped, "paths"

    target_dir = Path(source_dir) if source_dir else (_ROOT_DIR / "rag_ingest" / "source_docs")
    return _collect_source_files(target_dir, suffixes), "dir"


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
# - 入参：`text`(str)=全文文本。
# - 出参：`list[str]`=段落列表。
# - 方法逻辑：按空行切分段落并压缩段内空白。
def _split_paragraphs(text: str) -> list[str]:
    content = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not content:
        return []
    rows = re.split(r"\n\s*\n+", content)
    return [" ".join(row.split()) for row in rows if row.strip()]


# 方法注释（业务）:
# - 入参：`text`(str)=超长段落文本；`chunk_size`/`chunk_overlap`=切分参数。
# - 出参：`list[str]`=按长度兜底切分后的片段。
# - 方法逻辑：对单段过长内容按字符窗口切分，避免超大 chunk。
def _split_long_paragraph(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    content = str(text or "").strip()
    if not content:
        return []
    size = max(1, int(chunk_size))
    overlap = max(0, min(int(chunk_overlap), size - 1))
    step = max(1, size - overlap)
    chunks: list[str] = []
    start = 0
    length = len(content)
    while start < length:
        end = min(length, start + size)
        part = content[start:end].strip()
        if part:
            chunks.append(part)
        if end >= length:
            break
        start += step
    return chunks


def _split_sentences(text: str) -> list[str]:
    content = str(text or "").strip()
    if not content:
        return []
    parts = re.split(r"(?<=[。！？!?；;])\s*", content)
    rows = [str(item or "").strip() for item in parts if str(item or "").strip()]
    if rows:
        return rows
    return [content]


def _is_title_like_paragraph(text: str, next_text: str = "") -> bool:
    current = str(text or "").strip()
    if not current:
        return False
    # Heuristic: short standalone line is likely a title.
    if any(pattern.match(current) for pattern in _TITLE_PATTERNS):
        return True
    max_title_len = 24
    if len(current) <= max_title_len and not re.search(r"[。！？!?；;，,：:]", current):
        nxt = str(next_text or "").strip()
        if len(nxt) >= 16:
            return True
    return False


def _merge_title_with_body_paragraphs(paragraphs: list[str]) -> list[str]:
    rows = [str(item or "").strip() for item in list(paragraphs or []) if str(item or "").strip()]
    merged: list[str] = []
    i = 0
    while i < len(rows):
        current = rows[i]
        next_text = rows[i + 1] if i + 1 < len(rows) else ""
        if _is_title_like_paragraph(current, next_text) and next_text:
            body = next_text
            merged.append(f"{current}\n{body}".strip())
            i += 2
            continue
        merged.append(current)
        i += 1
    return merged


def _pick_first_sentence(text: str) -> str:
    rows = _split_sentences(text)
    return rows[0] if rows else ""


def _pick_last_sentence(text: str) -> str:
    rows = _split_sentences(text)
    return rows[-1] if rows else ""


# 方法注释（业务）:
# - 入参：`paragraphs`(list[str])=段落文本列表；`chunk_size`/`chunk_overlap`=切分参数。
# - 出参：`list[str]`=chunk 列表。
# - 方法逻辑：严格一段一块；每块拼接上一段最后一句与下一段第一句作为 overlap 语义增强。
def _split_chunks_by_paragraphs(
    paragraphs: list[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    _ = chunk_size, chunk_overlap  # 保留参数兼容性；严格段落切分时不使用长度窗口。
    cleaned = _merge_title_with_body_paragraphs(paragraphs)
    chunks: list[str] = []
    for idx, current in enumerate(cleaned):
        prev_tail = _pick_last_sentence(cleaned[idx - 1]) if idx > 0 else ""
        next_head = _pick_first_sentence(cleaned[idx + 1]) if idx + 1 < len(cleaned) else ""
        chunk_parts = [part for part in (prev_tail, current, next_head) if part]
        chunk_text = "\n".join(chunk_parts).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def _mime_type_from_name(name: str) -> str:
    lower = str(name or "").lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".bmp"):
        return "image/bmp"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".tiff") or lower.endswith(".tif"):
        return "image/tiff"
    return "application/octet-stream"


def _extract_markdown_images(text: str, base_path: Path) -> list[ImageAsset]:
    rows: list[ImageAsset] = []
    seen: set[str] = set()
    for raw_ref in re.findall(r"!\[[^\]]*]\(([^)]+)\)", str(text or "")):
        ref = str(raw_ref or "").strip().strip("\"'")
        if not ref or ref.startswith(("http://", "https://", "data:")):
            continue
        local_path = (base_path.parent / ref).resolve() if not Path(ref).is_absolute() else Path(ref)
        key = str(local_path)
        if key in seen or not local_path.is_file():
            continue
        seen.add(key)
        try:
            data = local_path.read_bytes()
        except Exception:
            continue
        rows.append(
            ImageAsset(
                name=local_path.name,
                mime_type=_mime_type_from_name(local_path.name),
                data=data,
            )
        )
    return rows


def _coerce_llm_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rows: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    rows.append(str(text))
            else:
                rows.append(str(item))
        return "\n".join(rows).strip()
    return str(content)


def _build_multimodal_llm_client() -> Any | None:
    global _MULTIMODAL_CLIENT, _MULTIMODAL_INIT_DONE
    if _MULTIMODAL_INIT_DONE:
        return _MULTIMODAL_CLIENT
    _MULTIMODAL_INIT_DONE = True
    if ChatOpenAI is None:
        _MULTIMODAL_CLIENT = None
        return None

    api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        _MULTIMODAL_CLIENT = None
        return None

    model = (
        str(os.getenv("AIOPS_MULTIMODAL_MODEL", "")).strip()
        or str(os.getenv("AIOPS_LLM_MODEL", "")).strip()
        or _DEFAULT_MULTIMODAL_MODEL
    )
    base_url = str(os.getenv("AIOPS_LLM_BASE_URL", _DEFAULT_LLM_BASE_URL)).strip() or _DEFAULT_LLM_BASE_URL
    try:
        _MULTIMODAL_CLIENT = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
            temperature=0,
        )
    except Exception:  # pragma: no cover - 运行时依赖异常
        _MULTIMODAL_CLIENT = None
    return _MULTIMODAL_CLIENT


def _build_image_prompt_by_context(context: str) -> str:
    lower = str(context or "").lower()
    if any(key in lower for key in ("error", "exception", "报错", "异常")):
        return "提取错误代码、异常类型、错误消息。"
    if any(key in lower for key in ("chart", "graph", "图表", "趋势")):
        return "描述图表类型、关键数值、趋势、异常点。"
    if any(key in lower for key in ("config", "setting", "配置", "参数")):
        return "提取配置项名称、当前值、默认值、单位。"
    return "提取所有可读的技术信息。"


def _describe_image_with_multimodal_llm(*, image: ImageAsset, source_path: Path, image_index: int) -> str:
    llm = _build_multimodal_llm_client()
    if llm is None or not image.data:
        return ""
    if not str(image.mime_type).lower().startswith("image/"):
        return ""

    context = f"{source_path.name} {image.name}".strip()
    image_prompt = _build_image_prompt_by_context(context)
    image_b64 = base64.b64encode(image.data).decode("ascii")
    try:
        result = llm.invoke(
            [
                (
                    "system",
                    "你是文档图片解析助手。根据任务指令提取图片里的技术信息，输出简洁中文描述（不超过120字）。",
                ),
                (
                    "human",
                    [
                        {
                            "type": "text",
                            "text": (
                                f"任务指令：{image_prompt}\n"
                                f"请描述该图片内容。文件：{source_path.name}，图片序号：{image_index}，"
                                f"上下文：{context}。优先提取可检索信息。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image.mime_type};base64,{image_b64}"},
                        },
                    ],
                ),
            ]
        )
    except Exception:  # pragma: no cover - 外部模型异常统一降级
        return ""

    text = _coerce_llm_text(getattr(result, "content", result)).strip()
    if not text:
        return ""
    return f"图片描述({image.name}): {text}"


def _extract_docx_relationships(zf: zipfile.ZipFile) -> dict[str, str]:
    rels_path = "word/_rels/document.xml.rels"
    if rels_path not in zf.namelist():
        return {}
    try:
        rel_root = ET.fromstring(zf.read(rels_path))
    except Exception:
        return {}

    mapping: dict[str, str] = {}
    id_attr = "Id"
    target_attr = "Target"
    for row in rel_root.findall("pr:Relationship", _DOCX_NS):
        rel_id = str(row.attrib.get(id_attr) or "").strip()
        target = str(row.attrib.get(target_attr) or "").strip()
        if rel_id and target:
            mapping[rel_id] = target
    return mapping


def _extract_docx_content(path: Path) -> tuple[list[str], list[ImageAsset]]:
    paragraphs: list[str] = []
    images: list[ImageAsset] = []

    try:
        with zipfile.ZipFile(path, "r") as zf:
            doc_path = "word/document.xml"
            if doc_path not in zf.namelist():
                return [], []
            root = ET.fromstring(zf.read(doc_path))
            rel_mapping = _extract_docx_relationships(zf)
            added_media_names: set[str] = set()

            for para in root.findall(".//w:p", _DOCX_NS):
                text = "".join(node.text or "" for node in para.findall(".//w:t", _DOCX_NS)).strip()
                if text:
                    paragraphs.append(" ".join(text.split()))

                for blip in para.findall(".//a:blip", _DOCX_NS):
                    rel_id = str(blip.attrib.get(_DOCX_REL_ATTR_ID) or "").strip()
                    target = str(rel_mapping.get(rel_id) or "").strip()
                    if not target:
                        continue
                    target_name = target[3:] if target.startswith("../") else target
                    zip_member = f"word/{target_name}".replace("\\", "/")
                    if zip_member not in zf.namelist():
                        continue
                    if zip_member in added_media_names:
                        continue
                    added_media_names.add(zip_member)
                    file_name = Path(target_name).name
                    images.append(
                        ImageAsset(
                            name=file_name,
                            mime_type=_mime_type_from_name(file_name),
                            data=zf.read(zip_member),
                        )
                    )
    except Exception:
        return [], []

    return paragraphs, images


def _extract_doc_content(path: Path) -> tuple[list[str], list[ImageAsset]]:
    # legacy .doc 优先尝试 antiword 提取文本；图片提取在无 Office 解析库时不稳定，暂不处理。
    try:
        result = subprocess.run(
            ["antiword", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        if result.returncode == 0:
            return _split_paragraphs(result.stdout), []
    except Exception:
        pass
    return [], []


def _extract_document(path: Path) -> tuple[list[str], list[ImageAsset]]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _extract_docx_content(path)
    if suffix == ".doc":
        return _extract_doc_content(path)
    if suffix == ".md":
        text = _read_text(path)
        return _split_paragraphs(text), _extract_markdown_images(text, path)
    if suffix in {".txt", ".text", ".log", ".rst"}:
        return _split_paragraphs(_read_text(path)), []
    return [], []


# 方法注释（业务）:
# - 入参：`doc_path`(Path)=原始文档路径；`paragraphs`(list[str])=文档段落列表；`chunk_size`/`chunk_overlap`=切分参数。
# - 出参：`tuple[str, list[dict[str, Any]]]`=(parent_id, qdrant_upsert_items)。
# - 方法逻辑：每篇原始文档生成唯一 parent_id，按段落切分后把 parent_id/path 写入每个 chunk 的 payload。
def _build_upsert_items(
    *,
    doc_path: Path,
    paragraphs: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[str, list[dict[str, Any]]]:
    # Use stable parent_id per absolute file path so one article keeps one parent_id
    # across repeated imports.
    abs_path = str(doc_path.resolve())
    parent_id = uuid5(NAMESPACE_URL, abs_path).hex
    chunks = _split_chunks_by_paragraphs(
        paragraphs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    items: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        items.append(
            {
                "id": uuid4().hex,
                "text": chunk,
                "payload": {
                    "parent_id": parent_id,
                    "path": abs_path,
                    "source_file_abs_path": abs_path,
                    "chunk_index": idx,
                    "file_name": doc_path.name,
                },
            }
        )
    return parent_id, items


# 方法注释（业务）:
# - 入参：`source_paths`(list[str]|None)=绝对路径文件/目录列表；`source_dir`(str|Path|None)=目录入口；其余为切分参数。
# - 出参：`dict[str, Any]`=导入统计结果。
# - 方法逻辑：先解析段落与图片，使用多模态模型生成图片描述，再统一切分、向量化并写入 DB/Qdrant。
def run_import_entry(
    *,
    source_paths: list[str] | None = None,
    source_dir: str | Path | None = None,
    suffixes: list[str] | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    resolved_source_dir = (Path(source_dir) if source_dir else (_ROOT_DIR / "rag_ingest" / "source_docs")).resolve()
    normalized_suffixes = list(suffixes or _DEFAULT_SUFFIXES)
    files, source_mode = _resolve_source_files(
        source_paths=source_paths,
        source_dir=source_dir,
        suffixes=normalized_suffixes,
    )

    qdrant_store = QdrantStore()

    file_count = 0
    chunk_count = 0
    image_desc_count = 0
    skipped_files: list[str] = []

    for doc_path in files:
        paragraphs, images = _extract_document(doc_path)
        image_paragraphs: list[str] = []
        for idx, image in enumerate(images, start=1):
            description = _describe_image_with_multimodal_llm(
                image=image,
                source_path=doc_path,
                image_index=idx,
            )
            if not description:
                continue
            image_paragraphs.append(description)
        image_desc_count += len(image_paragraphs)

        merged_paragraphs = [*paragraphs, *image_paragraphs]
        if not merged_paragraphs:
            skipped_files.append(str(doc_path))
            continue

        _parent_id, items = _build_upsert_items(
            doc_path=doc_path,
            paragraphs=merged_paragraphs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not items:
            skipped_files.append(str(doc_path))
            continue

        qdrant_store.upsert_texts(items)

        file_count += 1
        chunk_count += len(items)

    return {
        "source_mode": source_mode,
        "source_dir": str(resolved_source_dir) if source_mode == "dir" else "",
        "source_paths": [str(path) for path in files] if source_mode == "paths" else [],
        "suffixes": normalized_suffixes,
        "files_found": len(files),
        "files_ingested": file_count,
        "chunks_ingested": chunk_count,
        "image_descriptions_generated": image_desc_count,
        "qdrant_collection": qdrant_store.config.collection_name,
        "skipped_files": skipped_files,
    }


# 方法注释（业务）:
# - 入参：`source_dir`(str|Path|None)=文档目录；`suffixes`(list[str]|None)=后缀白名单；`chunk_size`/`chunk_overlap`=切分参数。
# - 出参：`dict[str, Any]`=导入统计结果。
# - 方法逻辑：兼容旧入口，内部复用 `run_import_entry`。
def run_import(
    *,
    source_dir: str | Path | None = None,
    suffixes: list[str] | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    return run_import_entry(
        source_dir=source_dir,
        suffixes=suffixes,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


# 方法注释（业务）:
# - 入参：无（从命令行读取）。
# - 出参：`None`。
# - 方法逻辑：解析参数后执行导入，并将结构化结果打印为 JSON。
def main() -> None:
    parser = argparse.ArgumentParser(description="导入本地文档到 Qdrant")
    parser.add_argument(
        "--source-dir",
        default="",
        help="文档目录，默认使用 rag_ingest/source_docs",
    )
    parser.add_argument(
        "--source-paths",
        default="",
        help="绝对路径列表（文件或目录），逗号分隔；优先级高于 --source-dir",
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

    result = run_import_entry(
        source_paths=_normalize_source_paths(args.source_paths),
        source_dir=args.source_dir or None,
        suffixes=_normalize_suffixes(args.suffixes),
        chunk_size=int(args.chunk_size),
        chunk_overlap=int(args.chunk_overlap),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
