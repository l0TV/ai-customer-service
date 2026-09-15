"""政策文档加载与分块。

支持 Markdown / 纯文本 / PDF / Word(.docx)。

分块策略采用「先按标题切章节，再按长度递归切分」两段式：
政策文档天然以条款/章节组织，尊重标题边界能显著提升检索片段的语义完整性，
避免把一条政策从中间截断。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from langchain_core.documents import Document

from app.core.config import Settings, settings as default_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Markdown 标题：## 一、退货政策
_MD_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")

# 「一、xxx」这类中文编号：既可能是小节标题，也可能是正文列表项。
# 判据组合使用：中文数字开头（阿拉伯数字的「1. xxx」在中文政策文档里几乎
# 总是列表项而非章节）+ 长度短 + 不以列表标点结尾。
_CN_NUMBERED_HEADING = re.compile(r"^[一二三四五六七八九十]{1,3}\s*[、.]\s*\S{1,20}$")

# 明确以章节关键字开头的，视为标题（不受长度限制，但仍限 40 字内）
_CN_CHAPTER_HEADING = re.compile(
    r"^(?:"
    r"第\s*[0-9一二三四五六七八九十百]+\s*[条章节款项]\s*[、.．:：]?\s*\S{0,30}"
    r"|[（(]\s*[0-9一二三四五六七八九十]+\s*[）)]\s*\S{0,30}"
    r")$"
)

# 以这些标点结尾说明是正文列表项/句子，不可能是标题
_LIST_TAIL_CHARS = "；;，,。.：:、"


def is_chinese_heading(line: str) -> bool:
    """判断一行是否是中文小节标题。

    正文里的编号列表项（如「1. 商品包装完整，未拆封的密封商品保持密封状态。」）
    与小节标题（如「三、配送时效」）形态接近，用三个条件区分：
    编号形式（中文数字）、长度短、不以列表标点结尾。
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 40:
        return False

    if stripped.endswith(tuple(_LIST_TAIL_CHARS)):
        return False

    if _CN_CHAPTER_HEADING.match(stripped):
        return True

    if not _CN_NUMBERED_HEADING.match(stripped):
        return False

    return len(stripped) <= 22

# 中文友好的递归分隔符：段落 -> 换行 -> 中文句末 -> 分句 -> 空格 -> 字符
_CN_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    "、",
    ". ",
    " ",
    "",
]


@dataclass
class LoadedDocument:
    """一份加载完成的原始文档。"""

    source_path: Path
    text: str
    file_hash: str
    suffix: str
    sections: list[tuple[str, str]] = field(default_factory=list)


def file_fingerprint(path: Path) -> str:
    """计算文件内容 SHA-256，用于增量入库判定。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_fingerprint(text: str) -> str:
    """计算文本内容的 SHA-256。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# 各格式解析
# ----------------------------------------------------------------------
def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # 最后兜底：忽略非法字节
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    """解析 PDF。

    pypdf 对中文 PDF 的文本抽取质量一般，因此优先尝试 PyMuPDF
    （若已安装），失败时回退到 pypdf。
    """
    try:
        import fitz  # PyMuPDF

        pages: list[str] = []
        with fitz.open(path) as doc:
            for index, page in enumerate(doc):
                text = page.get_text("text")
                if text and text.strip():
                    pages.append(text.strip())
        if pages:
            return "\n\n".join(pages)
        logger.warning("PyMuPDF 未从 %s 抽取到文本，回退 pypdf", path.name)
    except ImportError:
        logger.debug("未安装 PyMuPDF，使用 pypdf 解析 %s", path.name)
    except Exception as exc:  # noqa: BLE001 - 解析失败需回退而非中断整批入库
        logger.warning("PyMuPDF 解析 %s 失败(%s)，回退 pypdf", path.name, exc)

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    """解析 Word 文档，按文档顺序串联段落与表格。"""
    import docx

    document = docx.Document(str(path))
    blocks: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower()
        if style.startswith("heading"):
            # 还原为 Markdown 标题，便于后续按标题切章节
            level = "".join(ch for ch in style if ch.isdigit()) or "2"
            blocks.append(f"{'#' * min(int(level), 6)} {text}")
        else:
            blocks.append(text)

    for table in document.tables:
        rows: list[str] = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                rows.append("| " + " | ".join(cells) + " |")
        if rows:
            # 表格首行后插入 Markdown 分隔行
            rows.insert(1, "| " + " | ".join(["---"] * len(table.columns)) + " |")
            blocks.append("\n".join(rows))

    return "\n\n".join(blocks)


_PARSERS = {
    ".md": _read_text_file,
    ".markdown": _read_text_file,
    ".txt": _read_text_file,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


def load_document(path: Path) -> LoadedDocument:
    """加载单个政策文档。

    Raises:
        ValueError: 后缀不支持。
        FileNotFoundError: 文件不存在。
    """
    if not path.exists():
        raise FileNotFoundError(f"文档不存在: {path}")

    suffix = path.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"不支持的文档格式: {suffix}（支持 {sorted(_PARSERS)}）")

    text = parser(path)
    text = _normalize_text(text)
    if not text:
        logger.warning("文档解析结果为空: %s", path.name)

    return LoadedDocument(
        source_path=path,
        text=text,
        file_hash=file_fingerprint(path),
        suffix=suffix,
    )


def _normalize_text(text: str) -> str:
    """清理解析噪声：全角空格、过多空行、孤立回车。"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    # 三个以上连续换行压缩为两个
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 行内多余空白压缩
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ----------------------------------------------------------------------
# 章节切分
# ----------------------------------------------------------------------
def split_into_sections(text: str) -> list[tuple[str, str]]:
    """把文档按标题切成 ``(章节标题, 章节正文)`` 列表。

    无任何标题时返回 ``[("", 全文)]``。
    """
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []

    current_title = ""
    buffer: list[str] = []
    heading_stack: list[tuple[int, str]] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body or current_title:
            sections.append((current_title, body))

    def push_heading(level: int, title: str) -> None:
        """压入标题并维护层级路径，使片段能定位到完整章节链。"""
        nonlocal current_title
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        current_title = " > ".join(t for _, t in heading_stack)

    for line in lines:
        stripped = line.strip()

        md_match = _MD_HEADING.match(stripped)
        # 中文标题判定：长度受限，避免把整段正文误判为标题
        is_cn_heading = is_chinese_heading(stripped)

        if md_match:
            flush()
            buffer = []
            push_heading(len(md_match.group("hashes")), md_match.group("title").strip())
            continue

        if is_cn_heading:
            if buffer or heading_stack:
                flush()
                buffer = []
            # 「第X章/条」视为一级，「一、」「（二）」「1.」视为二级
            level = 1 if stripped.startswith("第") else 2
            push_heading(level, stripped)
            continue

        buffer.append(line)

    flush()
    return [(title, body) for title, body in sections if body.strip()]


# ----------------------------------------------------------------------
# 分块
# ----------------------------------------------------------------------
def chunk_document(
    loaded: LoadedDocument,
    *,
    settings: Settings | None = None,
) -> list[Document]:
    """把 ``LoadedDocument`` 切分为带元数据的 ``Document`` 列表。"""
    cfg = settings or default_settings

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=_CN_SEPARATORS,
        length_function=len,
        keep_separator=True,
        is_separator_regex=False,
    )

    sections = split_into_sections(loaded.text)
    if not sections:
        sections = [("", loaded.text)]

    documents: list[Document] = []
    title = loaded.source_path.stem
    suffix = loaded.suffix.lstrip(".")
    chunk_index = 0

    for section_title, section_body in sections:
        for piece in splitter.split_text(section_body):
            piece = piece.strip()
            if not piece:
                continue
            documents.append(
                Document(
                    page_content=piece,
                    metadata={
                        # chunk_id 全局唯一：文件哈希 + 序号，供 RRF 去重与引用定位
                        "chunk_id": f"{loaded.file_hash[:16]}-{chunk_index:04d}",
                        "source": loaded.source_path.name,
                        "source_path": str(loaded.source_path),
                        "file_hash": loaded.file_hash,
                        "doc_type": suffix,
                        "title": title,
                        "section": section_title or title,
                        "chunk_index": chunk_index,
                    },
                )
            )
            chunk_index += 1

    logger.info(
        "已分块 %s -> %d 个片段（%d 个章节）",
        loaded.source_path.name,
        len(documents),
        len(sections),
    )
    return documents


def iter_policy_files(
    directory: Path,
    suffixes: Sequence[str] | None = None,
) -> Iterator[Path]:
    """递归遍历政策文档目录。"""
    allowed = {s.lower() for s in (suffixes or default_settings.supported_suffixes)}
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed and not path.name.startswith("~$"):
            yield path


def load_and_chunk_all(
    directory: Path | None = None,
    *,
    settings: Settings | None = None,
) -> tuple[list[Document], dict[str, str]]:
    """加载目录下全部政策文档并分块。

    Returns:
        ``(全部片段, {文件名: 文件指纹})``。指纹用于增量判定。
    """
    cfg = settings or default_settings
    target = directory or cfg.policy_dir

    all_chunks: list[Document] = []
    manifest: dict[str, str] = {}

    for path in iter_policy_files(target, cfg.supported_suffixes):
        try:
            loaded = load_document(path)
        except Exception as exc:  # noqa: BLE001 - 单文件失败不应中断整批
            logger.error("加载文档失败 %s: %s", path, exc)
            continue

        manifest[str(path.relative_to(target))] = loaded.file_hash
        all_chunks.extend(chunk_document(loaded, settings=cfg))

    logger.info("共加载 %d 个文件，生成 %d 个片段", len(manifest), len(all_chunks))
    return all_chunks, manifest


def deduplicate(documents: Iterable[Document]) -> list[Document]:
    """按 ``chunk_id`` 去重，保持首次出现顺序。"""
    seen: set[str] = set()
    result: list[Document] = []
    for doc in documents:
        key = doc.metadata.get("chunk_id") or content_fingerprint(doc.page_content)
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result
