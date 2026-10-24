"""Structure-aware chunker.

Splits per-page normalized text into chunks of ~300-500 tokens with ~50 token
overlap, preferring paragraph and heading boundaries. Each chunk carries
page_start / page_end / section_title so retrieval can cite sources.

Token count is approximated as whitespace-split word count. This is a safe
underestimate for English text; we will swap in a tokenizer-aligned count once
the embedding model's tokenizer is pinned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TARGET_MIN_TOKENS = 300
TARGET_MAX_TOKENS = 500
OVERLAP_TOKENS = 50

# A heading line: short, mostly capitalized or numbered. Deliberately
# conservative — false positives hurt chunk coherence.
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,:'\-]{0,80}"
    r"|CHAPTER\s+\d+.*"
    r"|SECTION\s+\d+.*"
    r")$"
)


@dataclass
class PageText:
    """Normalized text for a single PDF page."""
    page_number: int
    text: str
    section_title: str | None = None  # most-recent heading on or before this page


@dataclass
class Chunk:
    chunk_index: int
    page_start: int
    page_end: int
    section_title: str | None
    text: str
    token_count: int
    source_page_numbers: list[int] = field(default_factory=list)


def _tokenize(text: str) -> list[str]:
    return text.split()


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if stripped.endswith((".", "?", "!", ",", ";", ":")):
        return False
    return bool(_HEADING_RE.match(stripped))


def _split_paragraphs(text: str) -> list[str]:
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def _tagged_paragraphs(pages: list[PageText]) -> list[tuple[int, str | None, str]]:
    """Return list of (page_number, section_title, paragraph_text).

    Also detects headings inline: a heading line becomes the running
    section_title for subsequent paragraphs on the same and later pages.
    """
    current_section: str | None = None
    tagged: list[tuple[int, str | None, str]] = []
    for page in pages:
        if page.section_title:
            current_section = page.section_title
        for para in _split_paragraphs(page.text):
            lines = para.splitlines()
            if len(lines) == 1 and _looks_like_heading(lines[0]):
                current_section = lines[0].strip()
                continue
            # Heading as first line of a multi-line paragraph: lift it out.
            if lines and _looks_like_heading(lines[0]):
                current_section = lines[0].strip()
                rest = "\n".join(lines[1:]).strip()
                if rest:
                    tagged.append((page.page_number, current_section, rest))
                continue
            tagged.append((page.page_number, current_section, para))
    return tagged


def chunk_pages(
    pages: list[PageText],
    target_min_tokens: int = TARGET_MIN_TOKENS,
    target_max_tokens: int = TARGET_MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk a chapter's pages into retrieval-ready passages."""
    if target_min_tokens <= 0 or target_max_tokens < target_min_tokens:
        raise ValueError("invalid chunk size bounds")

    paragraphs = _tagged_paragraphs(pages)
    chunks: list[Chunk] = []

    buf_paras: list[tuple[int, str | None, str]] = []
    buf_tokens = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal chunk_index, buf_paras, buf_tokens
        if not buf_paras:
            return
        text = "\n\n".join(p for _, _, p in buf_paras).strip()
        if not text:
            buf_paras = []
            buf_tokens = 0
            return
        page_nums = sorted({p for p, _, _ in buf_paras})
        section = next((s for _, s, _ in buf_paras if s), None)
        chunks.append(
            Chunk(
                chunk_index=chunk_index,
                page_start=page_nums[0],
                page_end=page_nums[-1],
                section_title=section,
                text=text,
                token_count=len(_tokenize(text)),
                source_page_numbers=page_nums,
            )
        )
        chunk_index += 1
        # Seed next buffer with overlap: tail tokens from the last paragraph.
        if overlap_tokens > 0 and buf_paras:
            tail_page, tail_section, tail_para = buf_paras[-1]
            tail_tokens = _tokenize(tail_para)
            if len(tail_tokens) > overlap_tokens:
                overlap_text = " ".join(tail_tokens[-overlap_tokens:])
                buf_paras = [(tail_page, tail_section, overlap_text)]
                buf_tokens = len(tail_tokens[-overlap_tokens:])
                return
        buf_paras = []
        buf_tokens = 0

    for page_num, section, para in paragraphs:
        para_tokens = _tokenize(para)
        n = len(para_tokens)

        # A paragraph larger than target_max must be split by sentences.
        if n > target_max_tokens:
            if buf_tokens >= target_min_tokens:
                flush()
            for piece in _split_long_paragraph(para_tokens, target_max_tokens):
                piece_text = " ".join(piece)
                buf_paras.append((page_num, section, piece_text))
                buf_tokens += len(piece)
                if buf_tokens >= target_min_tokens:
                    flush()
            continue

        if buf_tokens + n > target_max_tokens and buf_tokens >= target_min_tokens:
            flush()
        buf_paras.append((page_num, section, para))
        buf_tokens += n
        if buf_tokens >= target_max_tokens:
            flush()

    # Final partial chunk (even if below min — don't drop content).
    if buf_paras:
        text = "\n\n".join(p for _, _, p in buf_paras).strip()
        if text:
            page_nums = sorted({p for p, _, _ in buf_paras})
            section = next((s for _, s, _ in buf_paras if s), None)
            chunks.append(
                Chunk(
                    chunk_index=chunk_index,
                    page_start=page_nums[0],
                    page_end=page_nums[-1],
                    section_title=section,
                    text=text,
                    token_count=len(_tokenize(text)),
                    source_page_numbers=page_nums,
                )
            )

    return chunks


def _split_long_paragraph(tokens: list[str], max_tokens: int) -> list[list[str]]:
    """Split a long paragraph into max_tokens-sized windows at sentence-ish boundaries."""
    text = " ".join(tokens)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[list[str]] = []
    current: list[str] = []
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        if len(current) + len(words) > max_tokens and current:
            pieces.append(current)
            current = []
        if len(words) > max_tokens:
            # Single huge sentence — hard-split.
            for i in range(0, len(words), max_tokens):
                pieces.append(words[i : i + max_tokens])
        else:
            current.extend(words)
    if current:
        pieces.append(current)
    return pieces
