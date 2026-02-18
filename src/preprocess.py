import os
from pathlib import Path
from typing import Iterable, List, Tuple, Dict
import re
from collections import defaultdict
from pypdf import PdfReader

from src.models import Chunk


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"


def infer_topic_and_title(pdf_path: Path) -> Tuple[str, str]:
    """
    Infer a stable topic key + a human title from the filename.
    Examples:
      "Eating disorders.pdf" -> ("eating_disorders", "Eating disorders")
      "Depression.pdf" -> ("depression", "Depression")
    """
    title = pdf_path.stem.strip()
    topic = title.lower()
    topic = re.sub(r"[\s\-]+", "_", topic)
    topic = re.sub(r"[^a-z0-9_]+", "", topic)
    topic = re.sub(r"_+", "_", topic).strip("_")
    if not topic:
        topic = "unknown"
    return topic, title


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    texts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        texts.append(text)
    # Separate pages by a blank line to create natural paragraph boundaries.
    return "\n\n".join(texts)


def clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph boundaries.
    We keep double newlines (\n\n) between paragraphs so chunking
    can prioritize splitting at those points.
    """
    if not text:
        return ""

    # Normalize Windows newlines and carriage returns.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on two or more newlines to detect paragraphs.
    raw_paragraphs = re.split(r"\n{2,}", text)
    paragraphs: List[str] = []
    for para in raw_paragraphs:
        # Within a paragraph, collapse inner newlines and spaces.
        lines = [ln.strip() for ln in para.split("\n")]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue
        paragraph_text = " ".join(lines)
        # Collapse multiple spaces.
        paragraph_text = re.sub(r"\s+", " ", paragraph_text).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    # Join paragraphs with exactly two newlines.
    return "\n\n".join(paragraphs)


def _split_to_sentences(text: str) -> List[str]:
    """
    Heuristic sentence splitter that works reasonably for English and Hebrew.
    We split on punctuation marks followed by whitespace/newline.
    """
    # Normalize whitespace to make regex more reliable.
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    # Split on sentence-ending punctuation: . ? ! : and quotes; keep delimiters.
    parts = re.split(r"([\.!?:״”\"׳']\s+)", normalized)
    sentences: List[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        current += part
        if re.search(r"[\.!?]$", part.strip()):
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def _split_long_text(text: str, max_chars: int) -> List[str]:
    """
    Fallback splitter for very long text when sentence/paragraph grouping
    is still above max_chars. Prefer splitting on spaces so we don't cut words.
    """
    chunks: List[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        # Find last space within the window.
        split_pos = window.rfind(" ")
        if split_pos == -1:
            # No space found; hard cut.
            split_pos = max_chars
        chunk = remaining[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def count_tokens(text: str) -> int:
    """
    Approximate token count using word count.
    This is a rough proxy (1 word ~= 1 token) sufficient for sizing.
    """
    if not text:
        return 0
    return len(text.split())


def chunk_text(
    text: str,
    min_tokens: int = 150,
    target_tokens: int = 350,
    max_tokens: int = 600,
    overlap_sentences: int = 2,
) -> List[str]:
    """
    Heuristic semantic-ish chunking driven by structure and token targets:
    - Priority: paragraphs -> sentences -> spaces (fallback only).
    - Chunks are built from whole sentences.
    - Targets: ~250–450 tokens, bounded by [min_tokens, max_tokens].
    - Overlap: last 1–2 FULL sentences from previous chunk are repeated
      at the beginning of the next chunk when starting a new region.
    """
    if not text:
        return []

    # 1) Split into paragraphs.
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    all_sentences: List[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        sentences = _split_to_sentences(para)
        if not sentences:
            # Fallback: split long text on spaces to avoid overlong segments.
            all_sentences.extend(_split_long_text(para, max_tokens * 4))
        else:
            all_sentences.extend(sentences)

    if not all_sentences:
        return []

    chunks: List[str] = []
    current_sents: List[str] = []
    current_tokens = 0

    def flush_current():
        nonlocal current_sents, current_tokens
        if not current_sents:
            return
        chunk_text_val = " ".join(current_sents).strip()
        if not chunk_text_val:
            return
        chunks.append(chunk_text_val)
        # Prepare overlap sentences for the next chunk (tail).
        tail = current_sents[-overlap_sentences:] if overlap_sentences > 0 else []
        current_sents = tail.copy()
        current_tokens = count_tokens(" ".join(current_sents))

    for sent in all_sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_tokens = count_tokens(sent)

        # If adding this sentence would exceed max and we already have
        # at least min_tokens, flush and start a new chunk (with overlap).
        if current_sents and current_tokens + sent_tokens > max_tokens and current_tokens >= min_tokens:
            flush_current()

        # If the sentence itself is extremely long (> max_tokens),
        # force it into its own chunk.
        if sent_tokens > max_tokens:
            if current_sents:
                flush_current()
            chunks.append(sent)
            current_sents = [sent] if overlap_sentences > 0 else []
            current_tokens = count_tokens(" ".join(current_sents))
            continue

        current_sents.append(sent)
        current_tokens += sent_tokens

        # If we've reached the target range, consider flushing.
        if current_tokens >= target_tokens:
            flush_current()

    # Flush any remaining sentences as a final chunk.
    if current_sents:
        chunk_text_val = " ".join(current_sents).strip()
        if chunk_text_val:
            chunks.append(chunk_text_val)

    # Remove any empty/whitespace-only chunks.
    chunks = [c for c in chunks if c.strip()]
    return chunks


def generate_chunks() -> Iterable[Chunk]:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {RAW_DATA_DIR}. "
            f"Please put your psychology PDFs there (Trauma, Anxiety, Depression, Eating disorders, etc.)."
        )

    for pdf_path in pdf_files:
        topic, doc_title = infer_topic_and_title(pdf_path)
        raw_text = read_pdf(pdf_path)
        cleaned = clean_text(raw_text)
        text_chunks = chunk_text(cleaned)

        for idx, chunk_text_value in enumerate(text_chunks):
            chunk = Chunk(
                id=f"{topic}_{idx:04d}",
                text=chunk_text_value,
                topic=topic,
                doc_title=doc_title,
                source=pdf_path.name,
                page=-1,  # page-level tracking omitted for now
                chunk_index=idx,
                section_path=f"{topic}/{doc_title}",
            )
            yield chunk


def main() -> None:
    chunks = [ch for ch in generate_chunks() if ch.text.strip()]

    # Strict JSONL: one JSON object per line, no blank lines.
    with CHUNKS_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for ch in chunks:
            line = ch.model_dump_json(ensure_ascii=False)
            f.write(line + "\n")

    # Reporting: number of chunks per topic, avg token length, % with section_path.
    per_topic: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "tokens": 0.0})
    total_with_section_path = 0

    for ch in chunks:
        tokens = count_tokens(ch.text)
        stats = per_topic[ch.topic]
        stats["count"] += 1
        stats["tokens"] += tokens
        if ch.section_path:
            total_with_section_path += 1

    total_chunks = len(chunks)
    print(f"Wrote {total_chunks} chunks to {CHUNKS_PATH}")
    print("Chunk statistics by topic:")
    for topic, stats in per_topic.items():
        count = int(stats["count"])
        avg_tokens = stats["tokens"] / count if count else 0.0
        print(f"  - {topic}: {count} chunks, avg tokens ≈ {avg_tokens:.1f}")

    pct_section = (100.0 * total_with_section_path / total_chunks) if total_chunks else 0.0
    print(f"Chunks with section_path: {total_with_section_path}/{total_chunks} ({pct_section:.1f}%)")


if __name__ == "__main__":
    main()

