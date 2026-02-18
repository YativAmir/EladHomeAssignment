import os
from pathlib import Path
from typing import Iterable, List
import re
from pypdf import PdfReader

from models import Chunk


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

def infer_topic_and_title(pdf_path: Path) -> tuple[str, str]:
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
    return "\n".join(texts)


def clean_text(text: str) -> str:
    # Very simple cleaning for now; we can refine later.
    text = text.replace("\r", " ")
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]  # drop empty lines
    return "\n".join(lines)


def _split_to_sentences(text: str) -> List[str]:
    """
    Heuristic sentence splitter that works reasonably for English and Hebrew.
    We split on punctuation marks followed by whitespace/newline.
    """
    # Normalize whitespace to make regex more reliable
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    # Split on ., !, ? and the Hebrew question mark ן (just in case texts include it)
    # We keep the delimiter attached to the sentence.
    parts = re.split(r"([\.!?״”\"׳']\s+)", normalized)
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


def chunk_text(
    text: str,
    max_chars: int = 1800,
    overlap_chars: int = 450,
) -> List[str]:
    """
    Heuristic semantic-ish chunking:
    1) Split into sentences (paragraph/sentence boundaries).
    2) Group consecutive sentences into chunks up to `max_chars`.
    3) Add overlap between chunks based on characters.
    """
    if not text:
        return []

    sentences = _split_to_sentences(text)
    if not sentences:
        return []

    # First, build non-overlapping sentence groups up to max_chars.
    base_chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent) + 1  # +1 for space/newline
        if current and current_len + sent_len > max_chars:
            base_chunks.append(" ".join(current).strip())
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += sent_len

    if current:
        base_chunks.append(" ".join(current).strip())

    # Now add character-based overlap between consecutive chunks.
    if len(base_chunks) <= 1:
        return base_chunks

    overlapped_chunks: List[str] = []
    for i, chunk in enumerate(base_chunks):
        if i == 0:
            overlapped_chunks.append(chunk)
            continue

        prev = base_chunks[i - 1]
        # Take the last `overlap_chars` characters from previous chunk
        overlap_tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
        combined = (overlap_tail + " " + chunk).strip()
        overlapped_chunks.append(combined)

    return overlapped_chunks


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
                id=f"{topic}-{idx}",
                text=chunk_text_value,
                topic=topic,
                doc_title=doc_title,
                source=pdf_path.name,
                page=-1,  # page-level tracking omitted for now
                chunk_index=idx,
            )
            yield chunk


def main() -> None:
    chunks = list(generate_chunks())
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(ch.model_dump_json(ensure_ascii=False) + os.linesep)
    print(f"Wrote {len(chunks)} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()

