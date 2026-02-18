import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from src.config import get_settings

load_dotenv()
settings = get_settings()

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"

BATCH_SIZE = 16


def load_chunks(path: Path) -> List[Dict]:
    """Load preprocessed chunks from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found at {path}")

    chunks: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(obj)
    return chunks


def chunk_metadata(chunk: Dict) -> Dict:
    """
    Prepare the metadata dict that will be stored in Pinecone.
    We intentionally drop fields like 'page' and 'section_path'.
    """
    return {
        "id": chunk["id"],
        "text": chunk["text"],
        "topic": chunk["topic"],
        "doc_title": chunk["doc_title"],
        "source": chunk["source"],
        "chunk_index": int(chunk["chunk_index"]),
    }


def batched(iterable: Iterable, batch_size: int) -> Iterable[List]:
    """Simple utility to iterate over an iterable in fixed-size batches."""
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    index_name = settings.pinecone_index_name
    namespace = settings.pinecone_namespace
    model_name = settings.embedding_model_name

    print(f"Config: index={index_name}, namespace={namespace}, embedding={model_name}")

    print(f"Loading chunks from {CHUNKS_PATH} ...")
    chunks = load_chunks(CHUNKS_PATH)
    if not chunks:
        raise RuntimeError("No chunks loaded; ensure preprocessing ran successfully.")
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model '{model_name}' on CPU ...")
    model = SentenceTransformer(model_name, device="cpu")

    print("Initializing Pinecone client ...")
    pc = Pinecone(api_key=settings.pinecone_api_key)
    if settings.pinecone_index_host:
        index = pc.Index(host=settings.pinecone_index_host)
    else:
        index = pc.Index(index_name)

    total_upserted = 0
    for batch in batched(chunks, BATCH_SIZE):
        texts = [ch["text"] for ch in batch]

        # Encode with L2-normalization for cosine similarity.
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Ensure we have a concrete numpy array (for safety).
        embeddings = np.asarray(embeddings, dtype=np.float32)

        vectors = []
        for ch, emb in zip(batch, embeddings):
            vectors.append(
                {
                    "id": ch["id"],
                    "values": emb.tolist(),
                    "metadata": chunk_metadata(ch),
                }
            )

        # Upsert this batch into Pinecone.
        index.upsert(vectors=vectors, namespace=namespace)
        total_upserted += len(vectors)
        print(f"Upserted {len(vectors)} vectors (total so far: {total_upserted}).")

    print(
        f"Finished indexing. Total vectors upserted to index "
        f"'{index_name}' (namespace '{namespace}'): {total_upserted}"
    )


if __name__ == "__main__":
    main()

