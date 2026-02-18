"""
One-off script to debug Pinecone retrieval: index stats, embedding dimension, and a test query.
Run from project root: python -m src.debug_retrieval
"""
from dotenv import load_dotenv

load_dotenv()

from src.config import get_settings

settings = get_settings()

def main():
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer

    print("Config:")
    print("  PINECONE_INDEX_NAME:", settings.pinecone_index_name)
    print("  PINECONE_INDEX_HOST:", settings.pinecone_index_host or "(not set, using name)")
    print("  PINECONE_NAMESPACE:", settings.pinecone_namespace)
    print("  EMBEDDING_MODEL:", settings.embedding_model_name)
    print()

    pc = Pinecone(api_key=settings.pinecone_api_key)
    if settings.pinecone_index_host:
        index = pc.Index(host=settings.pinecone_index_host)
    else:
        index = pc.Index(settings.pinecone_index_name)

    # 1) Index stats
    print("Index stats:")
    try:
        stats = index.describe_index_stats()
        total = getattr(stats, "total_vector_count", None)
        if total is None and isinstance(stats, dict):
            total = stats.get("total_vector_count")
        ns = getattr(stats, "namespaces", None)
        if ns is None and isinstance(stats, dict):
            ns = stats.get("namespaces")
        print("  total_vector_count:", total)
        print("  namespaces:", ns)
        if total == 0:
            print("  >>> Index has 0 vectors. Run: python -m src.index_embeddings")
    except Exception as e:
        print("  ERROR:", e)
        return
    print()

    # 2) Embed a fixed Hebrew query (same as user would type)
    test_query = "מה היא חרדה"
    print("Test query (repr):", repr(test_query))
    print("Test query (len):", len(test_query))
    print("Loading embedding model ...")
    model = SentenceTransformer(settings.embedding_model_name, device="cpu")
    emb = model.encode([test_query], normalize_embeddings=True, show_progress_bar=False)[0]
    vec = emb.tolist()
    print("Embedding dimension:", len(vec))
    print()

    # 3) Query Pinecone
    print("Querying Pinecone (top_k=5, namespace=%r) ..." % settings.pinecone_namespace)
    try:
        resp = index.query(
            vector=vec,
            namespace=settings.pinecone_namespace,
            top_k=5,
            include_metadata=True,
        )
        matches = getattr(resp, "matches", None) or []
        print("Matches returned:", len(matches))
        for i, m in enumerate(matches):
            score = getattr(m, "score", None)
            mid = getattr(m, "id", None)
            print("  [%d] id=%r score=%s" % (i, mid, score))
    except Exception as e:
        print("  ERROR:", e)


if __name__ == "__main__":
    main()
