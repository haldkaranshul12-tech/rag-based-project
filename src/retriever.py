import re
import math
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

client = chromadb.PersistentClient(path="vector_store/chroma_db")

EMPTY_RESULTS = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

FINAL_TOP_K = 6         # chunks finally sent to the LLM, after reranking
VECTOR_CANDIDATES = 15  # candidates pulled from embedding search
BM25_CANDIDATES = 15    # candidates pulled from keyword search
RRF_K = 60              # standard smoothing constant for Reciprocal Rank Fusion


@lru_cache(maxsize=1)
def _get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_reranker_model():
    # Small, fast cross-encoder good for reranking short passages
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _bm25_search(query, all_docs, all_ids, top_k):
    tokenized_corpus = [_tokenize(doc) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [all_ids[i] for i in ranked_indices]


def _vector_search(query, collection, top_k):
    model = _get_embedding_model()
    query_embedding = model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
    )
    return results["ids"][0] if results.get("ids") else []


def _reciprocal_rank_fusion(rank_lists, k=RRF_K):
    """Merge several ranked ID lists into one fused ranking (RRF)."""
    scores = {}
    for ranked_ids in rank_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)


def retrieve_documents(query):
    try:
        collection = client.get_collection(name="pdf_documents")
    except Exception:
        return EMPTY_RESULTS

    all_data = collection.get()
    all_ids = all_data["ids"]
    all_docs = all_data["documents"]
    all_metas = all_data["metadatas"]

    if not all_ids:
        return EMPTY_RESULTS

    id_to_doc = dict(zip(all_ids, all_docs))
    id_to_meta = dict(zip(all_ids, all_metas))

    # Stage 1: two independent retrieval methods (keyword + semantic)
    vector_ids = _vector_search(query, collection, VECTOR_CANDIDATES)
    bm25_ids = _bm25_search(query, all_docs, all_ids, BM25_CANDIDATES)

    # Stage 2: fuse both ranked lists into one candidate pool
    fused_ids = _reciprocal_rank_fusion([vector_ids, bm25_ids])
    candidate_ids = fused_ids[:max(VECTOR_CANDIDATES, BM25_CANDIDATES)]

    if not candidate_ids:
        return EMPTY_RESULTS

    # Stage 3: rerank the fused candidates with a cross-encoder for precision
    reranker = _get_reranker_model()
    pairs = [[query, id_to_doc[doc_id]] for doc_id in candidate_ids]
    rerank_scores = reranker.predict(pairs)

    scored = sorted(zip(candidate_ids, rerank_scores), key=lambda pair: pair[1], reverse=True)
    top_matches = scored[:FINAL_TOP_K]

    documents = [id_to_doc[doc_id] for doc_id, _ in top_matches]
    metadatas = [id_to_meta[doc_id] for doc_id, _ in top_matches]
    # Convert cross-encoder score (higher = more relevant) into a
    # "distance" (lower = more relevant), so it plugs straight into the
    # existing relevance_label() thresholds in app.py with no changes there.
    distances = [1 - _sigmoid(score) for _, score in top_matches]

    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }