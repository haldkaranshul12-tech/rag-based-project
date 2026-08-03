"""
One-time cleanup: removes chunks stored before metadata (source/chunk_id) was
added to store_embeddings(). These legacy chunks have no metadata.

Run this once from the 'bank project' folder:
    python cleanup_legacy_chunks.py
"""

import chromadb

client = chromadb.PersistentClient(path="vector_store/chroma_db")
collection = client.get_collection(name="pdf_documents")

all_data = collection.get(include=["metadatas"])

legacy_ids = [
    id_ for id_, meta in zip(all_data["ids"], all_data["metadatas"])
    if not meta or "source" not in meta
]

if legacy_ids:
    collection.delete(ids=legacy_ids)
    print(f"Deleted {len(legacy_ids)} legacy chunk(s) with no metadata: {legacy_ids}")
else:
    print("No legacy chunks found. Nothing to clean up.")