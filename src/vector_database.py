import chromadb

client = chromadb.PersistentClient(path="vector_store/chroma_db")
collection = client.get_or_create_collection(name="pdf_documents")


def store_embeddings(chunks, embeddings, pdf_name, page_numbers=None, doc_type=None, headings=None):
    existing = collection.get(where={"source": pdf_name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    if page_numbers is None:
        page_numbers = [None] * len(chunks)
    if headings is None:
        headings = [None] * len(chunks)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        metadata = {
            "chunk_id": i + 1,
            "source": pdf_name,
            "doc_type": doc_type or "unknown",
        }
        if page_numbers[i] is not None:
            metadata["page"] = page_numbers[i]
        if headings[i]:
            metadata["heading"] = headings[i]

        collection.add(
            ids=[f"{pdf_name}_{i}"],
            documents=[chunk],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )

    return collection


def get_full_document_text(pdf_name=None):
    data = collection.get(where={"source": pdf_name}) if pdf_name else collection.get()

    grouped = {}
    for meta, doc in zip(data["metadatas"], data["documents"]):
        source = meta.get("source", "unknown")
        grouped.setdefault(source, []).append((meta.get("chunk_id", 0), doc))

    blocks = []
    for source, items in grouped.items():
        items.sort(key=lambda pair: pair[0])
        text = "\n".join(doc for _, doc in items)
        blocks.append(f"--- Document: {source} ---\n{text}")

    return "\n\n".join(blocks)


def clear_all_documents():
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])