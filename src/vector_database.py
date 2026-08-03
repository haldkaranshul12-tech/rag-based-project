import chromadb

client = chromadb.PersistentClient(path="vector_store/chroma_db")
collection = client.get_or_create_collection(name="pdf_documents")


def store_embeddings(chunks, embeddings, pdf_name):
    """
    Ek document ke chunks collection mein add karta hai. Sirf isi document
    ke purane chunks (agar dobara upload hua ho) delete hote hain — baaki
    documents touch nahi hote, isliye multiple documents saath reh sakte
    hain aur ek saath query ho sakte hain.
    """
    existing = collection.get(where={"source": pdf_name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[f"{pdf_name}_{i}"],
            documents=[chunk],
            embeddings=[embedding.tolist()],
            metadatas=[{"chunk_id": i + 1, "source": pdf_name}]
        )

    return collection


def get_full_document_text(pdf_name=None):
    """
    Saare chunks original order mein fetch karke ek text block banata hai.
    Agar pdf_name diya hai, sirf usi document ke chunks use honge. Agar
    None hai, to STORED SAARE documents ke chunks include honge (har
    document ke header ke saath) — jab multiple PDFs upload ho chuke hon
    aur koi "summarize/overview" type sawaal poocha jaye.
    """
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
    """Collection se saare stored documents hata deta hai."""
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])