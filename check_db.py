import chromadb

client = chromadb.PersistentClient(path="vector_store/chroma_db")
collection = client.get_collection(name="pdf_documents")

data = collection.get()
sources = set(m["source"] for m in data["metadatas"] if m and "source" in m)

print(f"Total chunks in DB: {len(data['ids'])}")
print(f"PDFs found in DB: {sources}")