# 🧠 DocMind — PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any text-based PDF and ask questions about it. Answers are grounded strictly in the uploaded document's content — not the model's general knowledge.

---

## 📖 Project Overview

DocMind extracts text from an uploaded PDF, splits it into chunks, converts those chunks into vector embeddings, stores them in a local vector database (ChromaDB), and retrieves the most relevant chunks to answer a user's question using an LLM (Llama 3.1 via Groq).

Each new PDF upload clears out any previously stored document, so answers are always based on the currently active PDF only — never mixed with old data.

---

## ✨ Features

- 📄 **PDF Upload** — Upload any text-based PDF directly from the browser
- ✂️ **Automatic Chunking** — Splits document text into manageable overlapping chunks for accurate retrieval
- 🔍 **Semantic Search** — Uses sentence embeddings + ChromaDB to find the most relevant chunks for a question
- 🤖 **Grounded Answers** — LLM answers only from retrieved context, with a safe fallback ("Content not available in this document.") when the answer isn't in the PDF
- 📚 **Source Citations** — Every answer shows which document and chunk it came from
- 💬 **Chat History** — Full conversation view with chat-bubble UI
- 🧹 **Fresh Start on Each Upload** — Old document data is automatically cleared so answers never leak between different PDFs
- ⚠️ **Blank / Image-PDF Validation** — Detects PDFs with no extractable text (e.g. scanned/image-based PDFs) and warns the user immediately instead of producing an ungrounded answer
- 🎨 **Custom Themed UI** — Dark navy/teal branded interface built with Streamlit

---

## 🖼️ Screenshots

> Place your screenshots inside the `screenshots/` folder using the filenames below, and they'll render here.

**Home Page**
![Home Page](screenshots/home_page.png)

**PDF Upload**
![PDF Upload](screenshots/pdf_upload.png)

**Successful Answer**
![Successful Answer](screenshots/successful_answer.png)

**Source Citation**
![Source Citation](screenshots/source_citation.png)

**Chat History**
![Chat History](screenshots/chat_history.png)

**Blank PDF Validation**
![Blank PDF Validation](screenshots/blank_pdf_validation.png)

---

## 🔄 Project Flow

```
1. User uploads a PDF
        ↓
2. Text is extracted from the PDF (pypdf)
        ↓
   ⚠️ If no readable text is found → show warning, stop here
        ↓
3. Text is split into overlapping chunks (LangChain text splitter)
        ↓
4. Each chunk is converted into a vector embedding (SentenceTransformer)
        ↓
5. Old data is cleared, new embeddings are stored in ChromaDB
        ↓
6. User asks a question
        ↓
7. Question is embedded and matched against stored chunks (vector similarity search)
        ↓
8. Top matching chunks are sent to the LLM as context
        ↓
   ⚠️ If context is empty/too short → return fallback message, skip LLM call
        ↓
9. LLM (Llama 3.1 via Groq) generates an answer strictly from the given context
        ↓
10. Answer + source chunks are displayed in the chat UI
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| PDF Text Extraction | pypdf |
| Text Chunking | LangChain Text Splitters |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB (persistent, local) |
| LLM | Llama 3.1 8B Instant (via Groq API) |

---

## ⚙️ Installation & Setup

### 1. Clone / open the project folder
Navigate to the project folder in your terminal.

### 2. Create a virtual environment (Windows)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install streamlit chromadb sentence-transformers langchain-groq langchain-text-splitters python-dotenv pypdf
```

### 4. Set up your API key
Create a `.env` file in the project root with your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### 6. Using the app
1. Upload a text-based PDF from the sidebar.
2. Wait for it to show "Ready — X chunks stored."
3. Type your question in the chat box at the bottom.
4. View the answer along with its source chunks in the expandable "Sources" section.

---

## 🚀 Future Enhancements

- 🖼️ **OCR support** for image-based / scanned PDFs (using `pdf2image` + `pytesseract`), so screenshots or handwritten-note PDFs can also be processed
- 📂 **Multi-PDF support** — ability to store and switch between multiple uploaded documents instead of clearing on every upload
- 📊 **Better summarization handling** — dedicated logic for "what is this PDF about" style questions instead of relying purely on top-k vector search
- 🔐 **User authentication** — per-user document storage and chat history
- 📥 **Export chat history** — download Q&A sessions as PDF or text
- 🌐 **Multi-language support** — for PDFs and questions in languages other than English
- 📈 **Analytics dashboard** — track most-asked questions, document usage, and retrieval accuracy

---

## 📌 Notes

- This project currently supports **text-based PDFs only**. Image-based or scanned PDFs (e.g., PDFs made from screenshots/slides) will not have extractable text and will trigger a warning message instead of an answer.
- Answers are strictly limited to the content of the uploaded document — the assistant is designed to avoid using outside/general knowledge.