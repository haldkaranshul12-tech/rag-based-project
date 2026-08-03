import time
import datetime
import streamlit as st

from src.document_loader import extract_text
from src.text_splitter import split_text
from src.embeddings import create_embeddings
from src.vector_database import store_embeddings, get_full_document_text, clear_all_documents
from src.retriever import retrieve_documents
from src.llm import generate_answer
from src.guardrails import check_input, redact_pii, check_groundedness

st.set_page_config(page_title="DocMind — Reading Room", page_icon="📖", layout="wide")

# =========================================================
#  STYLING — "Scholar's Reading Room" theme
#  Ink background, parchment text, brass/gold accent,
#  serif display type, source citations as margin notes.
# =========================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;0,700;1,500&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ---------- Header ---------- */
    .reading-header {
        display: flex;
        align-items: baseline;
        gap: 16px;
        border-bottom: 1px solid rgba(201, 162, 39, 0.25);
        padding-bottom: 14px;
        margin-bottom: 22px;
    }
    .reading-mark {
        font-family: 'Lora', serif;
        font-weight: 700;
        font-style: italic;
        font-size: 2.6rem;
        color: #C9A227;
        line-height: 1;
    }
    .reading-title {
        font-family: 'Lora', serif;
        font-weight: 600;
        font-size: 1.7rem;
        color: #E8E3D8;
        margin: 0;
    }
    .reading-subtitle {
        font-family: 'Inter', sans-serif;
        color: #8B8578;
        font-size: 0.88rem;
        letter-spacing: 0.02em;
    }

    /* ---------- Sidebar ---------- */
    .side-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #C9A227;
        margin-bottom: 6px;
        margin-top: 4px;
    }
    .doc-card {
        background: rgba(201, 162, 39, 0.07);
        border: 1px solid rgba(201, 162, 39, 0.28);
        border-left: 3px solid #C9A227;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .doc-card-title {
        font-family: 'Lora', serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: #E8E3D8;
        margin-bottom: 6px;
        word-break: break-word;
    }
    .doc-stat-row {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        color: #A39B8B;
        padding: 2px 0;
    }
    .doc-stat-value {
        color: #C9A227;
    }

    /* ---------- Chat bubbles ---------- */
    div[data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 2px 4px;
    }

    /* ---------- Margin-note source citation ---------- */
    .margin-note {
        background: rgba(47, 82, 51, 0.10);
        border-left: 3px solid #6B8F71;
        border-radius: 4px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #B7C4B9;
    }
    .margin-note-tag {
        color: #6B8F71;
        font-weight: 600;
    }

    /* ---------- Meta chips under an answer ---------- */
    .meta-chip {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #8B8578;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 999px;
        padding: 3px 10px;
        margin-right: 6px;
    }

    /* ---------- Empty state ---------- */
    .empty-state {
        border: 1px dashed rgba(201, 162, 39, 0.35);
        border-radius: 10px;
        padding: 46px 24px;
        text-align: center;
        color: #8B8578;
    }
    .empty-state-title {
        font-family: 'Lora', serif;
        font-style: italic;
        font-size: 1.3rem;
        color: #C9A227;
        margin-bottom: 8px;
    }

    /* ---------- Sample question chips ---------- */
    div[data-testid="column"] button {
        border-radius: 8px !important;
        border: 1px solid rgba(201, 162, 39, 0.3) !important;
        background: rgba(201, 162, 39, 0.06) !important;
        color: #E8E3D8 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="column"] button:hover {
        border-color: #C9A227 !important;
        color: #C9A227 !important;
    }

    hr { border-color: rgba(201, 162, 39, 0.15); }
</style>
""", unsafe_allow_html=True)

# =========================================================
#  SESSION STATE
# =========================================================
defaults = {
    "chat_history": [],
    "documents": [],        # list of dicts: name, word_count, chunk_count, upload_time
    "processed_files": [],  # names already indexed this session, so reruns don't reprocess them
    "pending_question": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def relevance_label(distance):
    """Turn a raw vector distance into a rough, human-readable match label."""
    if distance is None:
        return "—"
    if distance < 0.8:
        return "Strong match"
    elif distance < 1.2:
        return "Moderate match"
    else:
        return "Weak match"


# =========================================================
#  SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('<div class="side-eyebrow">Manuscript</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload documents", type=["pdf", "docx", "png", "jpg", "jpeg"],
        label_visibility="collapsed", accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.processed_files:
                continue  # already indexed this session, skip

            with st.spinner(f"Reading {uploaded_file.name}..."):
                try:
                    text = extract_text(uploaded_file)
                except (ValueError, RuntimeError) as e:
                    st.error(f"⚠️ {uploaded_file.name}: {e}")
                    st.session_state.processed_files.append(uploaded_file.name)
                    continue

            if not text or len(text.strip()) < 20:
                st.error(
                    f"⚠️ No readable text found in {uploaded_file.name}. It may be "
                    "image-based or scanned rather than real text."
                )
                st.session_state.processed_files.append(uploaded_file.name)
                continue

            with st.spinner(f"Splitting {uploaded_file.name} into passages..."):
                chunks = split_text(text)

            with st.spinner("Indexing meaning..."):
                embeddings = create_embeddings(chunks)

            with st.spinner("Filing into the archive..."):
                store_embeddings(chunks, embeddings, uploaded_file.name)

            st.session_state.documents.append({
                "name": uploaded_file.name,
                "word_count": len(text.split()),
                "chunk_count": len(chunks),
                "upload_time": datetime.datetime.now().strftime("%H:%M"),
            })
            st.session_state.processed_files.append(uploaded_file.name)
            st.toast(f"{uploaded_file.name} indexed and ready", icon="📖")

    st.divider()

    if st.session_state.documents:
        st.markdown('<div class="side-eyebrow">Now Reading</div>', unsafe_allow_html=True)
        for doc in st.session_state.documents:
            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-card-title">📄 {doc['name']}</div>
                <div class="doc-stat-row"><span>Words</span><span class="doc-stat-value">{doc['word_count']:,}</span></div>
                <div class="doc-stat-row"><span>Passages</span><span class="doc-stat-value">{doc['chunk_count']}</span></div>
                <div class="doc-stat-row"><span>Indexed at</span><span class="doc-stat-value">{doc['upload_time']}</span></div>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.chat_history:
            avg_time = round(
                sum(c.get("response_time", 0) for c in st.session_state.chat_history)
                / len(st.session_state.chat_history), 2
            )
            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-card-title">💬 Session</div>
                <div class="doc-stat-row"><span>Questions asked</span><span class="doc-stat-value">{len(st.session_state.chat_history)}</span></div>
                <div class="doc-stat-row"><span>Avg. response</span><span class="doc-stat-value">{avg_time}s</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        col_a, col_b = st.columns(2)
        if col_a.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        if st.session_state.chat_history:
            chat_text = "\n\n".join([
                f"Q: {c['question']}\nA: {c['answer']}"
                for c in st.session_state.chat_history
            ])
            col_b.download_button(
                "📥 Export", data=chat_text, file_name="chat_history.txt",
                mime="text/plain", use_container_width=True
            )

        if st.button("🧹 Remove all documents", use_container_width=True):
            clear_all_documents()
            st.session_state.documents = []
            st.session_state.processed_files = []
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown('<div class="side-eyebrow">Model</div>', unsafe_allow_html=True)
    st.caption("🧠 Llama 3.1 8B (Groq) · MiniLM embeddings · ChromaDB")# =========================================================
#  MAIN AREA — HEADER
# =========================================================
st.markdown("""
<div class="reading-header">
    <div class="reading-mark">𝔇</div>
    <div>
        <div class="reading-title">DocMind Reading Room</div>
        <div class="reading-subtitle">Ask your document. Answers stay grounded in its pages — nothing else.</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.documents:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-title">The desk is empty</div>
        Upload one or more documents from the left to begin reading.
    </div>
    """, unsafe_allow_html=True)
else:
    tab_chat, tab_insights = st.tabs(["💬 Chat", "📊 Insights"])

    # ---------------- CHAT TAB ----------------
    with tab_chat:
        if not st.session_state.chat_history:
            st.markdown('<div class="side-eyebrow">Try asking</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            if col1.button("📋 What is this document about?", use_container_width=True):
                st.session_state.pending_question = "What is this document about?"
            if col2.button("🔑 Summarize the key points", use_container_width=True):
                st.session_state.pending_question = "What are the key points in this document?"
            if col3.button("❓ Anything important to know?", use_container_width=True):
                st.session_state.pending_question = "Is there anything important I should know from this document?"
            st.write("")

        for idx, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(chat["question"])
            with st.chat_message("assistant"):
                st.markdown(chat["answer"])
                st.markdown(
                    f'<span class="meta-chip">⏱ {chat.get("response_time", "—")}s</span>'
                    f'<span class="meta-chip">📑 {len(chat.get("sources", []))} passages used</span>',
                    unsafe_allow_html=True
                )
                if chat.get("sources"):
                    with st.expander("📖 Referenced passages"):
                        for meta, dist in zip(chat["sources"], chat.get("distances", [])):
                            st.markdown(f"""
                            <div class="margin-note">
                                <span class="margin-note-tag">§ Chunk {meta.get('chunk_id', '?')}</span>
                                — {meta.get('source', 'unknown')}
                                &nbsp;·&nbsp; {relevance_label(dist)}
                            </div>
                            """, unsafe_allow_html=True)

                fb_col1, fb_col2, fb_spacer = st.columns([1, 1, 8])
                current_fb = chat.get("feedback")
                if fb_col1.button("👍" if current_fb != "up" else "✅👍", key=f"up_{idx}"):
                    st.session_state.chat_history[idx]["feedback"] = "up"
                    st.rerun()
                if fb_col2.button("👎" if current_fb != "down" else "✅👎", key=f"down_{idx}"):
                    st.session_state.chat_history[idx]["feedback"] = "down"
                    st.rerun()

        typed_question = st.chat_input("Ask a question about your document...")
        question = st.session_state.pending_question or typed_question
        st.session_state.pending_question = None

        if question:
            with st.chat_message("user"):
                st.markdown(question)

            is_safe, block_reason = check_input(question)
            if not is_safe:
                with st.chat_message("assistant"):
                    st.error(f"⚠️ {block_reason}")
                st.stop()

            with st.chat_message("assistant"):
                with st.spinner("Turning pages..."):
                    start_time = time.time()

                    # Broad/summary-type questions ("what is this about",
                    # "summarize", "overview", "key points") get the whole
                    # document as context instead of a narrow top-k search,
                    # since a vague query's embedding often matches only one
                    # unrepresentative chunk rather than the full picture.
                    summary_keywords = [
                        "what is this document about", "summarize",
                        "summary", "overview", "key points", "important to know"
                    ]
                    is_summary_question = any(kw in question.lower() for kw in summary_keywords)

                    if is_summary_question:
                        if len(st.session_state.documents) == 1:
                            # Only one document — use just its full text
                            active_name = st.session_state.documents[0]["name"]
                            context = get_full_document_text(active_name)
                            total_chunks = st.session_state.documents[0]["chunk_count"]
                            metadatas = [{"chunk_id": i + 1, "source": active_name}
                                         for i in range(total_chunks)]
                        else:
                            # Multiple documents — combine full text of all of them,
                            # each already grouped under its own header
                            context = get_full_document_text()
                            metadatas = [{"chunk_id": None, "source": doc["name"]}
                                         for doc in st.session_state.documents]
                        distances = [None] * len(metadatas)
                    else:
                        results = retrieve_documents(question)
                        documents = results["documents"][0]
                        metadatas = results["metadatas"][0]
                        distances = results.get("distances", [[]])[0]
                        context = "\n".join(documents)

                    answer = generate_answer(context, question)
                    answer = redact_pii(answer)
                    is_grounded = check_groundedness(answer, context)

                    response_time = round(time.time() - start_time, 2)

                st.markdown(answer)
                if not is_grounded:
                    st.caption(
                        "⚠️ This answer may not be fully grounded in the "
                        "document — double-check it against the referenced passages."
                    )

                valid_sources, valid_distances = [], []
                for meta, dist in zip(metadatas, distances):
                    if meta and "source" in meta:
                        valid_sources.append(meta)
                        valid_distances.append(dist)

                st.markdown(
                    f'<span class="meta-chip">⏱ {response_time}s</span>'
                    f'<span class="meta-chip">📑 {len(valid_sources)} passages used</span>',
                    unsafe_allow_html=True
                )

                if valid_sources:
                    with st.expander("📖 Referenced passages"):
                        for meta, dist in zip(valid_sources, valid_distances):
                            st.markdown(f"""
                            <div class="margin-note">
                                <span class="margin-note-tag">§ Chunk {meta.get('chunk_id', '?')}</span>
                                — {meta.get('source', 'unknown')}
                                &nbsp;·&nbsp; {relevance_label(dist)}
                            </div>
                            """, unsafe_allow_html=True)

            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": valid_sources,
                "distances": valid_distances,
                "response_time": response_time,
                "feedback": None
            })
            st.rerun()

    # ---------------- INSIGHTS TAB ----------------
    with tab_insights:
        total_words = sum(doc["word_count"] for doc in st.session_state.documents)
        total_chunks = sum(doc["chunk_count"] for doc in st.session_state.documents)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Words extracted", f"{total_words:,}")
        c2.metric("Passages stored", total_chunks)
        c3.metric("Questions asked", len(st.session_state.chat_history))
        avg_time = (
            round(sum(c.get("response_time", 0) for c in st.session_state.chat_history)
                  / len(st.session_state.chat_history), 2)
            if st.session_state.chat_history else 0
        )
        c4.metric("Avg. response time", f"{avg_time}s")

        rated = [c for c in st.session_state.chat_history if c.get("feedback")]
        if rated:
            up_count = sum(1 for c in rated if c["feedback"] == "up")
            satisfaction = round((up_count / len(rated)) * 100)
            c5.metric("Helpful rate", f"{satisfaction}%")
        else:
            c5.metric("Helpful rate", "—")

        st.divider()
        st.markdown('<div class="side-eyebrow">Question log</div>', unsafe_allow_html=True)
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history, 1):
                st.markdown(f"**{i}.** {chat['question']}  \n"
                            f"<span class='meta-chip'>⏱ {chat.get('response_time','—')}s</span>"
                            f"<span class='meta-chip'>📑 {len(chat.get('sources', []))} passages</span>",
                            unsafe_allow_html=True)
        else:
            st.caption("No questions asked yet this session.")