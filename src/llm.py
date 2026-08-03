import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


def generate_answer(context, question):

    # Safety check 1: if no context was retrieved at all, don't even call the LLM
    if not context or len(context.strip()) < 20:
        return "Content not available in this document."

    prompt = f"""
    You are a helpful PDF assistant.
    Answer ONLY using the given context below.
    Do not use any outside knowledge, even if you already know the answer.

    IMPORTANT: If the context mentions something under a heading like
    "Suggested Next Steps", "Future Work", "Roadmap", "To-Do", or similar,
    treat it as NOT yet implemented — do not say "yes" to whether that
    feature currently exists. Only confirm a feature exists if the context
    describes it as already built/working (e.g. under "Features Implemented"
    or "Current Status").

    If the context does not contain enough information to answer the question,
    reply with EXACTLY this sentence and nothing else:
    "Content not available in this document."

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    answer = response.content.strip()

    return answer