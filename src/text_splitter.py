from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    return chunks