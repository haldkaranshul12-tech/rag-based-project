import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Simple heuristics for detecting a "heading" line in extracted text:
# markdown headings, numbered sections ("1." / "2.3"), short ALL CAPS
# lines, and short Title Case lines with no ending punctuation.
_HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+.+"),
    re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z].{0,80}$"),
    re.compile(r"^[A-Z][A-Za-z0-9 ,&'/-]{2,60}$"),
    re.compile(r"^[A-Z\s]{4,60}$"),
]


def _is_heading(line):
    line = line.strip()
    if not line or len(line) > 90:
        return False
    if line.endswith((".", ",", ";", ":")) and not re.match(r"^\d+(\.\d+)*\.\s", line):
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)


def _split_into_sections(text):
    """
    Splits a block of text into (heading, body) sections using simple
    heading heuristics. Text before the first detected heading is kept as
    its own section with heading=None.
    """
    lines = text.split("\n")
    sections = []
    current_heading = None
    current_body = []

    for line in lines:
        if _is_heading(line):
            if current_body or current_heading:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body or current_heading:
        sections.append((current_heading, "\n".join(current_body).strip()))

    return sections if sections else [(None, text)]


def split_text(text):
    """Backwards-compatible: chunks a single string, no page/heading metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)


def split_pages(pages):
    """
    Chunks a list of per-page strings while keeping track of which page and
    which section heading each chunk came from. Headings are kept attached
    to their own content (and re-prefixed onto every resulting chunk if a
    section itself has to be split further), so a chunk is never separated
    from the heading that gives it context.

    Returns three parallel lists: (chunks, page_numbers, headings).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    page_numbers = []
    headings = []

    for page_number, page_text in enumerate(pages, start=1):
        if not page_text or not page_text.strip():
            continue

        for heading, body in _split_into_sections(page_text):
            full_text = f"{heading}\n{body}" if heading else body
            if not full_text.strip():
                continue

            section_chunks = splitter.split_text(full_text)

            for chunk in section_chunks:
                if heading and not chunk.startswith(heading):
                    chunk = f"{heading}\n{chunk}"
                chunks.append(chunk)
                page_numbers.append(page_number)
                headings.append(heading)

    return chunks, page_numbers, headings