"""
Unified text extraction for PDF, DOCX, and image files.

DOCX and image extraction use optional libraries (python-docx, pytesseract,
Pillow, PyMuPDF). Their imports are deferred into each function so that a
missing library only affects that specific file type — PDF upload (the
common case) keeps working regardless.

Image (and scanned-PDF) extraction also needs the Tesseract OCR *engine*
installed separately on your system — the pytesseract Python package alone
is just a wrapper around it and won't work without it.
Windows installer: https://github.com/UB-Mannheim/tesseract/wiki
"""

from pypdf import PdfReader


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if len(text.strip()) >= 20:
        return text

    # No real text layer found (likely a scanned PDF) — fall back to OCR
    return extract_text_from_scanned_pdf(file)


def extract_text_from_scanned_pdf(file):
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import pytesseract
        import io
    except ImportError:
        raise RuntimeError(
            "OCR support for scanned PDFs needs PyMuPDF, Pillow, and "
            "pytesseract. In your activated venv, run: "
            "pip install PyMuPDF Pillow pytesseract"
        )

    file.seek(0)
    pdf_doc = fitz.open(stream=file.read(), filetype="pdf")

    text = ""
    for page in pdf_doc:
        pix = page.get_pixmap(dpi=200)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            text += pytesseract.image_to_string(image) + "\n"
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                "Tesseract OCR engine not found on this system. Install it "
                "separately (not just via pip) — see "
                "https://github.com/UB-Mannheim/tesseract/wiki for Windows, "
                "then set pytesseract.pytesseract.tesseract_cmd to its install path."
            )

    return text


def extract_text_from_docx(file):
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError(
            "python-docx is not installed. Run this in your activated venv: "
            "pip install python-docx"
        )

    document = Document(file)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]

    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text)

    return "\n".join(paragraphs)


def extract_text_from_image(file):
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        raise RuntimeError(
            "pytesseract/Pillow are not installed. Run this in your "
            "activated venv: pip install pytesseract Pillow"
        )

    # If Windows doesn't auto-detect Tesseract, uncomment and set your path:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    image = Image.open(file)
    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR engine not found on this system. Install it "
            "separately (not just via pip) — see "
            "https://github.com/UB-Mannheim/tesseract/wiki for Windows, "
            "then set pytesseract.pytesseract.tesseract_cmd to its install path."
        )


def extract_text(uploaded_file):
    """
    Looks at the uploaded file's extension and calls the right extractor.
    `uploaded_file` is the Streamlit UploadedFile object.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif name.endswith((".png", ".jpg", ".jpeg")):
        return extract_text_from_image(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")