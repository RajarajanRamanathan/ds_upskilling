import os
import fitz
from docx import Document
import pytesseract
from PIL import Image


# ------------------------------------------------------------
# TESSERACT CONFIGURATION
# ------------------------------------------------------------

TESSERACT_PATH = r"C:\Users\rajarajan.ramanathan\AppData\Local\Tesseract-OCR"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ------------------------------------------------------------
# OCR
# ------------------------------------------------------------

def perform_ocr(page):
    """
    Perform OCR on a PDF page.
    """

    pix = page.get_pixmap(
        matrix=fitz.Matrix(2, 2)
    )

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    text = pytesseract.image_to_string(
        image
    )

    return text.strip()


# ------------------------------------------------------------
# PDF
# ------------------------------------------------------------

def extract_pdf_pages(file_path):
    """
    Extract PDF content page by page.

    Returns:

    [
        {
            "page_number": 1,
            "text": "...",
            "source": "text"
        },
        ...
    ]
    """

    pages = []

    document = fitz.open(file_path)

    try:

        for page_number, page in enumerate(
            document,
            start=1
        ):

            page_text = page.get_text().strip()

            # ---------------------------------------------
            # Normal PDF text
            # ---------------------------------------------

            if len(page_text) > 20:

                pages.append({
                    "page_number": page_number,
                    "text": page_text,
                    "source": "text"
                })

            # ---------------------------------------------
            # OCR fallback
            # ---------------------------------------------

            else:

                ocr_text = perform_ocr(page)

                pages.append({
                    "page_number": page_number,
                    "text": ocr_text,
                    "source": "OCR"
                })

    finally:

        document.close()

    return pages


# ------------------------------------------------------------
# DOCX
# ------------------------------------------------------------

def extract_docx_pages(file_path):
    """
    DOCX does not have reliable page boundaries
    during normal python-docx extraction.

    For now, treat the entire document as page 1.
    We will improve this later if required.
    """

    document = Document(
        file_path
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append(text)

    full_text = "\n".join(
        paragraphs
    )

    return [
        {
            "page_number": 1,
            "text": full_text,
            "source": "text"
        }
    ]


# ------------------------------------------------------------
# TXT
# ------------------------------------------------------------

def extract_txt_pages(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    return [
        {
            "page_number": 1,
            "text": text,
            "source": "text"
        }
    ]


# ------------------------------------------------------------
# PAGE-AWARE EXTRACTION
# ------------------------------------------------------------

def extract_pages(file_path, file_type):

    file_type = file_type.lower()

    if file_type == "pdf":

        return extract_pdf_pages(
            file_path
        )

    elif file_type == "docx":

        return extract_docx_pages(
            file_path
        )

    elif file_type == "txt":

        return extract_txt_pages(
            file_path
        )

    else:

        raise ValueError(
            f"Unsupported file type: {file_type}"
        )


# ------------------------------------------------------------
# BACKWARD COMPATIBILITY
# ------------------------------------------------------------

def extract_text(file_path, file_type):

    pages = extract_pages(
        file_path,
        file_type
    )

    text_parts = []

    for page in pages:

        text_parts.append(
            f"--- Page {page['page_number']} ---"
        )

        text_parts.append(
            page["text"]
        )

    return "\n".join(
        text_parts
    )