from io import BytesIO, StringIO
from typing import Dict, List

import pandas as pd
from docx import Document
from pypdf import PdfReader


ALLOWED_EXTENSIONS = {"pdf", "txt", "docx", "xlsx", "csv"}


def _read_txt(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="ignore")


def _read_csv(content: bytes) -> str:
    text = _read_txt(content)
    frame = pd.read_csv(StringIO(text))
    return frame.to_csv(index=False)


def _read_xlsx(content: bytes) -> str:
    frame = pd.read_excel(BytesIO(content))
    return frame.to_csv(index=False)


def _read_docx(content: bytes) -> str:
    doc = Document(BytesIO(content))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)


def _read_pdf(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    pages: List[str] = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return "\n".join([p for p in pages if p])


def extract_text_from_uploaded_file(uploaded_file) -> str:
    filename = uploaded_file.name
    extension = filename.split(".")[-1].lower() if "." in filename else ""

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Formato nao suportado: {extension}")

    content = uploaded_file.getvalue()

    if extension == "txt":
        text = _read_txt(content)
    elif extension == "csv":
        text = _read_csv(content)
    elif extension == "xlsx":
        text = _read_xlsx(content)
    elif extension == "docx":
        text = _read_docx(content)
    elif extension == "pdf":
        text = _read_pdf(content)
    else:
        raise ValueError(f"Formato nao suportado: {extension}")

    text = text.strip()
    if not text:
        raise ValueError(f"Arquivo sem conteudo textual util: {filename}")

    return text


def chunk_text(
    text: str,
    chunk_size: int = 650,
    chunk_overlap: int = 120,
    metadata: Dict = None,
) -> List[dict]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap deve ser menor que chunk_size")

    words = text.split()
    if not words:
        return []

    metadata = metadata or {}
    chunks: List[dict] = []

    start = 0
    step = chunk_size - chunk_overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        content = " ".join(words[start:end]).strip()

        if content:
            chunk_meta = dict(metadata)
            chunk_meta["word_start"] = start
            chunk_meta["word_end"] = end
            chunks.append({"content": content, "metadata": chunk_meta})

        if end == len(words):
            break

        start += step

    return chunks
