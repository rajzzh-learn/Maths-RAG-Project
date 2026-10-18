"""
PDF ingestion pipeline for Math Tutor.
Loads all PDFs from configured directories, splits them into chunks,
and persists a ChromaDB vector store for retrieval.
"""
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # load .env for local runs

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
import chromadb.utils.embedding_functions as ef
from langchain_chroma import Chroma
import os

# Force CPU — avoids NotImplementedError on Streamlit Cloud (no MPS/CUDA)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/st_cache")

from math_tutor.config import (
    PDF_DIRS,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_ocr_pages(pdf_path: Path) -> list:
    """Pages for a scanned PDF from its tools/ocr_pdf.py sidecar ("<name>.pdf.ocr.json")."""
    sidecar = pdf_path.with_name(pdf_path.name + ".ocr.json")
    if not sidecar.exists():
        logger.warning("No text in %s and no OCR sidecar - skipped (run tools/ocr_pdf.py)", pdf_path.name)
        return []
    from langchain_core.documents import Document
    texts = json.loads(sidecar.read_text(encoding="utf-8"))
    logger.info("Using OCR text for %s (%d pages)", pdf_path.name, len(texts))
    return [Document(page_content=t, metadata={"source": str(pdf_path), "page": i})
            for i, t in enumerate(texts) if t.strip()]


def load_pdfs(pdf_dirs: list[Path]) -> list:
    """Load every PDF found under the given directories."""
    docs = []
    for directory in pdf_dirs:
        if not directory.exists():
            logger.warning("Directory not found, skipping: %s", directory)
            continue
        for pdf_path in sorted(directory.glob("*.pdf")):
            logger.info("Loading: %s", pdf_path.name)
            pages = PyPDFLoader(str(pdf_path)).load()
            if not any(p.page_content.strip() for p in pages):
                pages = _load_ocr_pages(pdf_path) or pages
            docs.extend(pages)
    logger.info("Total pages loaded: %d", len(docs))
    return docs


class SafeMiniLMEmbeddings(Embeddings):
    """ONNX all-MiniLM-L6-v2 embeddings (same vectors as the HuggingFace model).

    Avoids the PyTorch meta-device NotImplementedError on Streamlit Cloud.
    """
    def __init__(self):
        self._ef = ef.ONNXMiniLM_L6_V2()

    def embed_documents(self, texts):
        return [[float(x) for x in v] for v in self._ef(texts)]

    def embed_query(self, text):
        return [float(x) for x in self._ef([text])[0]]


def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    Build (or load) the ChromaDB vector store.
    - On Streamlit Cloud: the pre-built vectorstore/ folder must be committed
      to the repo. Run `python -m math_tutor.ingest` locally first, then
      commit the vectorstore/ directory before deploying.
    - Locally: set force_rebuild=True to re-ingest all PDFs from scratch.
    """
    embeddings = SafeMiniLMEmbeddings()

    if VECTOR_STORE_DIR.exists() and not force_rebuild:
        logger.info("Loading existing vector store from %s", VECTOR_STORE_DIR)
        return Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embeddings,
        )

    logger.info("Building vector store — this may take a few minutes …")
    raw_docs = load_pdfs(PDF_DIRS)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    for chunk in chunks:
        # Put the file name (which carries the paper year) into the text so that queries
        # like "2023 paper" match the right chunks.
        stem = Path(chunk.metadata.get("source", "")).stem
        page = chunk.metadata.get("page", 0) + 1
        chunk.page_content = f"[{stem} | Page {page}]\n{chunk.page_content}"
    logger.info("Total chunks: %d", len(chunks))

    import shutil
    if VECTOR_STORE_DIR.exists():
        shutil.rmtree(VECTOR_STORE_DIR, ignore_errors=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )
    logger.info("Vector store saved to %s", VECTOR_STORE_DIR)
    return vector_store


if __name__ == "__main__":
    build_vector_store(force_rebuild=True)
