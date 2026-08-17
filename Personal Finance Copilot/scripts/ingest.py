"""Offline ingestion: PDFs -> markdown -> chunks -> embeddings -> FAISS on disk.

Run this by hand whenever data/corpus/ changes. It is deliberately NOT part of
app startup: it is slow and, with a paid embedding provider, costs money. The
server only ever loads the finished index.

    python scripts/ingest.py --report     # inspect the corpus, build nothing
    python scripts/ingest.py              # build the index

Extraction uses pymupdf4llm rather than pypdf. That was a measured decision:
on this corpus pypdf reproduced a defective page 48 times (267k characters out
of one page, 84% of the document) and flattened every table. pymupdf4llm
deduplicates the stacked layers and emits real markdown tables. --report shows
the numbers for whatever you have now.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf4llm  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import (  # noqa: E402
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import get_settings  # noqa: E402
from app.tools.embeddings import get_embeddings  # noqa: E402
from app.tools.vectorstore import get_faiss_class  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("ingest")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Markdown headings become chunk boundaries, so a chunk tends to be a whole
# section rather than 1000 arbitrary characters straddling two topics.
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def extract(pdf: Path) -> str:
    """PDF -> markdown, preserving headings and tables."""
    return pymupdf4llm.to_markdown(str(pdf), show_progress=False)


def analyse(pdf: Path, md: str) -> dict:
    """Corpus health check — catches scanned and defective PDFs."""
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    unique = len(set(lines))
    dup_ratio = 1 - unique / len(lines) if lines else 1.0
    return {
        "file": str(pdf.relative_to(pdf.parents[2])),
        "chars": len(md),
        "lines": len(lines),
        "dup_ratio": dup_ratio,
        "tables": md.count("|---"),
        "verdict": (
            "DISCARD (no text — likely scanned)"
            if len(md) < 2000
            else "SUSPECT (heavy duplication)"
            if dup_ratio > 0.40
            else "OK"
        ),
    }


def split(md: str, source: Path, topic: str) -> list[Document]:
    """Structure-aware split, then a size cap for oversized sections."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS, strip_headers=False
    )
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    try:
        sections = header_splitter.split_text(md)
    except Exception:  # documents with no headings at all
        sections = [Document(page_content=md)]

    docs = recursive.split_documents(sections)
    for d in docs:
        # `source` and `topic` are what the model cites, so keep them readable.
        d.metadata.update({"source": source.name, "topic": topic})
    return docs


def dedupe(docs: list[Document]) -> tuple[list[Document], int]:
    """Drop byte-identical chunks.

    A safety net for defects like the 48x-repeated page: extraction usually
    collapses them, but a duplicate chunk that survives would dominate
    retrieval for every query it matches.
    """
    seen: set[str] = set()
    kept = []
    for d in docs:
        key = hashlib.sha256(d.page_content.strip().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        kept.append(d)
    return kept, len(docs) - len(kept)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report", action="store_true", help="Inspect the corpus without building."
    )
    args = ap.parse_args()

    settings = get_settings()
    pdfs = sorted(settings.corpus_dir.rglob("*.pdf"))
    if not pdfs:
        log.error("No PDFs under %s", settings.corpus_dir)
        return 1

    log.info("Found %d PDF(s)", len(pdfs))

    all_docs: list[Document] = []
    rows = []
    for pdf in pdfs:
        md = extract(pdf)
        row = analyse(pdf, md)
        rows.append(row)
        log.info(
            "%-38s %7d chars  dup=%4.1f%%  tables=%2d  %s",
            pdf.name[:38],
            row["chars"],
            row["dup_ratio"] * 100,
            row["tables"],
            row["verdict"],
        )
        if row["verdict"].startswith("DISCARD"):
            log.warning("  skipping %s", pdf.name)
            continue
        all_docs.extend(split(md, pdf, topic=pdf.parent.name))

    if args.report:
        print("\nReport only — nothing built. Re-run without --report to index.")
        return 0

    chunks, dropped = dedupe(all_docs)
    log.info("Chunks: %d (dropped %d duplicates)", len(chunks), dropped)
    if not chunks:
        log.error("Nothing to index.")
        return 1

    log.info(
        "Embedding with %s (first run downloads the model)...",
        settings.embedding_provider,
    )
    embeddings = get_embeddings(settings)

    FAISS = get_faiss_class()
    store = FAISS.from_documents(chunks, embeddings)

    if settings.index_dir.exists():
        shutil.rmtree(settings.index_dir)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(settings.index_dir))

    log.info("Index saved to %s", settings.index_dir)
    log.info("Done. %d chunks from %d document(s).", len(chunks), len(pdfs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
