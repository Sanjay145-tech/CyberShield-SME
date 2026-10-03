"""Step 1: turn the raw ACSC documents into searchable chunks.

Usage:  python src/ingest.py

Reads every .pdf / .html / .htm / .txt / .md in data/raw/, splits it into
overlapping word chunks, tags each chunk with incident types, and writes
data/processed/chunks.jsonl. Each chunk keeps its source file and page so the
assistant can cite exactly where an answer came from.
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from incident import tag_chunk


def clean(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)      # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def read_pdf(path: Path):
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        yield i, clean(page.extract_text() or "")


def read_html(path: Path):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.body or soup
    yield 1, clean(main.get_text("\n"))


def read_text(path: Path):
    yield 1, clean(path.read_text(encoding="utf-8", errors="ignore"))


READERS = {".pdf": read_pdf, ".html": read_html, ".htm": read_html,
           ".txt": read_text, ".md": read_text}


def chunk_words(words, size, overlap):
    step = size - overlap
    for start in range(0, max(len(words) - overlap, 1), step):
        piece = words[start:start + size]
        if len(piece) >= 30:          # drop tiny fragments (headers, page numbers)
            yield " ".join(piece)


def load_source_meta():
    """Optional data/sources.csv with columns: filename,title,url,date_accessed."""
    meta = {}
    if config.SOURCES_FILE.exists():
        with open(config.SOURCES_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meta[row["filename"].strip()] = row
    return meta


def main():
    files = sorted(p for p in config.RAW_DIR.iterdir() if p.suffix.lower() in READERS)
    if not files:
        sys.exit(f"No documents found in {config.RAW_DIR}. Add the ACSC PDFs first.")

    meta = load_source_meta()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as out:
        for path in files:
            m = meta.get(path.name, {})
            title = m.get("title") or path.stem.replace("_", " ").replace("-", " ")
            file_chunks = 0
            for page, text in READERS[path.suffix.lower()](path):
                for chunk in chunk_words(text.split(), config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                    rec = {
                        "id": f"{path.stem}#p{page}#c{file_chunks}",
                        "source": path.name,
                        "title": title,
                        "url": m.get("url", ""),
                        "date_accessed": m.get("date_accessed", ""),
                        "page": page,
                        "text": chunk,
                        "incident_tags": tag_chunk(chunk),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    file_chunks += 1
            print(f"  {path.name}: {file_chunks} chunks")
            n += file_chunks
    print(f"Wrote {n} chunks from {len(files)} files -> {config.CHUNKS_FILE}")
    # Embeddings are now stale; delete so retrieval rebuilds them.
    if config.EMB_FILE.exists():
        config.EMB_FILE.unlink()


if __name__ == "__main__":
    main()
