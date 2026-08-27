from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = { ".txt", ".md", ".pdf", ".html", ".htm" }

@dataclass
class Document:
    source: str
    text: str


def _load_txt_or_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)

def _load_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    return soup.get_text(seperator="\n")

def load_corpus(corpus_path: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(corpus_path.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue

        if path.suffix.lower() in {".txt", ".md"}:
            text = _load_txt_or_md(path)
        elif path.suffix.lower() == ".pdf":
            text = _load_pdf(path)
        elif path.suffix.lower() == ".html":
            text = _load_html(path)

        text = text.strip()
        if(text):
            docs.append(Document(source=str(path.relative_to(corpus_path)), text=text))

    return docs