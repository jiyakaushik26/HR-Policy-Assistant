from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

import joblib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hr_manual.pdf"
PDF_PATH = DATA_PATH
DB_PATH = PROJECT_ROOT / "chroma_db"
VECTORIZER_PATH = DB_PATH / "local_tfidf_vectorizer.joblib"

SYSTEM_PROMPT = (
    "Answer only using the supplied HR Policy Manual context. Do not use outside "
    "knowledge. If the answer is not explicitly supported by the supplied context, "
    "return exactly: Not found. For supported answers, write 2 or 3 complete, "
    "concise sentences using only facts explicitly stated in the context. Do not "
    "provide source filenames or page numbers; those are supplied separately from "
    "document metadata."
)
MIN_RELEVANCE_SCORE = 0.08
CHROMA_COLLECTION_METADATA = {"hnsw:space": "cosine"}


class LocalTfidfEmbeddings(Embeddings):
    """TF-IDF embeddings sharing one persisted vector space across ingestion and queries."""

    def __init__(self, vectorizer: TfidfVectorizer):
        self.vectorizer = vectorizer

    @classmethod
    def fit(cls, texts: List[str]) -> "LocalTfidfEmbeddings":
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents="unicode",
            sublinear_tf=True,
        )
        vectorizer.fit(texts)
        return cls(vectorizer)

    @classmethod
    def load(cls, path: str | Path) -> "LocalTfidfEmbeddings":
        return cls(joblib.load(path))

    def save(self, path: str | Path) -> None:
        joblib.dump(self.vectorizer, path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.vectorizer.transform([text]).toarray()[0].tolist()


def load_env_file() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def get_gemini_api_key() -> str:
    load_env_file()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured. Add it to the .env file before using Gemini.")
    return api_key


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="models/gemini-3.5-flash-lite",
        temperature=0,
        google_api_key=get_gemini_api_key(),
    )


def _normalize_doc(doc: Document) -> Document:
    metadata = dict(doc.metadata or {})
    if "page" in metadata and isinstance(metadata["page"], int):
        metadata["page"] = max(0, metadata["page"])
    if "source" not in metadata and "source_path" in metadata:
        metadata["source"] = metadata["source_path"]
    if metadata.get("source"):
        metadata["source_filename"] = Path(str(metadata["source"])).name
    doc.metadata = metadata
    return doc


def create_vector_store(pdf_path: str | Path = DATA_PATH, persist_dir: str | Path = DB_PATH, reset: bool = False) -> Chroma:
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    persist_path = Path(persist_dir)
    if reset and persist_path.exists():
        shutil.rmtree(persist_path)

    loader = PyPDFLoader(str(pdf_file))
    raw_documents = [_normalize_doc(doc) for doc in loader.load()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(raw_documents)
    if not chunks:
        raise ValueError(f"No text chunks were extracted from PDF: {pdf_file}")

    persist_path.mkdir(parents=True, exist_ok=True)
    embeddings = LocalTfidfEmbeddings.fit([chunk.page_content for chunk in chunks])
    vectorizer_path = persist_path / VECTORIZER_PATH.name
    embeddings.save(vectorizer_path)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
        collection_name="hr_policy_manual",
        collection_metadata=CHROMA_COLLECTION_METADATA,
    )
    return vector_store


def load_vector_store(persist_dir: str | Path = DB_PATH) -> Chroma:
    persist_path = Path(persist_dir)
    vectorizer_path = persist_path / VECTORIZER_PATH.name
    if not persist_path.exists() or not any(persist_path.iterdir()) or not vectorizer_path.exists():
        return create_vector_store(pdf_path=DATA_PATH, persist_dir=persist_path)

    return Chroma(
        persist_directory=str(persist_path),
        embedding_function=LocalTfidfEmbeddings.load(vectorizer_path),
        collection_name="hr_policy_manual",
        collection_metadata=CHROMA_COLLECTION_METADATA,
    )


def retrieve_documents(question: str, k: int = 4, vector_store: Chroma | None = None) -> List[Document]:
    if not question or not question.strip():
        return []
    store = vector_store or load_vector_store()
    scored_documents = store.similarity_search_with_relevance_scores(question, k=k)
    return [
        _normalize_doc(document)
        for document, score in scored_documents
        if score >= MIN_RELEVANCE_SCORE and document.page_content.strip()
    ]


def format_context(documents: List[Document]) -> str:
    if not documents:
        return ""

    parts: List[str] = []
    for doc in documents:
        source = doc.metadata.get("source_filename")
        original_page = doc.metadata.get("page")
        content = (doc.page_content or "").strip()
        if not source or not isinstance(original_page, int) or original_page < 0 or not content:
            continue
        parts.append(f"Source: {source} | Page {original_page + 1}\n{content}")
    return "\n\n---\n\n".join(parts)


def _answer_sentence_count(answer: str) -> int:
    return len([sentence for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()) if sentence])


def _relevant_evidence_lines(question: str, answer: str, document: Document) -> List[str]:
    question_terms = set(re.findall(r"[a-z]{3,}", question.casefold()))
    answer_terms = set(re.findall(r"[a-z]{3,}", answer.casefold()))
    relevant_terms = question_terms | answer_terms
    answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", answer))

    ranked_lines = []
    for index, line in enumerate(document.page_content.splitlines()):
        cleaned = re.sub(r"\s+", " ", line).strip()
        if not cleaned:
            continue
        line_terms = set(re.findall(r"[a-z]{3,}", cleaned.casefold()))
        line_numbers = set(re.findall(r"\d+(?:\.\d+)?", cleaned))
        term_matches = len(line_terms & relevant_terms)
        number_matches = len(line_numbers & answer_numbers)
        if term_matches or number_matches:
            ranked_lines.append((number_matches * 3 + term_matches, index, cleaned))

    ranked_lines.sort(key=lambda item: (-item[0], item[1]))
    return [line for _, _, line in ranked_lines[:3]]


def answer_question(
    question: str,
    *,
    vector_store: Chroma | None = None,
    llm: ChatGoogleGenerativeAI | None = None,
) -> Dict[str, Any]:
    user_question = (question or "").strip()
    if not user_question:
        return {"answer": "Not found", "source_references": [], "evidence": []}

    doc_matches = retrieve_documents(user_question, k=4, vector_store=vector_store)
    if not doc_matches:
        return {"answer": "Not found", "source_references": [], "evidence": []}

    context_text = format_context(doc_matches)
    if not context_text:
        return {"answer": "Not found", "source_references": [], "evidence": []}
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Supplied HR Policy Manual context:\n{context}\n\n"
                "Question: {question}\n\n"
                "If supported, answer in 2 or 3 complete sentences and include "
                "only policy facts stated in the supplied context. Otherwise output "
                "exactly: Not found.",
            ),
        ]
    )
    answer = (prompt | (llm or get_llm()) | StrOutputParser()).invoke(
        {"context": context_text, "question": user_question}
    )
    cleaned = (answer or "").strip()
    if not cleaned or cleaned.casefold() in {"not found", "not found."}:
        return {"answer": "Not found", "source_references": [], "evidence": []}
    if not 2 <= _answer_sentence_count(cleaned) <= 3:
        return {"answer": "Not found", "source_references": [], "evidence": []}

    answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", cleaned))
    if answer_numbers:
        supporting_documents = [
            doc for doc in doc_matches
            if answer_numbers.issubset(set(re.findall(r"\d+(?:\.\d+)?", doc.page_content)))
        ]
        if not supporting_documents:
            return {"answer": "Not found", "source_references": [], "evidence": []}
    else:
        supporting_documents = doc_matches

    evidence_candidates = []
    for doc in supporting_documents:
        source = doc.metadata.get("source_filename")
        original_page = doc.metadata.get("page")
        evidence_lines = _relevant_evidence_lines(user_question, cleaned, doc)
        if (
            not source
            or not isinstance(original_page, int)
            or original_page < 0
            or not evidence_lines
        ):
            continue
        page = original_page + 1
        for line in evidence_lines:
            line_terms = set(re.findall(r"[a-z]{3,}", line.casefold()))
            line_numbers = set(re.findall(r"\d+(?:\.\d+)?", line))
            question_terms = set(re.findall(r"[a-z]{3,}", user_question.casefold()))
            answer_terms = set(re.findall(r"[a-z]{3,}", cleaned.casefold()))
            score = (
                3 * len(line_numbers & answer_numbers)
                + len(line_terms & question_terms)
                + len(line_terms & answer_terms)
            )
            evidence_candidates.append((score, source, page, line))

    selected_evidence = []
    seen_evidence = set()
    for _, source, page, line in sorted(evidence_candidates, key=lambda candidate: -candidate[0]):
        key = line.casefold()
        if key in seen_evidence:
            continue
        seen_evidence.add(key)
        selected_evidence.append({"source": source, "page": page, "text": line})
        if len(selected_evidence) == 3:
            break

    if not selected_evidence:
        return {"answer": "Not found", "source_references": [], "evidence": []}

    source_references = []
    seen_references = set()
    for item in selected_evidence:
        reference = (item["source"], item["page"])
        if reference not in seen_references:
            seen_references.add(reference)
            source_references.append({"source": item["source"], "page": item["page"]})

    return {
        "answer": cleaned,
        "source_references": source_references,
        "evidence": selected_evidence,
    }


if __name__ == "__main__":
    print("RAG module loaded. Use `python -m src.ingest` to ingest the PDF or `python -c \"from src.rag import answer_question; print(answer_question('...'))\"`.")
