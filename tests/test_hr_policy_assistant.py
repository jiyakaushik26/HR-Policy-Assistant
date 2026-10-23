import pytest
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

from src.rag import (
    DATA_PATH,
    LocalTfidfEmbeddings,
    MIN_RELEVANCE_SCORE,
    PROJECT_ROOT,
    SYSTEM_PROMPT,
    answer_question,
    create_vector_store,
    get_gemini_api_key,
    retrieve_documents,
)


class ScoredStore:
    def __init__(self, documents_and_scores):
        self.documents_and_scores = documents_and_scores

    def similarity_search_with_relevance_scores(self, question, k):
        return self.documents_and_scores[:k]


def policy_document(page=41, text="Permanent employees receive 18 days of Privilege Leave per year."):
    return Document(
        page_content=text,
        metadata={
            "source": str(DATA_PATH),
            "source_filename": DATA_PATH.name,
            "page": page,
        },
    )


def test_pdf_loading_and_original_page_metadata():
    documents = PyPDFLoader(str(DATA_PATH)).load()
    assert documents
    assert documents[0].metadata["source"] == str(DATA_PATH)
    assert documents[0].metadata["page"] == 0
    assert "HR Manual" in documents[0].page_content


def test_chroma_ingestion_persists_chunks_and_pdf_metadata(tmp_path):
    store = create_vector_store(DATA_PATH, tmp_path / "chroma_db", reset=True)
    stored = store.get()

    assert stored["documents"]
    assert all(metadata["source_filename"] == DATA_PATH.name for metadata in stored["metadatas"])
    assert all(isinstance(metadata["page"], int) for metadata in stored["metadatas"])
    retrieved = store.similarity_search("HR Manual", k=3)
    assert retrieved
    assert retrieved[0].metadata["source_filename"] == DATA_PATH.name
    assert (tmp_path / "chroma_db" / "local_tfidf_vectorizer.joblib").is_file()
    assert any(
        document.metadata["page"] == 59
        for document in retrieve_documents(
            "How many days of Privilege Leave (PL) does a permanent employee get per year?",
            vector_store=store,
        )
    )
    assert any(
        document.metadata["page"] == 61
        for document in retrieve_documents(
            "How many days of maternity leave are given?",
            vector_store=store,
        )
    )
    assert not retrieve_documents("What is the current GST rate in India?", vector_store=store)


def test_local_tfidf_embeddings_share_persisted_vector_space(tmp_path):
    embeddings = LocalTfidfEmbeddings.fit(["Privilege Leave is 12 days per year.", "Probation lasts 3 months."])
    path = tmp_path / "vectorizer.joblib"
    embeddings.save(path)
    restored = LocalTfidfEmbeddings.load(path)

    assert restored.embed_query("Privilege Leave") == embeddings.embed_query("Privilege Leave")
    assert len(restored.embed_documents(["Privilege Leave"])[0]) == len(embeddings.embed_query("Privilege Leave"))


def test_retrieval_filters_insufficient_relevance():
    relevant = policy_document()
    irrelevant = policy_document(page=3, text="Office cafeteria hours are posted at reception.")
    store = ScoredStore([(relevant, MIN_RELEVANCE_SCORE), (irrelevant, MIN_RELEVANCE_SCORE - 0.01)])

    assert retrieve_documents("Privilege Leave", vector_store=store) == [relevant]


def test_supported_answer_uses_only_retrieved_context_and_returns_evidence():
    document = policy_document(text="Permanent employees receive 18 days of Privilege Leave per year.\nThis leave applies for one calendar year.")
    captured_prompts = []
    llm = RunnableLambda(
        lambda prompt: captured_prompts.append(prompt.to_string())
        or "Permanent employees receive 18 days of Privilege Leave per year. The policy applies this entitlement for one calendar year."
    )
    store = ScoredStore([(document, 0.8)])

    result = answer_question("How much Privilege Leave?", vector_store=store, llm=llm)

    assert result["answer"] == (
        "Permanent employees receive 18 days of Privilege Leave per year. "
        "The policy applies this entitlement for one calendar year."
    )
    assert result["source_references"] == [{"source": DATA_PATH.name, "page": 42}]
    assert len(result["evidence"]) <= 3
    assert all(item["source"] == DATA_PATH.name and item["page"] == 42 for item in result["evidence"])
    assert any("18 days of Privilege Leave" in item["text"] for item in result["evidence"])
    assert "Answer only using the supplied HR Policy Manual context." in SYSTEM_PROMPT
    assert "Do not use outside knowledge." in SYSTEM_PROMPT
    assert "2 or 3 complete" in SYSTEM_PROMPT
    assert document.page_content in captured_prompts[0]
    assert DATA_PATH.name in captured_prompts[0]
    assert "Page 42" in captured_prompts[0]


def test_unsupported_gst_question_returns_exact_not_found_without_calling_llm():
    irrelevant = policy_document(page=5, text="The office has a visitor parking policy.")
    store = ScoredStore([(irrelevant, MIN_RELEVANCE_SCORE - 0.05)])
    llm = RunnableLambda(lambda prompt: pytest.fail("Gemini must not be called without relevant context"))

    result = answer_question("What is the current GST rate in India?", vector_store=store, llm=llm)

    assert result == {"answer": "Not found", "source_references": [], "evidence": []}


def test_model_not_found_is_exact_and_has_no_source_claim():
    document = policy_document()
    llm = RunnableLambda(lambda prompt: "Not found")
    result = answer_question("Question not answered in this context", vector_store=ScoredStore([(document, 0.8)]), llm=llm)

    assert result == {"answer": "Not found", "source_references": [], "evidence": []}


def test_supported_one_sentence_model_response_fails_closed():
    document = policy_document()
    llm = RunnableLambda(lambda prompt: "18 days of Privilege Leave.")
    result = answer_question(
        "How much Privilege Leave?",
        vector_store=ScoredStore([(document, 0.8)]),
        llm=llm,
    )

    assert result == {"answer": "Not found", "source_references": [], "evidence": []}


def test_numeric_answer_references_only_chunks_supporting_all_answer_numbers():
    overview = policy_document(page=59, text="Maternity Leave is available. Privilege Leave is 12 days.")
    entitlement = policy_document(page=61, text="Maternity Leave is 12 weeks (90 days) with pay.\nThe policy provides this leave with pay.")
    llm = RunnableLambda(lambda prompt: "Maternity leave is 12 weeks, or 90 days. The policy provides this leave with pay.")
    store = ScoredStore([(overview, 0.8), (entitlement, 0.7)])

    result = answer_question("How many days of maternity leave?", vector_store=store, llm=llm)

    assert result["answer"] == "Maternity leave is 12 weeks, or 90 days. The policy provides this leave with pay."
    assert result["source_references"] == [{"source": DATA_PATH.name, "page": 62}]
    assert all(item["source"] == DATA_PATH.name and item["page"] == 62 for item in result["evidence"])
    assert any("12 weeks (90 days)" in item["text"] for item in result["evidence"])


def test_streamlit_displays_answer_source_page_and_evidence(monkeypatch):
    monkeypatch.setattr(
        "src.rag.answer_question",
        lambda question: {
            "answer": "Grounded test answer",
            "source_references": [{"source": DATA_PATH.name, "page": 42}],
            "evidence": [{
                "source": DATA_PATH.name,
                "page": 42,
                "text": "Supporting policy text from the PDF.",
            }],
        },
    )
    app = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py")).run()
    app.text_input[0].set_value("Test question")
    app.button[0].click().run()
    rendered_text = "\n".join(item.value for item in [*app.markdown, *app.text])

    assert "Grounded test answer" in rendered_text
    assert DATA_PATH.name in rendered_text
    assert "42" in rendered_text
    assert "Supporting policy text from the PDF." in rendered_text
    assert not app.exception


def test_missing_api_key_behavior(monkeypatch):
    monkeypatch.setattr("src.rag.load_env_file", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        get_gemini_api_key()
