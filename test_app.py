"""Unit tests for the Streamlit-based Agentic RAG application.

Tests cover:
1. ConfigLoader - config.yaml loading and streamlit section
2. RAGApplication - backend logic (process_pdfs, chat, search, stats)
3. Streamlit UI entry point - main() imports and renders without error
4. No residual Gradio references in source files
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ──────────────────────────────────────────────
# 1. ConfigLoader Tests
# ──────────────────────────────────────────────

class TestConfigLoader:
    def setup_method(self):
        # Reset singleton so each test gets fresh config
        import config_loader
        config_loader._config = None
        self.config = config_loader.get_config()

    def test_config_loads_successfully(self):
        assert self.config is not None
        assert self.config.config is not None

    def test_streamlit_section_exists(self):
        st_config = self.config.get("streamlit")
        assert st_config is not None, "config.yaml must have a 'streamlit' section"

    def test_no_gradio_section(self):
        gradio_config = self.config.get("gradio")
        assert gradio_config is None, "config.yaml should not have a 'gradio' section"

    def test_streamlit_server_config(self):
        assert self.config.get("streamlit.server.address") == "127.0.0.1"
        assert self.config.get("streamlit.server.port") == 8501

    def test_streamlit_ui_config(self):
        assert self.config.get("streamlit.ui.title") == "Agentic RAG System"
        assert self.config.get("streamlit.ui.layout") == "wide"

    def test_streamlit_defaults(self):
        defaults = self.config.get("streamlit.defaults")
        assert defaults is not None
        assert defaults["chunk_size"] == 512
        assert defaults["temperature"] == 0
        assert defaults["top_k"] == 3
        assert defaults["search_top_k"] == 10

    def test_streamlit_slider_ranges(self):
        defaults = self.config.get("streamlit.defaults")
        assert defaults["chunk_size_range"] == [128, 1024, 128]
        assert defaults["temperature_range"] == [0, 1, 0.1]
        assert defaults["top_k_range"] == [1, 20, 1]
        assert defaults["search_top_k_range"] == [5, 50, 5]

    def test_dot_notation_access(self):
        assert self.config.get("models.embedding.name") == "sentence-transformers/all-MiniLM-L6-v2"
        assert self.config.get("models.llm.default") == "llama-3.3-70b-versatile"

    def test_get_path_returns_path_object(self):
        p = self.config.get_path("paths.faiss_index")
        assert isinstance(p, Path)

    def test_missing_key_returns_default(self):
        assert self.config.get("nonexistent.key") is None
        assert self.config.get("nonexistent.key", "fallback") == "fallback"

    def test_paths_property(self):
        paths = self.config.paths
        assert "faiss_index" in paths
        assert "bm25_index" in paths
        assert "annotated_chunks" in paths

    def test_models_property(self):
        models = self.config.models
        assert "embedding" in models
        assert "llm" in models
        assert "reranker" in models

    def test_prompts_property(self):
        prompts = self.config.prompts
        assert "grade" in prompts
        assert "rewrite" in prompts
        assert "generate" in prompts
        assert "simple_rag_system" in prompts


# ──────────────────────────────────────────────
# 2. RAGApplication Backend Tests
# ──────────────────────────────────────────────

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "paths.temp_uploads": "/tmp/test_rag_uploads",
        "paths.faiss_index": "vectorstores/faiss.index",
        "paths.faiss_metadata": "vectorstores/faiss_metadata.pkl",
        "paths.bm25_index": "vectorstores/bm25.pkl",
        "paths.bm25_metadata": "vectorstores/bm25_metadata.pkl",
        "models.embedding.name": "sentence-transformers/all-MiniLM-L6-v2",
        "models.reranker.name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "search.reciprocal_rank_k": 60,
        "document_processing.min_chunk_chars": 100,
        "prompts.simple_rag_system": "You are a helpful AI assistant. Context: {context}",
        "prompts.grade": "Grade this: {question} {context}",
        "prompts.rewrite": "Rewrite: {question}",
        "prompts.generate": "Answer: {question} {context}",
    }.get(key, default)
    config.get_path.side_effect = lambda key: Path({
        "paths.markdown": "data/markdown",
        "paths.chunks": "data/chunks",
        "paths.annotated_chunks": "data/processed/annotated_chunks.jsonl",
        "paths.vectors_file": "data/vectors.pkl",
        "embedding.checkpoint_dir": "data/vectors",
    }.get(key, "/tmp/default"))
    return config


@patch("app.SearchEngine")
def test_rag_application_init(MockSearchEngine, mock_config):
    from app import RAGApplication

    app = RAGApplication(mock_config)
    assert app.config is mock_config
    assert app.temp_upload_dir == Path("/tmp/test_rag_uploads")
    MockSearchEngine.assert_called_once()


@patch("app.SearchEngine")
def test_process_pdfs_no_files(MockSearchEngine, mock_config):
    from app import RAGApplication

    app = RAGApplication(mock_config)
    result = app.process_pdfs([], 512, "llama-3.3-70b-versatile", MagicMock(), MagicMock())
    assert result == "No files uploaded"


@patch("app.SearchEngine")
def test_chat_empty_message(MockSearchEngine, mock_config):
    from app import RAGApplication

    app = RAGApplication(mock_config)
    response, sources = app.chat("", "Simple RAG", "llama-3.3-70b-versatile", 0, 3)
    assert response == ""
    assert sources == ""


@patch("app.SearchEngine")
def test_chat_whitespace_message(MockSearchEngine, mock_config):
    from app import RAGApplication

    app = RAGApplication(mock_config)
    response, sources = app.chat("   ", "Agentic RAG", "llama-3.3-70b-versatile", 0, 3)
    assert response == ""
    assert sources == ""


@patch("app.SearchEngine")
def test_chat_simple_rag_mode(MockSearchEngine, mock_config):
    from app import RAGApplication

    mock_search_instance = MagicMock()
    MockSearchEngine.return_value = mock_search_instance

    with patch("app.SimpleRAG") as MockSimpleRAG:
        mock_rag = MagicMock()
        mock_rag.query.return_value = "This is a test answer"
        MockSimpleRAG.return_value = mock_rag

        app = RAGApplication(mock_config)
        response, sources = app.chat(
            "What is ML?", "Simple RAG", "llama-3.3-70b-versatile", 0.0, 3
        )

        assert response == "This is a test answer"
        assert "Simple RAG" in sources
        MockSimpleRAG.assert_called_once()


@patch("app.SearchEngine")
def test_chat_agentic_rag_mode(MockSearchEngine, mock_config):
    from app import RAGApplication

    mock_search_instance = MagicMock()
    MockSearchEngine.return_value = mock_search_instance

    with patch("app.AgenticRAG") as MockAgenticRAG:
        mock_rag = MagicMock()
        mock_rag.query.return_value = "Agentic answer here"
        MockAgenticRAG.return_value = mock_rag

        app = RAGApplication(mock_config)
        response, sources = app.chat(
            "Explain transformers", "Agentic RAG", "llama-3.3-70b-versatile", 0.0, 3
        )

        assert response == "Agentic answer here"
        assert "Agentic RAG" in sources
        MockAgenticRAG.assert_called_once()


@patch("app.SearchEngine")
def test_chat_error_handling(MockSearchEngine, mock_config):
    from app import RAGApplication

    mock_search_instance = MagicMock()
    MockSearchEngine.return_value = mock_search_instance

    with patch("app.SimpleRAG") as MockSimpleRAG:
        MockSimpleRAG.side_effect = RuntimeError("LLM connection failed")

        app = RAGApplication(mock_config)
        response, sources = app.chat(
            "test query", "Simple RAG", "llama-3.3-70b-versatile", 0.0, 3
        )

        assert "Error" in response
        assert "LLM connection failed" in response
        assert sources == ""


@patch("app.SearchEngine")
def test_search_documents_empty_query(MockSearchEngine, mock_config):
    from app import RAGApplication

    app = RAGApplication(mock_config)
    result = app.search_documents("", 10)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


@patch("app.SearchEngine")
def test_search_documents_with_results(MockSearchEngine, mock_config):
    from app import RAGApplication

    mock_search = MagicMock()
    mock_search.bm25_store.search.return_value = [
        {"doc_name": "test.pdf", "text": "some content", "score": 0.9},
        {"doc_name": "test2.pdf", "text": "other content", "score": 0.7},
    ]
    MockSearchEngine.return_value = mock_search

    app = RAGApplication(mock_config)
    result = app.search_documents("machine learning", 10)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == ["doc_name", "text", "score"]


@patch("app.SearchEngine")
def test_search_documents_error_handling(MockSearchEngine, mock_config):
    from app import RAGApplication

    mock_search = MagicMock()
    mock_search.bm25_store.search.side_effect = Exception("Index corrupted")
    MockSearchEngine.return_value = mock_search

    app = RAGApplication(mock_config)
    result = app.search_documents("test query", 10)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


@patch("app.SearchEngine")
def test_get_system_stats_no_chunks_file(MockSearchEngine, mock_config):
    from app import RAGApplication

    # Point to non-existent file
    mock_config.get_path.side_effect = lambda key: Path("/tmp/nonexistent_file.jsonl")

    mock_search = MagicMock()
    mock_search.vector_store.index = None
    mock_search.bm25_store.metadata = None
    MockSearchEngine.return_value = mock_search

    app = RAGApplication(mock_config)
    stats = app.get_system_stats()

    assert stats["Total Documents"] == 0
    assert stats["Total Chunks"] == 0
    assert stats["FAISS Vectors"] == 0
    assert stats["BM25 Documents"] == 0


@patch("app.SearchEngine")
def test_get_system_stats_returns_dict(MockSearchEngine, mock_config):
    from app import RAGApplication

    mock_config.get_path.side_effect = lambda key: Path("/tmp/nonexistent.jsonl")
    mock_search = MagicMock()
    mock_search.vector_store.index = None
    mock_search.bm25_store.metadata = None
    MockSearchEngine.return_value = mock_search

    app = RAGApplication(mock_config)
    stats = app.get_system_stats()
    assert isinstance(stats, dict)
    assert "Total Documents" in stats
    assert "Total Chunks" in stats
    assert "FAISS Vectors" in stats
    assert "BM25 Documents" in stats


# ──────────────────────────────────────────────
# 3. Streamlit Import and Module Tests
# ──────────────────────────────────────────────

def test_app_module_imports_streamlit():
    """Verify app.py imports streamlit, not gradio."""
    source = Path("app.py").read_text()
    assert "import streamlit" in source
    assert "import gradio" not in source
    assert "import gr" not in source


def test_app_has_main_function():
    """Verify app.py has a main() entry point for streamlit."""
    source = Path("app.py").read_text()
    assert "def main():" in source


def test_app_uses_st_set_page_config():
    source = Path("app.py").read_text()
    assert "st.set_page_config" in source


def test_app_uses_st_tabs():
    source = Path("app.py").read_text()
    assert "st.tabs" in source


def test_app_uses_st_chat_input():
    source = Path("app.py").read_text()
    assert "st.chat_input" in source


def test_app_uses_st_chat_message():
    source = Path("app.py").read_text()
    assert "st.chat_message" in source


def test_app_uses_st_file_uploader():
    source = Path("app.py").read_text()
    assert "st.file_uploader" in source


def test_app_uses_session_state():
    source = Path("app.py").read_text()
    assert "st.session_state" in source


def test_app_has_cache_resource():
    source = Path("app.py").read_text()
    assert "@st.cache_resource" in source


# ──────────────────────────────────────────────
# 4. No Residual Gradio References
# ──────────────────────────────────────────────

def test_no_gradio_in_app_py():
    source = Path("app.py").read_text()
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.split("#")[0]  # ignore comments
        assert "gradio" not in stripped.lower(), f"Gradio reference found at app.py:{i}"


def test_no_gradio_in_config_yaml():
    source = Path("config.yaml").read_text()
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.split("#")[0]
        assert "gradio" not in stripped.lower(), f"Gradio reference found at config.yaml:{i}"


def test_no_gradio_in_pyproject_toml():
    source = Path("pyproject.toml").read_text()
    assert "gradio" not in source.lower(), "pyproject.toml still references gradio"


def test_streamlit_in_pyproject_toml():
    source = Path("pyproject.toml").read_text()
    assert "streamlit" in source.lower(), "pyproject.toml should list streamlit as dependency"


# ──────────────────────────────────────────────
# 5. Process PDFs with Streamlit progress objects
# ──────────────────────────────────────────────

@patch("app.SearchEngine")
@patch("app.VectorStoreIngester")
@patch("app.EmbeddingGenerator")
@patch("app.DocumentProcessor")
def test_process_pdfs_calls_progress(
    MockDocProcessor, MockEmbedGen, MockIngester, MockSearchEngine, mock_config
):
    from app import RAGApplication

    mock_search = MagicMock()
    MockSearchEngine.return_value = mock_search

    # Mock embedding generator
    mock_embed = MagicMock()
    mock_embed.process_chunks.return_value = (pd.DataFrame({"text": ["a", "b"]}), [[0.1], [0.2]])
    MockEmbedGen.return_value = mock_embed

    # Mock file uploads (Streamlit UploadedFile interface)
    mock_file = MagicMock()
    mock_file.name = "test.pdf"
    mock_file.getbuffer.return_value = b"%PDF-1.4 fake content"

    # Mock Streamlit progress/status widgets
    progress_bar = MagicMock()
    status_text = MagicMock()

    app = RAGApplication(mock_config)
    # Ensure vectorstores don't exist so pipeline runs
    with patch.object(app, "_vectorstores_exist", return_value=False):
        result = app.process_pdfs([mock_file], 512, "llama-3.3-70b-versatile", progress_bar, status_text)

    assert "Successfully processed" in result
    assert "1 PDF(s)" in result
    # Verify progress was updated
    progress_bar.progress.assert_any_call(0)
    progress_bar.progress.assert_any_call(20)
    progress_bar.progress.assert_any_call(50)
    progress_bar.progress.assert_any_call(80)
    progress_bar.progress.assert_any_call(100)
    # Verify status text was updated
    assert status_text.text.call_count >= 4


@patch("app.SearchEngine")
def test_process_pdfs_skips_when_vectorstores_exist(MockSearchEngine, mock_config):
    from app import RAGApplication

    mock_search = MagicMock()
    MockSearchEngine.return_value = mock_search

    mock_file = MagicMock()
    mock_file.name = "test.pdf"
    mock_file.getbuffer.return_value = b"%PDF-1.4 fake content"

    progress_bar = MagicMock()
    status_text = MagicMock()

    app = RAGApplication(mock_config)
    with patch.object(app, "_vectorstores_exist", return_value=True):
        result = app.process_pdfs([mock_file], 512, "llama-3.3-70b-versatile", progress_bar, status_text)

    assert "already exist" in result
    assert "skipped" in result.lower()
    progress_bar.progress.assert_any_call(100)
    status_text.text.assert_called()
