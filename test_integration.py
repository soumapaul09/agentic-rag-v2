"""System integration tests for the Streamlit-based Agentic RAG application.

Tests the real end-to-end flow with actual vectorstores, models, and LLM calls.
Requires: GROQ_API_KEY in .env, pre-built vectorstores in vectorstores/
"""

import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

load_dotenv()

# Reset config singleton between test files
import config_loader
config_loader._config = None


# ──────────────────────────────────────────────
# 1. Config Integration
# ──────────────────────────────────────────────

class TestConfigIntegration:
    def setup_method(self):
        config_loader._config = None
        self.config = config_loader.get_config()

    def test_config_yaml_loads_with_all_sections(self):
        required_sections = ["paths", "models", "document_processing", "embedding",
                             "search", "rag", "prompts", "streamlit"]
        for section in required_sections:
            assert self.config.get(section) is not None, f"Missing config section: {section}"

    def test_all_vectorstore_paths_exist(self):
        paths_to_check = [
            "paths.faiss_index", "paths.faiss_metadata",
            "paths.bm25_index", "paths.bm25_metadata",
        ]
        for path_key in paths_to_check:
            p = Path(self.config.get(path_key))
            assert p.exists(), f"{path_key} -> {p} does not exist"

    def test_annotated_chunks_file_exists(self):
        p = Path(self.config.get("paths.annotated_chunks"))
        assert p.exists(), "annotated_chunks.jsonl does not exist"
        df = pd.read_json(p, lines=True)
        assert len(df) > 0, "annotated_chunks.jsonl is empty"
        assert "text" in df.columns
        assert "doc_name" in df.columns


# ──────────────────────────────────────────────
# 2. SearchEngine Integration (loads real indexes)
# ──────────────────────────────────────────────

class TestSearchEngineIntegration:
    @pytest.fixture(autouse=True)
    def setup(self):
        config_loader._config = None
        config = config_loader.get_config()
        from search_utils import SearchEngine
        self.engine = SearchEngine(
            faiss_index_path=config.get("paths.faiss_index"),
            faiss_metadata_path=config.get("paths.faiss_metadata"),
            bm25_index_path=config.get("paths.bm25_index"),
            bm25_metadata_path=config.get("paths.bm25_metadata"),
            embedding_model_name=config.get("models.embedding.name"),
            reranker_model_name=config.get("models.reranker.name"),
            reciprocal_rank_k=config.get("search.reciprocal_rank_k"),
        )

    def test_faiss_index_loaded(self):
        assert self.engine.vector_store.index is not None
        assert self.engine.vector_store.index.ntotal > 0
        print(f"  FAISS vectors loaded: {self.engine.vector_store.index.ntotal}")

    def test_bm25_index_loaded(self):
        assert self.engine.bm25_store.bm25 is not None
        assert self.engine.bm25_store.metadata is not None
        assert len(self.engine.bm25_store.metadata) > 0
        print(f"  BM25 documents loaded: {len(self.engine.bm25_store.metadata)}")

    def test_bm25_search_returns_results(self):
        results = self.engine.bm25_store.search("machine learning", top_k=5)
        assert len(results) == 5
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)
        assert all("doc_name" in r for r in results)
        print(f"  BM25 top result doc: {results[0]['doc_name']}, score: {results[0]['score']:.4f}")

    def test_vector_search_returns_results(self):
        result_text = self.engine.vector_search("What is a transformer model?", top_k=3)
        assert isinstance(result_text, str)
        assert len(result_text) > 0
        print(f"  Vector search result length: {len(result_text)} chars")

    def test_hybrid_search_returns_results(self):
        result_text = self.engine.hybrid_search("explain attention mechanism", top_k=3)
        assert isinstance(result_text, str)
        assert len(result_text) > 100, "Hybrid search result too short"
        print(f"  Hybrid search result length: {len(result_text)} chars")

    def test_hybrid_search_different_queries(self):
        queries = [
            "logistic regression",
            "neural network architecture",
            "prompt engineering techniques",
        ]
        for q in queries:
            result = self.engine.hybrid_search(q, top_k=3)
            assert isinstance(result, str) and len(result) > 0, f"Empty result for query: {q}"


# ──────────────────────────────────────────────
# 3. Embedding Generator Integration
# ──────────────────────────────────────────────

class TestEmbeddingIntegration:
    @pytest.fixture(autouse=True)
    def setup(self):
        config_loader._config = None
        config = config_loader.get_config()
        from embed import EmbeddingGenerator
        self.generator = EmbeddingGenerator(
            model_name=config.get("models.embedding.name")
        )

    def test_embed_single_query(self):
        embedding = self.generator.embed_query(["What is deep learning?"])
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
        print(f"  Embedding dimension: {len(embedding)}")

    def test_embed_multiple_texts(self):
        texts = ["Hello world", "Machine learning is great", "Transformers are powerful"]
        embeddings = self.generator.embed_texts(texts)
        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    def test_embedding_similarity(self):
        import numpy as np
        e1 = np.array(self.generator.embed_query(["deep learning neural networks"]))
        e2 = np.array(self.generator.embed_query(["machine learning algorithms"]))
        e3 = np.array(self.generator.embed_query(["cooking pasta recipes"]))

        sim_related = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        sim_unrelated = np.dot(e1, e3) / (np.linalg.norm(e1) * np.linalg.norm(e3))

        assert sim_related > sim_unrelated, "Related texts should be more similar"
        print(f"  Related similarity: {sim_related:.4f}, Unrelated: {sim_unrelated:.4f}")


# ──────────────────────────────────────────────
# 4. RAGApplication Integration (with real search)
# ──────────────────────────────────────────────

class TestRAGApplicationIntegration:
    @pytest.fixture(autouse=True)
    def setup(self):
        config_loader._config = None
        self.config = config_loader.get_config()
        from app import RAGApplication
        self.app = RAGApplication(self.config)

    def test_search_documents_returns_dataframe(self):
        df = self.app.search_documents("transformer", 5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["doc_name", "text", "score"]
        print(f"  Search returned {len(df)} results")
        print(f"  Top doc: {df.iloc[0]['doc_name']}, score: {df.iloc[0]['score']:.4f}")

    def test_search_documents_different_top_k(self):
        for k in [3, 10, 20]:
            df = self.app.search_documents("neural network", k)
            assert len(df) == k, f"Expected {k} results, got {len(df)}"

    def test_get_system_stats_returns_populated_stats(self):
        stats = self.app.get_system_stats()
        assert isinstance(stats, dict)
        assert stats["Total Documents"] > 0
        assert stats["Total Chunks"] > 0
        assert stats["FAISS Vectors"] > 0
        assert stats["BM25 Documents"] > 0
        print(f"  Stats: {stats}")

    def test_system_stats_consistency(self):
        stats = self.app.get_system_stats()
        assert stats["Total Chunks"] == stats["FAISS Vectors"], \
            "Chunk count should match FAISS vector count"
        assert stats["Total Chunks"] == stats["BM25 Documents"], \
            "Chunk count should match BM25 document count"


# ──────────────────────────────────────────────
# 5. LLM Chat Integration (actual Groq API calls)
# ──────────────────────────────────────────────

class TestLLMChatIntegration:
    @pytest.fixture(autouse=True)
    def setup(self):
        import os
        if not os.environ.get("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not set")
        config_loader._config = None
        self.config = config_loader.get_config()
        from app import RAGApplication
        self.app = RAGApplication(self.config)

    def test_simple_rag_query(self):
        response, sources = self.app.chat(
            message="What is a transformer model?",
            rag_mode="Simple RAG",
            model_name="llama-3.3-70b-versatile",
            temperature=0,
            top_k=3,
        )
        assert isinstance(response, str)
        assert len(response) > 20, f"Response too short: {response}"
        assert "Error" not in response, f"Got error: {response}"
        assert "Simple RAG" in sources
        print(f"  Simple RAG response ({len(response)} chars): {response[:150]}...")

    def test_agentic_rag_query(self):
        response, sources = self.app.chat(
            message="Explain the attention mechanism in neural networks",
            rag_mode="Agentic RAG",
            model_name="llama-3.3-70b-versatile",
            temperature=0,
            top_k=3,
        )
        assert isinstance(response, str)
        assert len(response) > 20, f"Response too short: {response}"
        assert "Error" not in response, f"Got error: {response}"
        assert "Agentic RAG" in sources
        print(f"  Agentic RAG response ({len(response)} chars): {response[:150]}...")

    def test_simple_rag_with_different_model(self):
        response, sources = self.app.chat(
            message="What is logistic regression?",
            rag_mode="Simple RAG",
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            top_k=2,
        )
        assert isinstance(response, str)
        assert len(response) > 20
        assert "Error" not in response
        print(f"  llama-3.1-8b response ({len(response)} chars): {response[:150]}...")


# ──────────────────────────────────────────────
# 6. Streamlit App Rendering (smoke test)
# ──────────────────────────────────────────────

class TestStreamlitAppSmoke:
    def test_app_module_importable(self):
        """Verify the app module can be imported without errors."""
        import app
        assert hasattr(app, "main")
        assert hasattr(app, "get_app")
        assert hasattr(app, "RAGApplication")

    def test_streamlit_is_importable(self):
        import streamlit as st
        assert hasattr(st, "tabs")
        assert hasattr(st, "chat_input")
        assert hasattr(st, "chat_message")
        assert hasattr(st, "file_uploader")
        assert hasattr(st, "set_page_config")
        assert hasattr(st, "session_state")
        assert hasattr(st, "cache_resource")

    def test_no_gradio_import_possible_from_app(self):
        """Ensure app.py doesn't try to import gradio at all."""
        source = Path("app.py").read_text()
        import_lines = [l for l in source.splitlines() if l.strip().startswith(("import ", "from "))]
        for line in import_lines:
            assert "gradio" not in line.lower(), f"Gradio import found: {line}"

    def test_app_source_has_all_four_tabs(self):
        source = Path("app.py").read_text()
        assert "Document Management" in source
        assert "Chat Interface" in source
        assert "Document Explorer" in source
        assert "System Status" in source

    def test_config_has_streamlit_not_gradio(self):
        config_loader._config = None
        config = config_loader.get_config()
        assert config.get("streamlit") is not None
        assert config.get("gradio") is None

    def test_pyproject_has_streamlit_dependency(self):
        content = Path("pyproject.toml").read_text()
        assert "streamlit" in content
        assert "gradio" not in content


# ──────────────────────────────────────────────
# 7. End-to-End Pipeline Validation
# ──────────────────────────────────────────────

class TestEndToEndValidation:
    @pytest.fixture(autouse=True)
    def setup(self):
        config_loader._config = None
        self.config = config_loader.get_config()

    def test_full_data_pipeline_artifacts_exist(self):
        """Verify all artifacts from the data pipeline are present."""
        artifacts = {
            "Annotated chunks": self.config.get("paths.annotated_chunks"),
            "Vectors pickle": self.config.get("paths.vectors_file"),
            "FAISS index": self.config.get("paths.faiss_index"),
            "FAISS metadata": self.config.get("paths.faiss_metadata"),
            "BM25 index": self.config.get("paths.bm25_index"),
            "BM25 metadata": self.config.get("paths.bm25_metadata"),
        }
        for name, path_str in artifacts.items():
            p = Path(path_str)
            assert p.exists(), f"{name} not found at {p}"
            assert p.stat().st_size > 0, f"{name} is empty at {p}"
            print(f"  {name}: {p} ({p.stat().st_size / 1024 / 1024:.1f} MB)")

    def test_chunk_count_matches_vector_count(self):
        """Chunks, FAISS vectors, and BM25 docs should all be the same count."""
        chunks_df = pd.read_json(self.config.get("paths.annotated_chunks"), lines=True)
        chunk_count = len(chunks_df)

        from search_utils import FAISSVectorStore, BM25Store
        faiss_store = FAISSVectorStore(
            index_path=self.config.get("paths.faiss_index"),
            metadata_path=self.config.get("paths.faiss_metadata"),
        )
        bm25_store = BM25Store(
            metadata_path=self.config.get("paths.bm25_metadata"),
            bm25_index_path=self.config.get("paths.bm25_index"),
        )

        assert faiss_store.index.ntotal == chunk_count, \
            f"FAISS has {faiss_store.index.ntotal} vectors but {chunk_count} chunks exist"
        assert len(bm25_store.metadata) == chunk_count, \
            f"BM25 has {len(bm25_store.metadata)} docs but {chunk_count} chunks exist"
        print(f"  All counts match: {chunk_count} chunks = {faiss_store.index.ntotal} vectors = {len(bm25_store.metadata)} BM25 docs")

    def test_pdfs_directory_has_source_files(self):
        pdf_dir = Path(self.config.get("paths.pdfs"))
        assert pdf_dir.exists()
        pdfs = list(pdf_dir.glob("*.pdf"))
        assert len(pdfs) > 0, "No PDFs found in source directory"
        print(f"  Source PDFs: {len(pdfs)} files")
