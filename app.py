from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
from loguru import logger

from agentic_rag import AgenticRAG
from config_loader import get_config
from document_processor import DocumentProcessor
from embed import EmbeddingGenerator
from ingest import VectorStoreIngester
from search_utils import SearchEngine
from simple_rag import SimpleRAG


class RAGApplication:
    def __init__(self, config):
        self.config = config
        self.temp_upload_dir = Path(config.get("paths.temp_uploads"))
        self.temp_upload_dir.mkdir(exist_ok=True)

        self.search_engine = SearchEngine(
            faiss_index_path=config.get("paths.faiss_index"),
            faiss_metadata_path=config.get("paths.faiss_metadata"),
            bm25_index_path=config.get("paths.bm25_index"),
            bm25_metadata_path=config.get("paths.bm25_metadata"),
            embedding_model_name=config.get("models.embedding.name"),
            reranker_model_name=config.get("models.reranker.name"),
            reciprocal_rank_k=config.get("search.reciprocal_rank_k"),
        )

    def _vectorstores_exist(self) -> bool:
        """Check if all vector store files already exist on disk."""
        paths_to_check = [
            Path(self.config.get("paths.faiss_index")),
            Path(self.config.get("paths.faiss_metadata")),
            Path(self.config.get("paths.bm25_index")),
            Path(self.config.get("paths.bm25_metadata")),
        ]
        return all(p.exists() and p.stat().st_size > 0 for p in paths_to_check)

    def process_pdfs(
        self,
        files: List,
        chunk_size: int,
        model_name: str,
        progress_bar,
        status_text,
    ) -> str:
        if not files:
            return "No files uploaded"

        try:
            # Skip the entire ingestion pipeline if vector stores already exist
            if self._vectorstores_exist():
                progress_bar.progress(100)
                status_text.text("Vector stores already exist — skipping ingestion.")
                logger.info("Vectorstores already exist. Skipping ingestion pipeline.")
                return (
                    "Vector databases already exist. Ingestion skipped. "
                    "Delete the vectorstores/ directory to force re-ingestion."
                )

            status_text.text("Initializing...")
            progress_bar.progress(0)
            pdf_dir = self.temp_upload_dir / "pdfs"
            pdf_dir.mkdir(exist_ok=True)

            for uploaded_file in files:
                dest_path = pdf_dir / uploaded_file.name
                dest_path.write_bytes(uploaded_file.getbuffer())

            status_text.text("Converting PDFs to Markdown...")
            progress_bar.progress(20)
            processor = DocumentProcessor(
                llm_model=model_name,
                chunk_size=chunk_size,
                min_chunk_chars=self.config.get("document_processing.min_chunk_chars"),
            )
            processor.process_documents(
                source_dir=pdf_dir,
                markdown_dir=self.config.get_path("paths.markdown"),
                chunk_dir=self.config.get_path("paths.chunks"),
                final_output=self.config.get_path("paths.annotated_chunks"),
            )

            status_text.text("Generating embeddings...")
            progress_bar.progress(50)
            embedding_gen = EmbeddingGenerator(
                model_name=self.config.get("models.embedding.name")
            )
            chunks_df, embeddings = embedding_gen.process_chunks(
                chunks_filepath=self.config.get("paths.annotated_chunks"),
                output_filepath=self.config.get_path("paths.vectors_file"),
                batch_size=self.config.get("embedding.batch_size"),
                checkpoint_dir=self.config.get_path("embedding.checkpoint_dir"),
            )

            status_text.text("Ingesting to vector stores...")
            progress_bar.progress(80)
            ingester = VectorStoreIngester(
                faiss_index_path=self.config.get("paths.faiss_index"),
                faiss_metadata_path=self.config.get("paths.faiss_metadata"),
                bm25_index_path=self.config.get("paths.bm25_index"),
                bm25_metadata_path=self.config.get("paths.bm25_metadata"),
            )
            ingester.ingest_to_faiss(
                chunks_filepath=self.config.get("paths.annotated_chunks"),
                vectors_filepath=self.config.get("paths.vectors_file"),
            )
            ingester.ingest_to_bm25(
                chunks_filepath=self.config.get("paths.annotated_chunks")
            )

            progress_bar.progress(100)
            status_text.text("Complete!")
            return f"Successfully processed {len(files)} PDF(s) into {len(chunks_df)} chunks"

        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            return f"Error: {str(e)}"

    def chat(
        self,
        message: str,
        rag_mode: str,
        model_name: str,
        temperature: float,
        top_k: int,
    ) -> Tuple[str, str]:
        if not message.strip():
            return "", ""

        try:
            search_engine = SearchEngine(
                faiss_index_path=self.config.get("paths.faiss_index"),
                faiss_metadata_path=self.config.get("paths.faiss_metadata"),
                bm25_index_path=self.config.get("paths.bm25_index"),
                bm25_metadata_path=self.config.get("paths.bm25_metadata"),
                embedding_model_name=self.config.get("models.embedding.name"),
                reranker_model_name=self.config.get("models.reranker.name"),
                reciprocal_rank_k=self.config.get("search.reciprocal_rank_k"),
            )

            if rag_mode == "Simple RAG":
                simple_rag = SimpleRAG(
                    model_name=model_name,
                    temperature=temperature,
                    top_k=top_k,
                    search_engine=search_engine,
                    system_prompt=self.config.get("prompts.simple_rag_system"),
                )
                response = simple_rag.query(message)
                sources = "Simple RAG - No detailed source tracking"
            else:
                agentic_rag = AgenticRAG(
                    model_name=model_name,
                    temperature=temperature,
                    search_engine=search_engine,
                    grade_prompt=self.config.get("prompts.grade"),
                    rewrite_prompt=self.config.get("prompts.rewrite"),
                    generate_prompt=self.config.get("prompts.generate"),
                )
                logger.info(f"AgenticRAG query: {message}")
                response = agentic_rag.query(message)
                logger.info(f"AgenticRAG response: {response}")
                sources = "Agentic RAG - Includes document grading and query rewriting"

            return response, sources

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {str(e)}", ""

    def search_documents(self, query: str, top_k: int) -> pd.DataFrame:
        if not query.strip():
            return pd.DataFrame()

        try:
            bm25_results = self.search_engine.bm25_store.search(query, top_k)
            df = pd.DataFrame(bm25_results)
            return df[["doc_name", "text", "score"]] if not df.empty else pd.DataFrame()
        except Exception as e:
            logger.error(f"Search error: {e}")
            return pd.DataFrame()

    def get_system_stats(self) -> dict:
        try:
            chunks_file = self.config.get_path("paths.annotated_chunks")
            if chunks_file.exists():
                df = pd.read_json(chunks_file, lines=True)
                total_chunks = len(df)
                total_docs = df["doc_name"].nunique()
            else:
                total_chunks = 0
                total_docs = 0

            faiss_vectors = (
                self.search_engine.vector_store.index.ntotal
                if self.search_engine.vector_store.index
                else 0
            )
            bm25_docs = (
                len(self.search_engine.bm25_store.metadata)
                if self.search_engine.bm25_store.metadata
                else 0
            )

            return {
                "Total Documents": total_docs,
                "Total Chunks": total_chunks,
                "FAISS Vectors": faiss_vectors,
                "BM25 Documents": bm25_docs,
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}


@st.cache_resource
def get_app():
    config = get_config()
    return RAGApplication(config)


def main():
    config = get_config()
    st_config = config.get("streamlit", {})
    ui_config = st_config.get("ui", {}) if isinstance(st_config, dict) else {}
    defaults = st_config.get("defaults", {}) if isinstance(st_config, dict) else {}

    st.set_page_config(
        page_title=ui_config.get("title", "Agentic RAG System"),
        page_icon="🤖",
        layout="wide",
    )

    app = get_app()

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("🤖 Agentic RAG System")
    st.caption("Upload PDFs, process them, and chat with your documents using advanced RAG techniques")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Document Management",
        "💬 Chat Interface",
        "🔍 Document Explorer",
        "📊 System Status",
    ])

    # ── Tab 1: Document Management ──
    with tab1:
        st.subheader("Upload and Process PDFs")

        col_upload, col_status = st.columns([2, 1])

        with col_upload:
            pdf_files = st.file_uploader(
                "Upload PDFs",
                type=["pdf"],
                accept_multiple_files=True,
            )

            chunk_size_range = defaults.get("chunk_size_range", [128, 1024, 128])
            chunk_size = st.slider(
                "Chunk Size",
                min_value=chunk_size_range[0],
                max_value=chunk_size_range[1],
                value=defaults.get("chunk_size", 512),
                step=chunk_size_range[2],
            )

            model_choices = [config.get("models.llm.default")] + config.get(
                "models.llm.alternatives"
            )
            llm_model = st.selectbox(
                "LLM Model for Processing",
                options=model_choices,
            )

            process_btn = st.button("🚀 Process Documents", type="primary", use_container_width=True)

        with col_status:
            st.markdown("**Status**")
            status_placeholder = st.empty()
            progress_placeholder = st.empty()

        if process_btn and pdf_files:
            progress_bar = progress_placeholder.progress(0)
            status_text = status_placeholder
            result = app.process_pdfs(
                pdf_files, chunk_size, llm_model, progress_bar, status_text
            )
            if result.startswith("Error"):
                st.error(f"❌ {result}")
            else:
                st.success(f"✅ {result}")
        elif process_btn and not pdf_files:
            st.warning("No files uploaded")

    # ── Tab 2: Chat Interface ──
    with tab2:
        st.subheader("Chat with Your Documents")

        chat_col, config_col = st.columns([3, 1])

        with config_col:
            st.markdown("### Configuration")
            rag_mode = st.radio(
                "RAG Mode",
                options=["Simple RAG", "Agentic RAG"],
                index=1,
            )

            model_choices = [config.get("models.llm.default")] + config.get(
                "models.llm.alternatives"
            )
            chat_model = st.selectbox(
                "Chat Model",
                options=model_choices,
                key="chat_model",
            )

            temp_range = defaults.get("temperature_range", [0, 1, 0.1])
            temperature = st.slider(
                "Temperature",
                min_value=float(temp_range[0]),
                max_value=float(temp_range[1]),
                value=float(defaults.get("temperature", 0)),
                step=float(temp_range[2]),
            )

            top_k_range = defaults.get("top_k_range", [1, 20, 1])
            top_k_chat = st.slider(
                "Top-K Results",
                min_value=top_k_range[0],
                max_value=top_k_range[1],
                value=defaults.get("top_k", 3),
                step=top_k_range[2],
            )

            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        with chat_col:
            # Display chat history
            for msg_role, msg_content in st.session_state.chat_history:
                with st.chat_message(msg_role):
                    st.markdown(msg_content)

            # Chat input
            user_input = st.chat_input("Ask a question about your documents...")

            if user_input:
                # Display user message immediately
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Get response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response, sources = app.chat(
                            user_input, rag_mode, chat_model, temperature, top_k_chat
                        )
                    st.markdown(response)
                    if sources:
                        st.caption(f"_{sources}_")

                # Update history
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", response))

    # ── Tab 3: Document Explorer ──
    with tab3:
        st.subheader("Search and Browse Documents")

        search_col, topk_col, btn_col = st.columns([3, 1, 1])

        with search_col:
            search_query = st.text_input(
                "Search Query",
                placeholder="Enter keywords to search...",
            )

        with topk_col:
            search_top_k_range = defaults.get("search_top_k_range", [5, 50, 5])
            search_top_k = st.slider(
                "Results",
                min_value=search_top_k_range[0],
                max_value=search_top_k_range[1],
                value=defaults.get("search_top_k", 10),
                step=search_top_k_range[2],
            )

        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.button("🔍 Search", type="primary", use_container_width=True)

        if search_btn and search_query:
            with st.spinner("Searching..."):
                results_df = app.search_documents(search_query, search_top_k)
            if not results_df.empty:
                results_df.columns = ["Document", "Text", "Score"]
                st.dataframe(results_df, use_container_width=True)
            else:
                st.info("No results found.")
        elif search_btn:
            st.warning("Please enter a search query.")

    # ── Tab 4: System Status ──
    with tab4:
        st.subheader("Vector Store Statistics")

        if st.button("🔄 Refresh Stats", type="primary"):
            st.cache_resource.clear()

        stats = app.get_system_stats()
        if stats:
            cols = st.columns(len(stats))
            for col, (label, value) in zip(cols, stats.items()):
                col.metric(label, value)

            st.json(stats)
        else:
            st.info("No statistics available. Process some documents first.")

    # ── Footer ──
    st.divider()
    st.markdown(
        """
        ### 📚 How to Use
        1. **Document Management**: Upload PDFs and process them into the vector store
        2. **Chat Interface**: Ask questions and get AI-powered answers with context
        3. **Document Explorer**: Search through your document chunks
        4. **System Status**: Monitor your vector store statistics
        """
    )


if __name__ == "__main__":
    main()
