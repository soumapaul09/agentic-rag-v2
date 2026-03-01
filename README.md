# Agentic RAG

## Quick Start

Run the complete pipeline with:

```bash
bash run_pipeline.sh
```

This script executes the entire data pipeline in order:
1. Document Processing (`document_processor.py`)
2. Embedding Generation (`embed.py`)
3. Vector Store Ingestion (`ingest.py`)

After completion, you can run queries using:
- Simple RAG: `.venv/bin/python main.py rag --query 'your question'`
- Agentic RAG: `.venv/bin/python main.py agentic_rag --query 'your question'`
- Streamlit interface: `.venv/bin/streamlit run app.py`

## Data Preparation

Running `document_processor.py` processes PDFs and creates the following files:

1. Markdown files: `data/markdown/{pdf_stem}.md` - one for each PDF
2. Chunk files: `data/chunks/{pdf_stem}.json` - one for each PDF
3. Consolidated chunks: `data/processed/annotated_chunks.jsonl` - all chunks combined

The script processes PDFs from the `pdfs/` directory, converts them to markdown, chunks them, and creates a final JSONL file with all annotated chunks.

## Embedding

Running `embed.py` creates embeddings from the annotated chunks:

1. Batch checkpoint files: `data/vectors/{index}.pkl` - intermediate vectors saved per batch during processing
2. Final vectors file: `data/vectors.pkl` - all embeddings saved as a single file

The script reads from `data/processed/annotated_chunks.jsonl` and uses sentence-transformers to create embeddings.

## Ingestion

Running `ingest.py` creates vector stores from embeddings and chunks:

1. FAISS index: `vectorstores/faiss.index` - vector search index
2. FAISS metadata: `vectorstores/faiss_metadata.pkl` - associated metadata for vectors
3. BM25 index: `vectorstores/bm25.pkl` - keyword search index
4. BM25 metadata: `vectorstores/bm25_metadata.pkl` - associated metadata for BM25

The script reads from `data/processed/annotated_chunks.jsonl` and `data/vectors.pkl` to create searchable indexes.

## Search Utilities

`search_utils.py` provides search functionality:

- `FAISSVectorStore` - manages FAISS vector similarity search
- `BM25Store` - manages BM25 keyword-based search
- `hybrid_search()` - combines BM25 and vector search with cross-encoder reranking
- `vector_search()` - pure vector similarity search
- `bm25_search()` - pure keyword search

## RAG Implementations

### Simple RAG

`simple_rag.py` implements a basic RAG pipeline using LangGraph:

- Search node: retrieves top-k documents using hybrid search
- Answer node: generates response using retrieved context
- Uses OpenAI GPT-4.1 for answer generation

### Agentic RAG

`agentic_rag.py` implements an advanced RAG pipeline with self-correction capabilities:

**Workflow:**
1. `generate_query_or_respond` - LLM decides whether to search or respond directly (tool calling)
2. `retrieve` - executes hybrid search if LLM chose to search
3. `grade_documents` - evaluates if retrieved documents are relevant to the question
   - If relevant → proceed to `generate_answer`
   - If not relevant → proceed to `rewrite_question`
4. `rewrite_question` - reformulates the query to improve retrieval
5. Loop back to step 1 with the rewritten question

**Key features:**
- Self-reflection: grades retrieved documents before generating answers
- Query rewriting: automatically refines questions when retrieval fails
- Tool-based retrieval: LLM autonomously decides when to search
- Recursion handling: catches infinite loops with GraphRecursionError

## Evaluation

`evaluate.py` provides local evaluation using open-source models:

- `LocalCorrectness` - judges answer correctness using google/flan-t5-base
- `evaluate_rag()` - compares generated answers against ground truth
- Returns boolean score and reasoning for the judgment