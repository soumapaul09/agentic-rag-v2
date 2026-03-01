#!/bin/bash

# Agentic RAG Pipeline Runner
# This script runs the complete data pipeline in the correct order:
# 1. Document Processing (PDFs -> Markdown -> Chunks)
# 2. Embedding Generation (Chunks -> Vectors)
# 3. Vector Store Ingestion (Vectors -> FAISS & BM25 indexes)
# 4. Run Simple RAG or Agentic RAG query

set -e  # Exit on any error

# Get script directory and activate virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found at $VENV_PYTHON"
    echo "Please create it first with: uv sync"
    exit 1
fi

echo "========================================="
echo "Starting Agentic RAG Pipeline"
echo "========================================="

# Phase 1: Document Processing
echo ""
echo "[1/3] Processing PDFs to chunks..."
"$VENV_PYTHON" document_processor.py
if [ $? -ne 0 ]; then
    echo "Error: Document processing failed"
    exit 1
fi
echo "✓ Document processing complete"

# Phase 2: Embedding Generation
echo ""
echo "[2/3] Generating embeddings from chunks..."
"$VENV_PYTHON" embed.py
if [ $? -ne 0 ]; then
    echo "Error: Embedding generation failed"
    exit 1
fi
echo "✓ Embedding generation complete"

# Phase 3: Vector Store Ingestion
echo ""
echo "[3/3] Ingesting embeddings into vector stores..."
"$VENV_PYTHON" ingest.py
if [ $? -ne 0 ]; then
    echo "Error: Vector store ingestion failed"
    exit 1
fi
echo "✓ Vector store ingestion complete"

echo ""
echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="
echo ""
echo "Vector stores ready at:"
echo "  - FAISS: vectorstores/faiss.index"
echo "  - BM25: vectorstores/bm25.pkl"
echo ""
echo "You can now run queries using:"
echo "  .venv/bin/python main.py rag --query 'your question'"
echo "  .venv/bin/python main.py agentic_rag --query 'your question'"
echo "  streamlit run app.py  # for Streamlit interface"
