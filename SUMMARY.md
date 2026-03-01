# Agentic RAG v2 - Project Summary

## 1. What This Codebase Does

Agentic RAG v2 is an end-to-end Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents, process them into searchable vector stores, and then ask natural language questions that are answered using the content of those documents. The system is built in Python, uses Groq-hosted LLMs (Llama 3.3, Mixtral) for inference, and is served through a Streamlit web interface.

The core idea behind RAG is simple: instead of relying solely on an LLM's training data, the system first **retrieves** relevant passages from your own documents, then **generates** an answer grounded in that retrieved context. This makes responses more accurate and specific to your data.

---

## 2. System Architecture and Pipeline

The system operates in two phases: **Data Ingestion** (offline, run once) and **Query/Chat** (online, per user question).

### Phase 1: Data Ingestion Pipeline

The ingestion pipeline is orchestrated by `run_pipeline.sh` and consists of three steps:

**Step 1 - Document Processing (`document_processor.py`)**
PDFs are converted to Markdown using `MarkItDown` (powered by a Groq LLM), then chunked into smaller text segments using `RecursiveChunker` from the `chonkie` library. The default chunk size is 512 tokens. All chunks are consolidated into a single JSONL file (`annotated_chunks.jsonl`) with metadata tracking the source document name and chunk index.

**Step 2 - Embedding Generation (`embed.py`)**
Each text chunk is converted into a dense vector (numerical representation) using the `sentence-transformers/all-MiniLM-L6-v2` model. Embeddings are generated in batches with checkpointing, so the process can resume if interrupted. The final embeddings are saved as a pickle file.

**Step 3 - Vector Store Ingestion (`ingest.py`)**
Two search indexes are created from the processed data:
- **FAISS Index**: A vector similarity search index that finds documents by semantic meaning (what the text *means*).
- **BM25 Index**: A keyword-based search index that finds documents by exact term matching (what words the text *contains*).

### Phase 2: Query and Chat

**Search (`search_utils.py`)**
When a user asks a question, the `SearchEngine` class runs a hybrid search that combines both FAISS (semantic) and BM25 (keyword) results in parallel using thread pooling. The combined results are then re-ranked using a Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) to produce the most relevant passages.

**Simple RAG (`simple_rag.py`)**
A straightforward two-step LangGraph pipeline: (1) search for relevant documents, (2) pass the retrieved context and user question to the LLM to generate an answer. No self-correction or iteration.

**Agentic RAG (`agentic_rag.py`)**
An advanced LangGraph pipeline with a self-correcting feedback loop:
1. The LLM decides whether it needs to search for documents or can answer directly (tool calling).
2. If it searches, retrieved documents are **graded for relevance** by a second LLM call.
3. If documents are deemed irrelevant, the LLM **rewrites the question** and retries the search.
4. This loop continues until relevant documents are found or a recursion limit (25 iterations) is reached.

**Streamlit App (`app.py`)**
The web UI has four tabs:
- **Document Management**: Upload PDFs and trigger the full ingestion pipeline.
- **Chat Interface**: Ask questions with configurable RAG mode (Simple vs Agentic), model selection, temperature, and top-k settings.
- **Document Explorer**: Keyword search through indexed document chunks.
- **System Status**: View vector store statistics (total documents, chunks, FAISS vectors, BM25 documents).

### Configuration (`config.yaml`)
All paths, model names, prompt templates, search parameters, and UI defaults are centralized in a single YAML configuration file, loaded via `config_loader.py`.

---

## 3. Key Technologies Used

| Component          | Technology                                |
|--------------------|-------------------------------------------|
| LLM Inference      | Groq API (Llama 3.3 70B, Mixtral 8x7B)   |
| Embeddings         | sentence-transformers (all-MiniLM-L6-v2)  |
| Reranking          | Cross-Encoder (ms-marco-MiniLM-L-6-v2)    |
| Vector Search      | FAISS (Facebook AI Similarity Search)      |
| Keyword Search     | BM25 (rank-bm25 library)                  |
| Orchestration      | LangGraph (LangChain)                     |
| PDF Conversion     | MarkItDown                                |
| Text Chunking      | Chonkie (RecursiveChunker)                 |
| Web UI             | Streamlit                                 |
| Package Management | uv (pyproject.toml)                        |

---

---

# How to Commit and Push a Codebase from Local Machine to GitHub Using Git CLI

## Prerequisites

1. **Git** installed on your machine (`brew install git` on Mac, or download from git-scm.com).
2. **GitHub CLI (gh)** installed (`brew install gh`) and authenticated (`gh auth login`).
3. A **GitHub account**.

---

## Step 1: Initialize a Git Repository

Navigate to your project folder and initialize git:

```bash
cd /path/to/your/project
git init
```

This creates a hidden `.git/` directory that tracks all changes.

---

## Step 2: Create a `.gitignore` File

Before adding files, create a `.gitignore` to exclude files that should NOT go to GitHub (large files, secrets, build artifacts):

```
# Example .gitignore
__pycache__/
*.pyc
.venv/
.env
data/
*.pkl
.DS_Store
```

This prevents accidentally pushing sensitive API keys, large data files, or virtual environments.

---

## Step 3: Stage Your Files

Add files to the staging area (tells git which files to include in the next commit):

```bash
# Add specific files
git add file1.py file2.py README.md

# Or add everything (respects .gitignore)
git add .
```

Check what's staged:

```bash
git status
```

---

## Step 4: Commit Your Changes

Create a commit (a snapshot of your staged files with a message):

```bash
git commit -m "Initial commit: project description here"
```

Good commit messages are concise and describe *why* the change was made, not just *what* changed.

---

## Step 5: Create a GitHub Repository and Push

**Option A: Using GitHub CLI (recommended)**

This single command creates the remote repo and pushes in one step:

```bash
# Public repo
gh repo create my-project-name --public --source=. --push

# Private repo
gh repo create my-project-name --private --source=. --push
```

**Option B: Manual approach**

1. Create a repo on github.com (click "New Repository").
2. Copy the repo URL and add it as a remote:

```bash
git remote add origin https://github.com/your-username/your-repo-name.git
```

3. Push your code:

```bash
git push -u origin main
```

The `-u` flag sets the upstream so future pushes only need `git push`.

---

## Step 6: Ongoing Workflow (Making Changes)

After the initial push, the daily workflow for making changes is:

```bash
# 1. Make your code changes

# 2. Check what changed
git status
git diff

# 3. Stage the changed files
git add changed_file.py another_file.py

# 4. Commit with a descriptive message
git commit -m "Fix: corrected search ranking logic"

# 5. Push to GitHub
git push
```

---

## Common Troubleshooting

| Problem | Solution |
|---------|----------|
| Push rejected (secrets detected) | Remove the file from git history with `git filter-branch` or `git filter-repo`, add it to `.gitignore`, and push again. Rotate any exposed API keys. |
| Permission denied on push | Run `gh auth login` to re-authenticate, or check your SSH keys with `ssh -T git@github.com`. |
| Accidentally committed a large file | Add it to `.gitignore`, remove from tracking with `git rm --cached <file>`, commit, and push. |
| Merge conflicts | Pull latest changes with `git pull`, resolve conflicts in the marked files, then `git add` and `git commit`. |

---

## Quick Reference Cheat Sheet

```bash
git init                          # Initialize a new repo
git add .                         # Stage all files
git commit -m "message"           # Commit staged changes
git status                        # Check repo status
git log --oneline                 # View commit history
git push                          # Push to remote
git pull                          # Pull latest from remote
gh repo create name --public      # Create GitHub repo via CLI
gh auth login                     # Authenticate with GitHub
```
