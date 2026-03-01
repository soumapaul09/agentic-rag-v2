import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config_loader import get_config
from embed import EmbeddingGenerator

load_dotenv()


class FAISSVectorStore:
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index = None
        self.metadata = None
        self.load()

    def load(self):
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            logger.warning("FAISS index or metadata not found. Please create them first.")

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved FAISS index to {self.index_path}")

    def create_from_embeddings(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata = metadata.to_dict("records")
        self.save()

    def search(self, query_vector: np.ndarray, top_k: int):
        if self.index is None:
            raise ValueError("Index not loaded. Call load() first.")

        query_vector = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(dist)
                results.append(result)

        return results


class BM25Store:
    def __init__(self, metadata_path: str, bm25_index_path: str):
        self.metadata_path = Path(metadata_path)
        self.bm25_index_path = Path(bm25_index_path)
        self.metadata = None
        self.bm25 = None
        self.load()

    def load(self):
        if self.metadata_path.exists() and self.bm25_index_path.exists():
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            with open(self.bm25_index_path, "rb") as f:
                self.bm25 = pickle.load(f)
            logger.info(f"Loaded BM25 index with {len(self.metadata)} documents")
        else:
            logger.warning("BM25 index or metadata not found. Please create them first.")

    def save(self):
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        with open(self.bm25_index_path, "wb") as f:
            pickle.dump(self.bm25, f)
        logger.info(f"Saved BM25 index to {self.bm25_index_path}")

    def create_from_texts(self, texts: List[str], metadata: pd.DataFrame):
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.metadata = metadata.to_dict("records")
        self.save()

    def search(self, query: str, top_k: int):
        if self.bm25 is None:
            raise ValueError("BM25 index not loaded. Call load() first.")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(scores[idx])
                results.append(result)

        return results


class SearchEngine:
    def __init__(
        self,
        faiss_index_path: str,
        faiss_metadata_path: str,
        bm25_index_path: str,
        bm25_metadata_path: str,
        embedding_model_name: str,
        reranker_model_name: str,
        reciprocal_rank_k: int,
    ):
        self.vector_store = FAISSVectorStore(
            index_path=faiss_index_path,
            metadata_path=faiss_metadata_path,
        )
        self.bm25_store = BM25Store(
            metadata_path=bm25_metadata_path,
            bm25_index_path=bm25_index_path,
        )
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model_name)
        self.reranker = CrossEncoder(reranker_model_name)
        self.reciprocal_rank_k = reciprocal_rank_k

    def _sentence_transformer_rerank(
        self, query: str, documents: List[str], top_k: int
    ) -> List[str]:
        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.predict(pairs)

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in ranked_indices]

    def _reciprocal_rank_fusion(self, result_lists):
        scores, all_results = {}, {}
        for results in result_lists:
            for rank, item in enumerate(results, start=1):
                item_id = item.get("id", str(hash(item["text"])))
                scores[item_id] = scores.get(item_id, 0) + 1.0 / (self.reciprocal_rank_k + rank)
                all_results[item_id] = item

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [all_results[doc_id] for doc_id, score in sorted_results]

    def hybrid_search(self, query: str, top_k: int) -> str:
        """
        Search the database for the most relevant documents using a hybrid approach.
        """
        query_vector = self.embedding_generator.embed_query([query])

        with ThreadPoolExecutor() as executor:
            bm25_future = executor.submit(self.bm25_store.search, query, top_k)
            vector_future = executor.submit(self.vector_store.search, query_vector, top_k)

            bm25_results = bm25_future.result()
            vector_results = vector_future.result()

            bm25_texts = [r["text"] for r in bm25_results]
            vector_texts = [r["text"] for r in vector_results]

            unique_texts = list(set(bm25_texts + vector_texts))
            results = self._sentence_transformer_rerank(query, unique_texts, top_k)
            return "\n\n".join(results)

    def vector_search(self, query: str, top_k: int) -> str:
        query_vector = self.embedding_generator.embed_query([query])
        results = self.vector_store.search(query_vector, top_k)
        texts = [r["text"] for r in results]
        return "\n\n".join(texts)

    def bm25_search(self, query: str, top_k: int) -> str:
        results = self.bm25_store.search(query, top_k)
        texts = [r["text"] for r in results]
        return "\n\n".join(texts)


if __name__ == "__main__":
    config = get_config()

    # Default search engine singleton instance
    search_engine = SearchEngine(
        faiss_index_path=config.get("paths.faiss_index"),
        faiss_metadata_path=config.get("paths.faiss_metadata"),
        bm25_index_path=config.get("paths.bm25_index"),
        bm25_metadata_path=config.get("paths.bm25_metadata"),
        embedding_model_name=config.get("models.embedding.name"),
        reranker_model_name=config.get("models.reranker.name"),
        reciprocal_rank_k=config.get("search.reciprocal_rank_k")
    )

    # Module-level convenience references for backward compatibility
    vector_store = search_engine.vector_store
    bm25_store = search_engine.bm25_store
    embedding_generator = search_engine.embedding_generator