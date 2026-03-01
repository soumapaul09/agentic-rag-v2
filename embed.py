from itertools import batched
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config_loader import get_config

load_dotenv()


class EmbeddingGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        logger.info(f"Initialized embedding model: {model_name}")

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        return [text.replace("\n", " ") for text in texts]

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        processed_texts = self._preprocess_texts(texts)
        embeddings = self.embedding_model.encode(processed_texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, texts: List[str]) -> List[float]:
        processed_texts = self._preprocess_texts(texts)
        embeddings = self.embedding_model.encode(processed_texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings[0].tolist()

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int,
        checkpoint_dir: Path
    ) -> List[np.ndarray]:
        all_embeddings = []
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx, text_batch in tqdm(
            enumerate(batched(texts, batch_size)),
            desc=f"Embedding texts with {self.model_name}",
            total=len(texts) // batch_size,
            unit="batch",
        ):
            batch_embeddings = self.embedding_model.encode(
                list(text_batch),
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.extend(batch_embeddings.tolist())

            checkpoint_file = checkpoint_dir / f"{batch_idx}.pkl"
            joblib.dump(all_embeddings, checkpoint_file)

        return all_embeddings

    def process_chunks(
        self,
        chunks_filepath: str,
        output_filepath: Path,
        batch_size: int,
        checkpoint_dir: Path,
    ) -> tuple[pd.DataFrame, List[List[float]]]:
        chunks_df = pd.read_json(chunks_filepath, lines=True)
        text_data = chunks_df["text"].tolist()

        logger.info(f"Processing {len(text_data)} chunks from {chunks_filepath}")
        embeddings = self.embed_batch(text_data, batch_size=batch_size, checkpoint_dir=checkpoint_dir)

        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(embeddings, output_filepath)
        logger.info(f"Saved {len(embeddings)} embeddings to {output_filepath}")

        return chunks_df, embeddings


if __name__ == "__main__":
    config = get_config()

    generator = EmbeddingGenerator(
        model_name=config.get("models.embedding.name")
    )

    chunks_df, embeddings = generator.process_chunks(
        chunks_filepath=config.get("paths.annotated_chunks"),
        output_filepath=config.get_path("paths.vectors_file"),
        batch_size=config.get("embedding.batch_size"),
        checkpoint_dir=config.get_path("embedding.checkpoint_dir")
    )