import json
from pathlib import Path

import pandas as pd
from chonkie import RecursiveChunker
from dotenv import load_dotenv
from loguru import logger
from markitdown import MarkItDown
from groq import Groq
from tqdm.auto import tqdm

from config_loader import get_config

load_dotenv()


class DocumentProcessor:
    def __init__(self, llm_model: str, chunk_size: int, min_chunk_chars: int):
        self.llm_client = Groq()
        self.markdown_converter = MarkItDown(llm_client=self.llm_client, llm_model=llm_model)
        self.text_chunker = RecursiveChunker(
            chunk_size=chunk_size,
            min_characters_per_chunk=min_chunk_chars,
            return_type="texts",
        ).from_recipe("markdown", lang="en")

    def convert_to_markdown(self, source_file: Path, destination_dir: Path):
        output_file = destination_dir / f"{source_file.stem}.md"
        if not output_file.exists():
            try:
                conversion_result = self.markdown_converter.convert(source_file)
            except Exception as e:
                logger.error(f"Error converting {source_file}: {e}. Skipping!")
                return
            with open(output_file, "w") as f:
                f.write(conversion_result.text_content)

    def chunk_markdown(self, source_file: Path, chunk_dir: Path, markdown_dir: Path):
        chunk_output = chunk_dir / f"{source_file.stem}.json"
        if not chunk_output.exists():
            with open(chunk_output, "w") as f:
                markdown_source = Path(markdown_dir / f"{source_file.stem}.md")
                content = open(markdown_source).read()
                text_chunks = [chunk.text for chunk in self.text_chunker(content)]
                f.write(json.dumps(text_chunks, indent=4))

    def consolidate_chunks(self, chunk_dir: Path, final_output: Path):
        if final_output.exists():
            logger.info(f"Skipping {final_output} because it already exists")
            return

        all_chunks = []
        for chunk_file in chunk_dir.rglob("*.json"):
            with open(chunk_file) as f:
                try:
                    chunk_data = json.loads(f.read())
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing {chunk_file}: {e}. Skipping!")
                    continue
                all_chunks.extend(
                    [
                        {
                            "doc_name": chunk_file.stem,
                            "text": text_content,
                            "chunk_index": idx,
                        }
                        for idx, text_content in enumerate(chunk_data)
                    ]
                )

        chunks_df = pd.DataFrame(all_chunks)
        chunks_df.to_json(final_output, orient="records", lines=True)
        logger.info(f"Created {len(chunks_df)} chunks and saved to {final_output}")

    def process_documents(self, source_dir: Path, markdown_dir: Path, chunk_dir: Path, final_output: Path):
        source_dir, markdown_dir, chunk_dir, final_output = (
            Path(source_dir),
            Path(markdown_dir),
            Path(chunk_dir),
            Path(final_output),
        )

        markdown_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        final_output.parent.mkdir(parents=True, exist_ok=True)

        source_files = list(source_dir.glob("*.pdf"))

        if not source_files:
            logger.warning(f"No PDF files found in {source_dir}")
            return

        logger.info(f"Found {len(source_files)} PDF files in {source_dir}")

        with tqdm(source_files, desc="Processing PDFs", unit="file") as progress:
            for pdf_file in progress:
                progress.set_postfix(file=pdf_file.name, refresh=False)
                self.convert_to_markdown(pdf_file, markdown_dir)
                self.chunk_markdown(pdf_file, chunk_dir, markdown_dir)

        self.consolidate_chunks(chunk_dir=chunk_dir, final_output=final_output)


if __name__ == "__main__":
    config = get_config()

    processor = DocumentProcessor(
        llm_model=config.get("document_processing.default_llm"),
        chunk_size=config.get("document_processing.chunk_size"),
        min_chunk_chars=config.get("document_processing.min_chunk_chars")
    )

    processor.process_documents(
        source_dir=config.get_path("paths.pdfs"),
        markdown_dir=config.get_path("paths.markdown"),
        chunk_dir=config.get_path("paths.chunks"),
        final_output=config.get_path("paths.annotated_chunks")
    )