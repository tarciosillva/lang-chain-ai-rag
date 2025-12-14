import logging
import shutil
from pathlib import Path
from typing import List

import nltk
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger", quiet=True)

settings = Settings()

CHROMA_PATH = Path(settings.chroma_path)
DATA_PATH = Path("data/books")


def load_documents() -> List[Document]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_PATH}")
    
    loader = DirectoryLoader(
        str(DATA_PATH),
        glob="*.pdf",
        show_progress=True
    )
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def split_text(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    if chunks:
        logger.debug(f"Sample chunk metadata: {chunks[0].metadata}")
    
    return chunks


def save_to_chroma(chunks: List[Document]) -> None:
    if CHROMA_PATH.exists():
        logger.info(f"Removing existing ChromaDB at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    
    embedding_function = OpenAIEmbeddings(api_key=settings.openai_api_key)
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=str(CHROMA_PATH)
    )
    
    db.persist()
    logger.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


def generate_data_store() -> None:
    try:
        documents = load_documents()
        if not documents:
            logger.warning("No documents found to process")
            return
        
        chunks = split_text(documents)
        save_to_chroma(chunks)
        logger.info("Data store generation completed successfully")
    except Exception as e:
        logger.error(f"Error generating data store: {e}", exc_info=True)
        raise


def main() -> None:
    generate_data_store()


if __name__ == "__main__":
    main()
