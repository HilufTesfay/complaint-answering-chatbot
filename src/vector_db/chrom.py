import chromadb
from chromadb.config import Settings
from typing import List, Dict
import pandas as pd
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up the root directory for the project
root_dir = Path(__file__).resolve().parent.parent
# Configure logging
log_dir = root_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_dir / "chroma_db.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    def __init__(self, persist_dir: str = "../../vector_store/chromadb"):
        self.persist_dir = str(Path(persist_dir).resolve())
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=100, length_function=len
        )
        self.collection = self.client.get_or_create_collection(
            name="customer_complaints",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB initialized at {self.persist_dir}")

    def chunk_and_embed(self, df: pd.DataFrame) -> List[Dict]:
        # Validate required columns
        required_columns = ["Consumer complaint narrative", "Product"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns in DataFrame: {missing_columns}")
            raise ValueError(f"Required columns missing: {missing_columns}")

        documents = []
        metadatas = []
        ids = []

        for idx, row in df.iterrows():
            chunks = self.text_splitter.split_text(row["Consumer complaint narrative"])
            for chunk_num, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append(
                    {
                        "product": row["Product"],
                        "complaint_id": str(row.get("Complaint ID", "")),
                        "chunk_num": chunk_num,
                        "source": "CFPB",
                    }
                )
                ids.append(f"id_{idx}_{chunk_num}")

        logger.info(f"Created {len(documents)} chunks from {len(df)} complaints")
        return documents, metadatas, ids

    def add_to_collection(
        self, documents: List[str], metadatas: List[Dict], ids: List[str]
    ):
        # Generate embeddings with progress bar for large datasets
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list,
        )
        logger.info(
            f"Added {len(documents)} items to collection. Total: {self.collection.count()}"
        )

    def query_collection(
        self, query: str, n_results: int = 5, product_filter: str = None
    ):
        query_embedding = self.embedding_model.encode(query).tolist()
        filters = {"product": product_filter} if product_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        # Handle empty results
        if not results["documents"] or not results["documents"][0]:
            logger.info(f"No results found for query: '{query}'")
            return {"documents": [], "metadatas": [], "distances": []}

        logger.info(f"Query: '{query}' returned {len(results['documents'][0])} results")
        return results


def process_to_chroma(df: pd.DataFrame, chunk_size: int = 1000000) -> dict:
    """End-to-end ChromaDB processing pipeline with chunking"""
    try:
        chroma_db = ChromaVectorStore()
        total_rows = len(df)
        for start_idx in range(0, total_rows, chunk_size):
            batch_df = df[start_idx : start_idx + chunk_size]
            logger.info(
                f"Processing chunk {start_idx // chunk_size + 1} ({len(batch_df)} rows)"
            )
            documents, metadatas, ids = chroma_db.chunk_and_embed(batch_df)
            chroma_db.add_to_collection(documents, metadatas, ids)

        test_results = chroma_db.query_collection(
            "unauthorized credit card charges", product_filter="Credit card"
        )
        logger.info("ChromaDB processing completed successfully")
        return test_results

    except Exception as e:
        logger.error(f"Chroma processing failed: {str(e)}", exc_info=True)
        raise
