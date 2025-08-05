import chromadb
from chromadb.config import Settings
from typing import List, Dict
import pandas as pd
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    filename="../week-6/logs/chroma_db.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    def __init__(self, persist_dir: str = "../../vector_store/chromadb"):
        # Initialize Chroma client with persistent storage
        self.client = chromadb.PersistentClient(path=persist_dir)
        # Configure text splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=100, length_function=len
        )

        # Create/load collection
        self.collection = self.client.get_or_create_collection(
            name="customer_complaints",
            metadata={"hnsw:space": "cosine"},  # Optimized for semantic search

        )

        logger.info(f"ChromaDB initialized at {persist_dir}")

    def chunk_and_embed(self, df: pd.DataFrame) -> List[Dict]:
        """Process DataFrame into chunks with metadata"""
        documents = []
        metadatas = []
        ids = []

        for idx, row in df.iterrows():
            chunks = self.text_splitter.split_text(row["cleaned_narrative"])

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
        """Add documents to Chroma collection with embeddings"""
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents)

        # Convert to Chroma format
        embeddings_list = [embedding.tolist() for embedding in embeddings]

        # Upsert into Chroma
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list,
        )

        # Persist to disk
        self.client.persist()
        logger.info(
            f"Added {len(documents)} items to collection. Total: {self.collection.count()}"
        )

    def query_collection(
        self, query: str, n_results: int = 5, product_filter: str = None
    ):
        """Query the vector store"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build filters
        filters = {}
        if product_filter:
            filters = {"product": product_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        logger.info(f"Query: '{query}' returned {len(results['documents'][0])} results")
        return results


def process_to_chroma(
    cleaned_data_path: str = "../../data/raw/complaints.csv",
):
    """End-to-end ChromaDB processing pipeline"""
    try:
        # Initialize Chroma
        chroma_db = ChromaVectorStore()

        # Load cleaned data
        df = pd.read_csv(cleaned_data_path)
        logger.info(f"Loaded {len(df)} cleaned complaints")

        # Process and embed
        documents, metadatas, ids = chroma_db.chunk_and_embed(df)
        chroma_db.add_to_collection(documents, metadatas, ids)

        # Test query
        test_results = chroma_db.query_collection(
            "unauthorized credit card charges", product_filter="Credit card"
        )

        logger.info("ChromaDB processing completed successfully")
        return test_results

    except Exception as e:
        logger.error(f"Chroma processing failed: {str(e)}")
        raise
