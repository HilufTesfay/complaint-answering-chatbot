import logging
from typing import List, Dict, Any
from chrom import ChromaVectorStore


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


logger = setup_logging()
chroma_client = ChromaVectorStore()


def retrieve_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve documents from the vector database based on a query."""
    try:
        embed_query = chroma_client.embed_query(query)
        results = chroma_client.query(
            collection_name="complaints", query=embed_query, top_k=top_k
        )
        return results
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []
