import pandas as pd
import sys
import logging
from src.vector_db.chrom import process_to_chroma
from src.utils.path import get_project_root
from src.utils.logger import setup_logging

setup_logging("main")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting application")
    """Main function to run the ChromaDB processing pipeline"""
    chunk_size = 1000
    root_dir = get_project_root()
    data_path = root_dir / "data" / "processed" / "complaints_cleaned.csv"
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        print(f"Data file not found: {data_path}")
        return
    try:
        df = pd.read_csv(data_path, chunksize=chunk_size)
        for chunk in df:
            result = process_to_chroma(chunk)
            logger.info(f"ChromaDB processing completed successfully")
            logger.info(f"Processed {len(result['documents'])} documents")

    except Exception as e:
        logger.error(f"Error occurred: {e}")


if __name__ == "__main__":
    main()