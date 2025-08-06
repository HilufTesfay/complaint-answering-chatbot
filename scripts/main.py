import pandas as pd
import os
import sys
import logging
from src.vector_db.chrom import process_to_chroma
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def setup_logging() -> logging.Logger:
    """Set up logging configuration"""
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "main.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

def main() -> None:
    """Main function to run the ChromaDB processing pipeline"""
    chunk_size = 1000000
    data_path = os.path.join(root_dir, "data", "processed", "complaints_cleaned.csv")
    if not os.path.exists(data_path):
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